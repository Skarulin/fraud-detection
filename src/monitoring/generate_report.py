"""
CLI-скрипт расчёта дрейфа: data drift, target drift и concept drift (proxy).

Что делает:
1. Грузит модель НАПРЯМУЮ из локального CatBoost-файла (model.cb / model.cbm),
   а НЕ из MLflow Registry (там абсолютный Windows-путь, которого нет в
   контейнере, поэтому mlflow.catboost.load_model падал с "No such artifact").
2. На каждом запуске берёт НОВУЮ случайную выборку (seed не фиксирован),
   поэтому при --mode serve каждые --interval секунд считается дрейф на
   разных кусках датасета — метрики в Prometheus/Grafana реально меняются
   со временем, а не остаются одним и тем же числом.
3. Пишет models/drift.json (для UI, поля drift_detected/score) и
   reports/drift/latest_metrics.json (расширенная сводка).
4. Обновляет Prometheus-метрики на порту 9200.

Пример запуска:
    python -m src.monitoring.generate_report --mode once
    python -m src.monitoring.generate_report --mode once --sample 30000
    python -m src.monitoring.generate_report --mode serve --interval 300
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.monitoring import drift_metrics as dm
from src.monitoring import metrics_exporter as exporter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from utilits import split_feature_types as _split_raw

    def split_feature_types(df: pd.DataFrame):
        return _split_raw(df)

except ImportError:
    logger.warning("Не нашёл utilits — использую запасную реализацию по dtype.")

    def split_feature_types(df: pd.DataFrame):
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        continuous_cols = df.select_dtypes(include=["number"]).columns.tolist()
        return df.copy(), cat_cols, continuous_cols


DROP_COLS = ["transaction_id", "customer_id", "merchant_id"]
TARGET_COL = "is_fraud"
DATA_DIR = Path("data/raw")

UI_DRIFT_JSON = Path("models/drift.json")
FULL_METRICS_JSON = Path("reports/drift/latest_metrics.json")


def find_model_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        logger.warning("Указанный --model-path %s не найден, ищу автоматически", explicit)

    for c in (Path("models/model.cbm"), Path("models/model.cb")):
        if c.is_file():
            return c

    mlruns_models = sorted(Path("mlruns").rglob("model.cb"))
    if mlruns_models:
        return mlruns_models[-1]

    raise FileNotFoundError(
        "Не нашёл файл модели. Положи model.cb или model.cbm в папку models/ "
        "или укажи путь через --model-path."
    )


def load_model(model_path: Path) -> CatBoostClassifier:
    logger.info("Загружаю модель из локального файла: %s", model_path)
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def load_inference_artifacts() -> dict | None:
    path = Path("models/inference_artifacts.pkl")
    if path.is_file():
        with open(path, "rb") as f:
            return pickle.load(f)
    logger.warning("Не нашёл models/inference_artifacts.pkl — определю фичи автоматически.")
    return None


def load_and_split_data(reference_frac: float, sample: int, seed: int):
    """
    seed передаётся СНАРУЖИ и каждый раз новый (см. run_once) — поэтому
    каждый запуск берёт другой случайный кусок датасета на сэмплирование
    и другой порядок перемешивания при делении на reference/current.
    Это и даёт меняющиеся со временем метрики дрейфа в Prometheus.
    """
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Нет parquet-файлов в {DATA_DIR}. Положи туда части датасета."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logger.info("Загружено строк всего: %d", len(df))

    if sample and sample < len(df):
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(
            df,
            test_size=sample / len(df),
            stratify=df[TARGET_COL],
            random_state=seed,
        )
        df = df.reset_index(drop=True)
        logger.info(
            "Сэмплировано строк: %d (seed=%d, fraud: %d, %.2f%%)",
            len(df), seed, int(df[TARGET_COL].sum()), df[TARGET_COL].mean() * 100,
        )

    df = df.drop(columns=DROP_COLS, errors="ignore")
    df = df.fillna(-999)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    split_idx = int(len(df) * reference_frac)
    reference = df.iloc[:split_idx].reset_index(drop=True)
    current = df.iloc[split_idx:].reset_index(drop=True)
    logger.info("Reference: %d строк, current: %d строк", len(reference), len(current))
    return reference, current


def get_scores(model, X: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    pool = Pool(X, cat_features=cat_cols)
    return model.predict_proba(pool)[:, 1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Записан JSON: %s", path)


def run_once(args: argparse.Namespace) -> None:
    # Новый случайный seed на КАЖДЫЙ вызов run_once (а не один раз на старте
    # процесса) — поэтому в --mode serve каждая итерация цикла берёт другую
    # выборку, и метрики дрейфа в Prometheus реально "живут" во времени.
    seed = random.randint(0, 2**31 - 1)

    reference, current = load_and_split_data(
        reference_frac=args.reference_frac, sample=args.sample, seed=seed,
    )

    artifacts = load_inference_artifacts()

    ref_features = reference.drop(columns=[TARGET_COL])
    cur_features = current.drop(columns=[TARGET_COL])

    X_ref, cat_cols, continuous_cols = split_feature_types(ref_features)
    X_cur, _, _ = split_feature_types(cur_features)

    if artifacts and "feature_order" in artifacts:
        feature_order = artifacts["feature_order"]
        cat_cols = artifacts.get("cat_features", cat_cols)
        continuous_cols = [c for c in feature_order if c not in cat_cols]
        X_ref = X_ref.reindex(columns=feature_order)
        X_cur = X_cur.reindex(columns=feature_order)

    model = load_model(find_model_path(args.model_path))

    ref_scores = get_scores(model, X_ref, cat_cols)
    cur_scores = get_scores(model, X_cur, cat_cols)

    data_drift_result = dm.compute_data_drift(
        reference, current,
        numerical_features=continuous_cols,
        categorical_features=cat_cols,
        psi_threshold=args.psi_threshold,
    )
    exporter.update_data_drift_metrics(
        data_drift_result, share_threshold=args.drift_share_threshold
    )

    target_drift_result = dm.compute_target_drift(
        ref_scores, cur_scores, psi_threshold=args.psi_threshold,
    )
    exporter.update_target_drift_metrics(target_drift_result)

    ref_quality = dm.compute_quality(reference[TARGET_COL].values, ref_scores)
    cur_quality = dm.compute_quality(current[TARGET_COL].values, cur_scores)
    concept_drift_result = dm.detect_concept_drift(
        ref_quality, cur_quality, pr_auc_drop_threshold=args.pr_auc_drop_threshold,
    )
    exporter.update_concept_drift_metrics(concept_drift_result)
    exporter.mark_run_complete()

    dataset_drift_detected = data_drift_result.dataset_drift_detected(
        share_threshold=args.drift_share_threshold
    )

    ui_payload = {
        "drift_detected": bool(
            dataset_drift_detected
            or target_drift_result.drift_detected
            or concept_drift_result.concept_drift_detected
        ),
        "score": round(float(data_drift_result.drift_share), 4),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
    }
    write_json(UI_DRIFT_JSON, ui_payload)

    full_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "data_drift": {
            "drift_detected": bool(dataset_drift_detected),
            "drift_share": round(float(data_drift_result.drift_share), 4),
            "n_drifted_features": int(data_drift_result.n_drifted_features),
            "n_features": int(data_drift_result.n_features),
            "per_feature": [
                {
                    "feature": r.feature,
                    "psi": round(float(r.psi), 4),
                    "ks_p_value": round(float(r.ks_p_value), 4),
                    "wasserstein": round(float(r.wasserstein_distance), 4),
                    "drift_detected": bool(r.drift_detected),
                }
                for r in data_drift_result.numerical_results
            ] + [
                {
                    "feature": r.feature,
                    "psi": round(float(r.psi), 4),
                    "drift_detected": bool(r.drift_detected),
                }
                for r in data_drift_result.categorical_results
            ],
        },
        "target_drift": {
            "drift_detected": bool(target_drift_result.drift_detected),
            "psi": round(float(target_drift_result.psi), 4),
            "ks_p_value": round(float(target_drift_result.ks_p_value), 4),
            "wasserstein": round(float(target_drift_result.wasserstein_distance), 4),
        },
        "concept_drift": {
            "drift_detected": bool(concept_drift_result.concept_drift_detected),
            "pr_auc_drop": round(float(concept_drift_result.pr_auc_drop), 4),
            "reference_pr_auc": round(float(ref_quality.pr_auc), 4),
            "current_pr_auc": round(float(cur_quality.pr_auc), 4),
            "reference_recall": round(float(ref_quality.recall), 4),
            "current_recall": round(float(cur_quality.recall), 4),
        },
    }
    write_json(FULL_METRICS_JSON, full_payload)

    logger.info(
        "Готово. seed=%d  drift_detected=%s  data_drift_share=%.3f  target_psi=%.3f  "
        "ref_pr_auc=%.4f  cur_pr_auc=%.4f  pr_auc_drop=%.4f",
        seed,
        ui_payload["drift_detected"],
        data_drift_result.drift_share,
        target_drift_result.psi,
        ref_quality.pr_auc,
        cur_quality.pr_auc,
        concept_drift_result.pr_auc_drop,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Расчёт дрейфа для fraud-detection")
    parser.add_argument(
        "--model-path", default=None,
        help="Путь к CatBoost-модели (model.cb/.cbm). По умолчанию ищется автоматически.",
    )
    parser.add_argument("--reference-frac", type=float, default=0.7)
    parser.add_argument(
        "--sample", type=int, default=50000,
        help="Сколько строк брать из датасета на одну итерацию (выбираются случайно).",
    )
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--drift-share-threshold", type=float, default=0.3)
    parser.add_argument("--pr-auc-drop-threshold", type=float, default=0.05)
    parser.add_argument("--metrics-port", type=int, default=9200)
    parser.add_argument("--mode", choices=["once", "serve"], default="once")
    parser.add_argument("--interval", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter.serve_metrics(port=args.metrics_port)
    run_once(args)
    if args.mode == "serve":
        logger.info(
            "Serve-режим: интервал %d сек, порт %d. Каждая итерация — новая "
            "случайная выборка из датасета.",
            args.interval, args.metrics_port,
        )
        while True:
            time.sleep(args.interval)
            try:
                run_once(args)
            except Exception:
                logger.exception("Ошибка пересчёта, попробую в следующем интервале")


if __name__ == "__main__":
    main()
