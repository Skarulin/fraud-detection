"""
Prometheus-экспортёр метрик дрейфа для fraud-detection.

Поднимает HTTP-сервер на отдельном порту (по умолчанию 9200), который
Prometheus скрейпит независимо от FastAPI-сервиса коллеги. Метрики
обновляются вызовом update_*_metrics(...) после каждого расчёта дрейфа
в generate_report.py.
"""
from __future__ import annotations

import logging
import time

from prometheus_client import Gauge, start_http_server

from src.monitoring.drift_metrics import ConceptDriftResult, DatasetDriftResult, FeatureDriftResult

logger = logging.getLogger(__name__)

# --- Метрики уровня датасета -------------------------------------------------
DATA_DRIFT_SHARE = Gauge(
    "fraud_data_drift_share",
    "Доля признаков с обнаруженным дрейфом (0..1)",
)
DATA_DRIFT_DETECTED = Gauge(
    "fraud_data_drift_detected",
    "1, если по датасету в целом зафиксирован дрейф, иначе 0",
)
N_DRIFTED_FEATURES = Gauge(
    "fraud_n_drifted_features",
    "Количество признаков с обнаруженным дрейфом",
)

# --- Метрики на уровне отдельного признака (с лейблом feature) --------------
FEATURE_PSI = Gauge(
    "fraud_feature_psi",
    "PSI по конкретному признаку",
    labelnames=("feature",),
)
FEATURE_KS_PVALUE = Gauge(
    "fraud_feature_ks_pvalue",
    "p-value теста Колмогорова-Смирнова по признаку (только числовые)",
    labelnames=("feature",),
)

# --- Target / prediction drift ----------------------------------------------
TARGET_DRIFT_PSI = Gauge(
    "fraud_target_drift_psi",
    "PSI по распределению предсказанных моделью вероятностей",
)
TARGET_DRIFT_DETECTED = Gauge(
    "fraud_target_drift_detected",
    "1, если зафиксирован дрейф предсказаний модели, иначе 0",
)

# --- Concept drift proxy / качество модели -----------------------------------
MODEL_PR_AUC = Gauge(
    "fraud_model_pr_auc",
    "PR-AUC модели на батче",
    labelnames=("batch",),
)
CONCEPT_DRIFT_DETECTED = Gauge(
    "fraud_concept_drift_detected",
    "1, если зафиксирована деградация качества между батчами (proxy для concept drift)",
)
CONCEPT_DRIFT_PR_AUC_DROP = Gauge(
    "fraud_concept_drift_pr_auc_drop",
    "Абсолютное падение PR-AUC между reference и current батчами",
)

# --- Служебная метрика -------------------------------------------------------
LAST_RUN_TIMESTAMP = Gauge(
    "fraud_drift_last_run_timestamp_seconds",
    "Unix-таймстемп последнего успешного расчёта дрейфа",
)


def update_data_drift_metrics(result: DatasetDriftResult, share_threshold: float = 0.3) -> None:
    DATA_DRIFT_SHARE.set(result.drift_share)
    DATA_DRIFT_DETECTED.set(1 if result.dataset_drift_detected(share_threshold) else 0)
    N_DRIFTED_FEATURES.set(result.n_drifted_features)

    for feature_result in (*result.numerical_results, *result.categorical_results):
        FEATURE_PSI.labels(feature=feature_result.feature).set(feature_result.psi)
        if isinstance(feature_result, FeatureDriftResult):
            FEATURE_KS_PVALUE.labels(feature=feature_result.feature).set(feature_result.ks_p_value)


def update_target_drift_metrics(result: FeatureDriftResult) -> None:
    TARGET_DRIFT_PSI.set(result.psi)
    TARGET_DRIFT_DETECTED.set(1 if result.drift_detected else 0)


def update_concept_drift_metrics(result: ConceptDriftResult) -> None:
    MODEL_PR_AUC.labels(batch="reference").set(result.reference_quality.pr_auc)
    MODEL_PR_AUC.labels(batch="current").set(result.current_quality.pr_auc)
    CONCEPT_DRIFT_DETECTED.set(1 if result.concept_drift_detected else 0)
    CONCEPT_DRIFT_PR_AUC_DROP.set(result.pr_auc_drop)


def mark_run_complete() -> None:
    LAST_RUN_TIMESTAMP.set(time.time())


_server_started = False


def serve_metrics(port: int = 9200) -> None:
    """Поднять HTTP-сервер с метриками. Безопасно вызывать несколько раз — повторный запуск пропускается."""
    global _server_started
    if _server_started:
        return
    logger.info("Starting Prometheus metrics server on port %s", port)
    start_http_server(port)
    _server_started = True
