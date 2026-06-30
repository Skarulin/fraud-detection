"""
Статистические метрики для расчёта дрейфа данных, дрейфа таргета и
индикатора деградации качества (concept drift proxy) для fraud-detection.

Модуль не зависит от Evidently/MLflow — только numpy/pandas/scipy/sklearn.
Это сделано специально, по тому же принципу, что и split_feature_types
в src/utils.py: тесты должны быть быстрыми и не требовать тяжёлых
зависимостей.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Базовые статистические метрики дрейфа (см. слайд "Расчёт")
# ---------------------------------------------------------------------------


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    PSI - Population Stability Index.

    Принятые на практике пороги:
    PSI < 0.1          - распределения практически не отличаются
    0.1 <= PSI < 0.25   - умеренный дрейф, стоит присмотреться
    PSI >= 0.25         - значимый дрейф
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        # Вырожденный случай: почти все значения в reference одинаковые
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    psi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(psi_values))


def psi_categorical(reference: pd.Series, current: pd.Series, epsilon: float = 1e-4) -> float:
    """PSI для категориальных признаков - по долям категорий."""
    categories = set(reference.unique()) | set(current.unique())
    ref_counts = reference.value_counts(normalize=True)
    cur_counts = current.value_counts(normalize=True)

    psi = 0.0
    for category in categories:
        ref_pct = ref_counts.get(category, 0.0) + epsilon
        cur_pct = cur_counts.get(category, 0.0) + epsilon
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(psi)


def kl_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
    """Расхождение Кульбака-Лейблера (несимметричное)."""
    epsilon = 1e-10
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    combined_min = min(reference.min(), current.min())
    combined_max = max(reference.max(), current.max())
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
    cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)

    ref_hist = ref_hist + epsilon
    cur_hist = cur_hist + epsilon
    ref_hist /= ref_hist.sum()
    cur_hist /= cur_hist.sum()

    return float(np.sum(ref_hist * np.log(ref_hist / cur_hist)))


def js_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
    """Расхождение Йенсена-Шеннона - симметричная версия KL."""
    epsilon = 1e-10
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    combined_min = min(reference.min(), current.min())
    combined_max = max(reference.max(), current.max())
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
    cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)

    ref_hist = ref_hist + epsilon
    cur_hist = cur_hist + epsilon
    ref_hist /= ref_hist.sum()
    cur_hist /= cur_hist.sum()

    mixture = 0.5 * (ref_hist + cur_hist)
    kl_ref = np.sum(ref_hist * np.log(ref_hist / mixture))
    kl_cur = np.sum(cur_hist * np.log(cur_hist / mixture))
    return float(0.5 * kl_ref + 0.5 * kl_cur)


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Тест Колмогорова-Смирнова. Возвращает (статистика, p-value)."""
    statistic, p_value = ks_2samp(reference, current)
    return float(statistic), float(p_value)


def wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Метрика Васерштейна (Earth Mover's Distance)."""
    return float(wasserstein_distance(reference, current))


# ---------------------------------------------------------------------------
# Дрейф по одному признаку - сводка метрик сразу
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    wasserstein_distance: float
    drift_detected: bool


def detect_numerical_drift(
    reference: pd.Series,
    current: pd.Series,
    psi_threshold: float = 0.2,
    ks_p_value_threshold: float = 0.05,
) -> FeatureDriftResult:
    psi_score = population_stability_index(reference.values, current.values)
    ks_stat, ks_p = ks_test(reference.values, current.values)
    w_dist = wasserstein(reference.values, current.values)

    drift_detected = (psi_score >= psi_threshold) or (ks_p < ks_p_value_threshold)

    return FeatureDriftResult(
        feature=reference.name or "unknown",
        psi=psi_score,
        ks_statistic=ks_stat,
        ks_p_value=ks_p,
        wasserstein_distance=w_dist,
        drift_detected=drift_detected,
    )


@dataclass
class CategoricalDriftResult:
    feature: str
    psi: float
    drift_detected: bool


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    psi_threshold: float = 0.2,
) -> CategoricalDriftResult:
    psi_score = psi_categorical(reference, current)
    return CategoricalDriftResult(
        feature=reference.name or "unknown",
        psi=psi_score,
        drift_detected=psi_score >= psi_threshold,
    )


# ---------------------------------------------------------------------------
# Дрейф по всему датасету (data drift) и по предсказаниям (target drift)
# ---------------------------------------------------------------------------


@dataclass
class DatasetDriftResult:
    numerical_results: list[FeatureDriftResult] = field(default_factory=list)
    categorical_results: list[CategoricalDriftResult] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.numerical_results) + len(self.categorical_results)

    @property
    def n_drifted_features(self) -> int:
        n = sum(r.drift_detected for r in self.numerical_results)
        n += sum(r.drift_detected for r in self.categorical_results)
        return n

    @property
    def drift_share(self) -> float:
        if self.n_features == 0:
            return 0.0
        return self.n_drifted_features / self.n_features

    def dataset_drift_detected(self, share_threshold: float = 0.3) -> bool:
        return self.drift_share >= share_threshold


def compute_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    psi_threshold: float = 0.2,
) -> DatasetDriftResult:
    """Дрейф входных данных (data drift) по всем переданным признакам."""
    numerical_results = [
        detect_numerical_drift(reference[col], current[col], psi_threshold=psi_threshold)
        for col in numerical_features
    ]
    categorical_results = [
        detect_categorical_drift(reference[col], current[col], psi_threshold=psi_threshold)
        for col in categorical_features
    ]
    return DatasetDriftResult(numerical_results=numerical_results, categorical_results=categorical_results)


def compute_target_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    psi_threshold: float = 0.2,
) -> FeatureDriftResult:
    """
    Target / Prediction drift - дрейф распределения вероятностей,
    которые выдаёт модель (predict_proba), а не самого таргета
    (он недоступен в момент инференса).
    """
    series_ref = pd.Series(reference_scores, name="prediction_score")
    series_cur = pd.Series(current_scores, name="prediction_score")
    return detect_numerical_drift(series_ref, series_cur, psi_threshold=psi_threshold)


# ---------------------------------------------------------------------------
# Concept drift proxy: деградация качества модели между батчами
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    pr_auc: float
    precision: float
    recall: float
    n_samples: int
    n_positive: int


def compute_quality(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> QualityResult:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return QualityResult(
        pr_auc=float(pr_auc),
        precision=float(precision),
        recall=float(recall),
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
    )


@dataclass
class ConceptDriftResult:
    reference_quality: QualityResult
    current_quality: QualityResult
    pr_auc_drop: float
    concept_drift_detected: bool


def detect_concept_drift(
    reference_quality: QualityResult,
    current_quality: QualityResult,
    pr_auc_drop_threshold: float = 0.05,
) -> ConceptDriftResult:
    """
    Прокси для concept drift: настоящий сдвиг P(Y|X) посчитать без
    размеченных продакшен-данных с лагом нельзя. Используем падение
    PR-AUC между батчами как сигнал: модель стала хуже разделять классы,
    значит подозреваем изменение зависимости между признаками и таргетом
    (а не только дрейф самого входа).
    """
    pr_auc_drop = reference_quality.pr_auc - current_quality.pr_auc
    return ConceptDriftResult(
        reference_quality=reference_quality,
        current_quality=current_quality,
        pr_auc_drop=float(pr_auc_drop),
        concept_drift_detected=pr_auc_drop >= pr_auc_drop_threshold,
    )
