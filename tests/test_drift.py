"""
Тесты для src/monitoring/drift_metrics.py.

Зависят только от numpy/pandas/scipy/sklearn - без evidently и mlflow,
чтобы CI не тянул тяжёлые зависимости только для проверки математики.
"""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_metrics import (
    compute_data_drift,
    compute_quality,
    compute_target_drift,
    detect_categorical_drift,
    detect_concept_drift,
    detect_numerical_drift,
    js_divergence,
    kl_divergence,
    population_stability_index,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def test_psi_zero_for_identical_distributions(rng):
    data = rng.normal(loc=0, scale=1, size=5000)
    psi = population_stability_index(data, data.copy())
    assert psi < 0.01


def test_psi_high_for_shifted_distribution(rng):
    reference = rng.normal(loc=0, scale=1, size=5000)
    current = rng.normal(loc=5, scale=1, size=5000)  # сильный сдвиг
    psi = population_stability_index(reference, current)
    assert psi > 0.25


def test_kl_divergence_zero_for_identical(rng):
    data = rng.normal(size=3000)
    kl = kl_divergence(data, data.copy())
    assert kl < 0.05


def test_js_divergence_is_bounded_and_symmetric(rng):
    reference = rng.normal(loc=0, size=3000)
    current = rng.normal(loc=3, size=3000)
    js_forward = js_divergence(reference, current)
    js_backward = js_divergence(current, reference)
    assert js_forward == pytest.approx(js_backward, rel=1e-6)
    assert 0 <= js_forward <= np.log(2) + 1e-6


def test_detect_numerical_drift_flags_shift(rng):
    reference = pd.Series(rng.normal(loc=0, size=2000), name="amount")
    current = pd.Series(rng.normal(loc=4, size=2000), name="amount")
    result = detect_numerical_drift(reference, current)
    assert result.drift_detected is True
    assert result.psi > 0


def test_detect_numerical_drift_no_drift_for_same_distribution(rng):
    reference = pd.Series(rng.normal(loc=0, size=2000), name="amount")
    current = pd.Series(rng.normal(loc=0, size=2000), name="amount")
    result = detect_numerical_drift(reference, current)
    assert result.drift_detected is False


def test_detect_categorical_drift_flags_shift():
    reference = pd.Series(["a"] * 800 + ["b"] * 200, name="merchant_category")
    current = pd.Series(["a"] * 200 + ["b"] * 800, name="merchant_category")
    result = detect_categorical_drift(reference, current)
    assert result.drift_detected is True


def test_compute_data_drift_aggregates_features(rng):
    reference = pd.DataFrame(
        {
            "amount": rng.normal(size=1000),
            "hour": rng.normal(size=1000),
            "category": rng.choice(["a", "b"], size=1000),
        }
    )
    current = pd.DataFrame(
        {
            "amount": rng.normal(loc=5, size=1000),  # дрейф
            "hour": rng.normal(size=1000),  # без дрейфа
            "category": rng.choice(["a", "b"], size=1000),
        }
    )
    result = compute_data_drift(
        reference, current, numerical_features=["amount", "hour"], categorical_features=["category"]
    )
    assert result.n_features == 3
    assert result.n_drifted_features >= 1


def test_compute_target_drift(rng):
    reference_scores = rng.uniform(0, 0.1, size=2000)
    current_scores = rng.uniform(0.4, 0.6, size=2000)
    result = compute_target_drift(reference_scores, current_scores)
    assert result.drift_detected is True


def test_compute_quality_good_separation():
    y_true = np.array([0] * 900 + [1] * 100)
    y_score = np.array([0.01] * 900 + [0.99] * 100)
    quality = compute_quality(y_true, y_score, threshold=0.5)
    assert quality.pr_auc > 0.9
    assert quality.precision > 0.9
    assert quality.recall > 0.9


def test_detect_concept_drift_flags_quality_drop():
    reference_quality = compute_quality(
        np.array([0] * 900 + [1] * 100),
        np.array([0.01] * 900 + [0.99] * 100),
    )
    current_quality = compute_quality(
        np.array([0] * 900 + [1] * 100),
        np.random.default_rng(1).uniform(0, 1, size=1000),  # случайные скоры — модель "сломалась"
    )
    result = detect_concept_drift(reference_quality, current_quality, pr_auc_drop_threshold=0.05)
    assert result.concept_drift_detected is True
