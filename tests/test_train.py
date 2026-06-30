"""
Smoke-тесты для src/train.py
Запуск: pytest tests/ -v
"""

import sys
import os
import pandas as pd

# Добавляем src/ в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utilits import split_feature_types


# ============================================================
# Тесты split_feature_types
# ============================================================

def test_split_feature_types_continuous():
    """Числовой столбец с >2 уникальными значениями → continuous."""
    X = pd.DataFrame({'price': [1.0, 2.5, 3.7, 4.1, 5.0]})
    _, cat_cols, continuous_cols = split_feature_types(X)
    assert 'price' in continuous_cols
    assert 'price' not in cat_cols


def test_split_feature_types_categorical():
    """Строковый столбец → categorical."""
    X = pd.DataFrame({'city': ['Moscow', 'SPb', 'Kazan', 'Moscow', 'SPb']})
    _, cat_cols, continuous_cols = split_feature_types(X)
    assert 'city' in cat_cols
    assert 'city' not in continuous_cols


def test_split_feature_types_binary_int():
    """Бинарный int-столбец {0,1} → categorical."""
    X = pd.DataFrame({'flag': [0, 1, 0, 1, 1]})
    _, cat_cols, continuous_cols = split_feature_types(X)
    assert 'flag' in cat_cols


def test_split_feature_types_datetime():
    """Datetime-столбец → categorical (конвертируется в строку)."""
    X = pd.DataFrame({'ts': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])})
    X_out, cat_cols, _ = split_feature_types(X)
    assert 'ts' in cat_cols
    assert X_out['ts'].dtype == object


def test_split_feature_types_returns_copy():
    """Оригинальный DataFrame не мутируется."""
    X = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    original_dtypes = X.dtypes.copy()
    split_feature_types(X)
    pd.testing.assert_series_equal(X.dtypes, original_dtypes)


def test_split_feature_types_mixed():
    """Смешанные типы корректно разделяются."""
    X = pd.DataFrame({
        'amount':   [10.5, 20.0, 30.1, 40.2, 50.0],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'is_new':   [0, 1, 0, 0, 1],
    })
    _, cat_cols, continuous_cols = split_feature_types(X)
    assert 'amount' in continuous_cols
    assert 'category' in cat_cols
    assert 'is_new' in cat_cols





# ============================================================
# Тест наличия ключевых зависимостей
# ============================================================

def test_catboost_importable():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=1, verbose=0)
    assert model is not None


def test_mlflow_importable():
    import mlflow
    assert mlflow.__version__ is not None
