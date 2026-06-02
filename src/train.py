import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-режим для CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost

# ============================================================
# НАСТРОЙКА ПУТЕЙ
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

file_names = [
    "part-00000-524fc3d6-f3dd-4163-88ff-a6cff258d76e-c000.snappy.parquet",
    "part-00001-524fc3d6-f3dd-4163-88ff-a6cff258d76e-c000.snappy.parquet",
    "part-00002-524fc3d6-f3dd-4163-88ff-a6cff258d76e-c000.snappy.parquet",
    "part-00003-524fc3d6-f3dd-4163-88ff-a6cff258d76e-c000.snappy.parquet"
]

file_paths = [os.path.join(DATA_RAW, f) for f in file_names]

missing = [f for f in file_paths if not os.path.isfile(f)]
if missing:
    raise FileNotFoundError("Следующие файлы не найдены:\n" + "\n".join(missing))

print("✅ Все parquet-файлы найдены.")
df = pd.concat([pd.read_parquet(f) for f in file_paths], ignore_index=True)

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

target_col = [col for col in df.columns if 'fraud' in col.lower()][0]
print("Target column:", target_col)
print(df[target_col].value_counts())

# ============================================================
# EDA-ГРАФИКИ (сохраняются как файлы, не plt.show())
# ============================================================
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(PLOTS_DIR, exist_ok=True)

fig, ax = plt.subplots()
sns.countplot(x=df[target_col], ax=ax)
ax.set_title("Class Distribution")
fig.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', ax=ax)
ax.set_title("Correlation Matrix")
fig.savefig(os.path.join(PLOTS_DIR, "correlation_matrix.png"))
plt.close(fig)

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================
df = df.copy()
target_col = 'is_fraud'

drop_cols = ['transaction_id', 'customer_id', 'merchant_id']
df = df.drop(columns=drop_cols, errors='ignore')
df = df.fillna(-999)

X = df.drop(columns=[target_col])
y = df[target_col]

print(X.shape, y.shape)
print(y.value_counts())


def split_feature_types(X: pd.DataFrame):
    X = X.copy()
    cat_cols = []
    continuous_cols = []

    for col in X.columns:
        col_data = X[col]

        if pd.api.types.is_datetime64_any_dtype(col_data):
            X[col] = col_data.astype(str).fillna('missing')
            cat_cols.append(col)

        elif (
            pd.api.types.is_object_dtype(col_data)
            or pd.api.types.is_categorical_dtype(col_data)
            or pd.api.types.is_bool_dtype(col_data)
        ):
            X[col] = col_data.astype(str).fillna('missing')
            cat_cols.append(col)

        else:
            non_null_unique = pd.Series(col_data.dropna().unique())
            if len(non_null_unique) <= 2:
                unique_set = set(non_null_unique.tolist())
                if unique_set.issubset({0, 1}) or unique_set.issubset({0.0, 1.0}) or unique_set.issubset({'0', '1'}):
                    X[col] = col_data.fillna(-1).astype(str)
                    cat_cols.append(col)
                else:
                    continuous_cols.append(col)
            else:
                continuous_cols.append(col)

    return X, cat_cols, continuous_cols


X, cat_cols, continuous_cols = split_feature_types(X)

print('Всего признаков:', X.shape[1])
print('Categorical/Binary:', len(cat_cols))
print('Continuous:', len(continuous_cols))

for col in continuous_cols:
    X[col] = X[col].fillna(X[col].median())

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

print('Train:', X_train.shape)
print('Valid:', X_valid.shape)
print('Test:', X_test.shape)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
raw_pos_weight = neg / pos
pos_weight = min(raw_pos_weight, 50)

print('raw_pos_weight =', raw_pos_weight)
print('used_pos_weight =', pos_weight)

# ============================================================
# ОБУЧЕНИЕ + MLFLOW
# ============================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("fraud_detection")

MODEL_PARAMS = {
    "iterations": 500,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "random_seed": 42,
}

with mlflow.start_run(run_name="catboost_baseline"):

    # --- Логируем параметры ---
    mlflow.log_params({
        **MODEL_PARAMS,
        "pos_weight": pos_weight,
        "raw_pos_weight": raw_pos_weight,
        "train_size": len(X_train),
        "valid_size": len(X_valid),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
        "n_cat_features": len(cat_cols),
    })

    # --- Обучение ---
    model = CatBoostClassifier(
        **MODEL_PARAMS,
        class_weights=[1, pos_weight],
        verbose=100
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    # --- Подбор порога на валидации ---
    valid_proba = model.predict_proba(X_valid)[:, 1]
    thresholds = np.arange(0.01, 1.00, 0.01)
    rows = []

    for thr in thresholds:
        y_pred = (valid_proba >= thr).astype(int)
        rec = recall_score(y_valid, y_pred, zero_division=0)
        prec = precision_score(y_valid, y_pred, zero_division=0)
        f1 = f1_score(y_valid, y_pred, zero_division=0)
        rows.append({'threshold': thr, 'recall': rec, 'precision': prec, 'f1': f1})

    thr_df = pd.DataFrame(rows)
    best_row = thr_df.sort_values(['f1', 'precision'], ascending=False).iloc[0]
    best_threshold = float(best_row['threshold'])

    print('Best threshold:', best_threshold)
    mlflow.log_param("best_threshold", best_threshold)

    # --- Метрики на тесте ---
    test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (test_proba >= best_threshold).astype(int)

    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)

    print('TEST METRICS')
    print('Recall   :', round(test_recall, 6))
    print('Precision:', round(test_precision, 6))
    print('F1       :', round(test_f1, 6))
    print('ROC-AUC  :', round(test_roc_auc, 6))
    print('PR-AUC   :', round(test_pr_auc, 6))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_test_pred))

    mlflow.log_metrics({
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
    })

    # --- Feature importance ---
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(30))

    # --- График вероятностей ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_proba[y_test == 1], bins=50, alpha=0.5, label='fraud')
    ax.legend()
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability separation')
    prob_plot_path = os.path.join(PLOTS_DIR, "probability_separation.png")
    fig.savefig(prob_plot_path)
    plt.close(fig)

    # --- Логируем артефакты в MLflow ---
    mlflow.log_artifact(prob_plot_path)
    mlflow.log_artifact(os.path.join(PLOTS_DIR, "class_distribution.png"))
    mlflow.log_artifact(os.path.join(PLOTS_DIR, "correlation_matrix.png"))

    # --- Регистрируем модель ---
    mlflow.catboost.log_model(
        model,
        artifact_path="catboost_model",
        registered_model_name="fraud_detection_catboost"
    )

    print(f"✅ MLflow run logged. Run ID: {mlflow.active_run().info.run_id}")

# ============================================================
# СОХРАНЕНИЕ АРТЕФАКТОВ ИНФЕРЕНСА (локально, как раньше)
# ============================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

model.save_model(os.path.join(MODELS_DIR, "model.cbm"))

inference_artifacts = {
    'best_threshold': best_threshold,
    'cat_features': cat_cols,
    'feature_order': list(X_train.columns),
    'metrics': {
        'recall': test_recall,
        'precision': test_precision,
        'f1': test_f1,
        'roc_auc': test_roc_auc,
        'pr_auc': test_pr_auc
    }
}

with open(os.path.join(MODELS_DIR, "inference_artifacts.pkl"), 'wb') as f:
    pickle.dump(inference_artifacts, f)

with open(os.path.join(MODELS_DIR, "metrics.json"), 'w') as f:
    json.dump(inference_artifacts['metrics'], f, indent=2)

print(f"✅ Model and artifacts saved to {MODELS_DIR}")
