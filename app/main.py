# import os
# import pickle
# import pandas as pd
#
# from fastapi import FastAPI, Request, HTTPException, Depends
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
#
# from sqlalchemy.orm import Session
# from catboost import CatBoostClassifier
# from pydantic import BaseModel, Field
# from typing import Optional
#
# from app.database import engine, Base, get_db
# from app.schemas import Prediction
#
# # ---------------- APP ----------------
# app = FastAPI(
#     title="Fraud Detection API",
#     description="Сервис предсказания мошеннических транзакций",
#     version="1.0.0"
# )
#
# # ---------------- PATHS (FIXED) ----------------
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# TEMPLATES_DIR = "/app/templates"
# STATIC_DIR = "/app/static"
# MODELS_DIR = "/app/models"
# from jinja2 import Environment, FileSystemLoader, select_autoescape
# print("TEMPLATES DIR =", TEMPLATES_DIR)
# print("FILES =", os.listdir(TEMPLATES_DIR))
# # from jinja2 import Environment, FileSystemLoader, select_autoescape
# # from starlette.templating import Jinja2Templates
# #
# # env = Environment(
# #     loader=FileSystemLoader("/app/templates"),
# #     autoescape=select_autoescape(["html", "xml"]),
# #     cache_size=0   # 🔥 ВАЖНО: полностью отключает кеш (убирает баг)
# # )
# #
# # templates = Jinja2Templates(env=env)
# # #templates = Jinja2Templates(directory=TEMPLATES_DIR)
# # app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# from jinja2 import Environment, FileSystemLoader
#
# env = Environment(loader=FileSystemLoader("/app/templates"))
#
# @app.get("/")
# def home(request: Request):
#     template = env.get_template("index.html")
#     return HTMLResponse(template.render(request=request))
# # ---------------- DB INIT (FIXED) ----------------
# @app.on_event("startup")
# def startup():
#     Base.metadata.create_all(bind=engine)
#
# # ---------------- MODEL ----------------
# MODEL_PATH = os.path.join(MODELS_DIR, "model.cbm")
# ARTIFACTS_PATH = os.path.join(MODELS_DIR, "inference_artifacts.pkl")
#
# model = CatBoostClassifier()
# model.load_model(MODEL_PATH)
#
# with open(ARTIFACTS_PATH, "rb") as f:
#     artifacts = pickle.load(f)
#
# best_threshold = artifacts["best_threshold"]
# feature_order = artifacts["feature_order"]
#
# # ---------------- PAGES ----------------
# # @app.get("/", response_class=HTMLResponse)
# # def home(request: Request):
# #     return templates.TemplateResponse("index.html", {"request": request})
#
#
# @app.get("/predictions", response_class=HTMLResponse)
# def predictions_page(request: Request, db: Session = Depends(get_db)):
#     rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(50).all()
#
#     return templates.TemplateResponse(
#         "predictions.html",
#         {
#             "request": request,
#             "predictions": rows
#         },
#     )
#
#
# @app.get("/experiments", response_class=HTMLResponse)
# def experiments_page(request: Request):
#     metrics = artifacts.get("metrics", {})
#
#     return templates.TemplateResponse(
#         "experiments.html",
#         {
#             "request": request,
#             "threshold": best_threshold,
#             "roc": metrics.get("roc_auc", 0),
#             "f1": metrics.get("f1", 0),
#         },
#     )
#
# # ---------------- PREDICT SCHEMA ----------------
# class Transaction(BaseModel):
#     transaction_id: str = Field(example="T00000001")
#     customer_id: str = Field(example="C043212")
#     merchant_id: str = Field(example="M009249")
#     timestamp: str = Field(example="2025-02-17 17:41:57.841354")
#     date: str = Field(example="2025-02-17")
#     amount: float = Field(example=3.59)
#     currency: str = Field(example="USD")
#     is_foreign: int = Field(example=0)
#     is_online: int = Field(example=0)
#     is_night: int = Field(example=0)
#     is_weekend: int = Field(example=0)
#     hour: float = Field(example=17.47)
#     day_of_week: int = Field(example=0)
#     ip_address: Optional[str] = Field(None, example="78.159.28.180")     # теперь можно не передавать
#     device_id: Optional[str] = Field(None, example="2bb48282-ce20-4fc8-8e1f-e3194be99082")
#     operating_system: str = Field(example="MacOS")
#     age: int = Field(example=37)
#     income: int = Field(example=33392)
#     credit_score: int = Field(example=730)
#     merchant_category: str = Field(example="services")
#     risk_score: float = Field(example=0.01)
#     merchant_country: str = Field(example="MM")
#     amount_deviation: float = Field(example=-0.98)
#     time_since_last_tx: float = Field(example=0.91)
#     rule_high_amount_night: int = Field(example=0)
#     rule_foreign_new_device: int = Field(example=0)
#     rule_high_risk_merchant: int = Field(example=0)
#     rule_fast_sequence: int = Field(example=0)
#     rule_country_change: int = Field(example=0)
#
# # ---------------- PREDICT ----------------
# @app.post("/predict")
# def predict(transaction: Transaction, db: Session = Depends(get_db)):
#     try:
#         data = transaction.model_dump()
#         df = pd.DataFrame([data])
#
#         # align features
#         for col in feature_order:
#             if col not in df.columns:
#                 df[col] = 0
#
#         df = df[feature_order]
#
#         proba = model.predict_proba(df)[:, 1][0]
#         pred = int(proba >= best_threshold)
#
#         record = Prediction(
#             probability=float(proba),
#             prediction=pred,
#             anomaly=proba > 0.90,
#         )
#
#         db.add(record)
#         db.commit()
#
#         return {
#             "probability": round(proba, 6),
#             "prediction": pred
#         }
#
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))
#
# # ---------------- HEALTH ----------------
# @app.get("/health")
# def health():
#     return {"status": "ok"}


import os
import pickle
import pandas as pd

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sqlalchemy.orm import Session
from catboost import CatBoostClassifier
from pydantic import BaseModel, Field
from typing import Optional

from app.database import engine, Base, get_db
from app.schemas import Prediction

# ---------------- APP ----------------
app = FastAPI(
    title="Fraud Detection API",
    description="Сервис предсказания мошеннических транзакций",
    version="1.0.0"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATES_DIR = "/app/templates"
STATIC_DIR = "/app/static"
MODELS_DIR = "/app/models"

print("TEMPLATES DIR =", TEMPLATES_DIR)
print("FILES =", os.listdir(TEMPLATES_DIR))

# ---------------- TEMPLATES (FIX) ----------------
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------------- STATIC (FIX) ----------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------- DB INIT ----------------
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

# ---------------- MODEL ----------------
MODEL_PATH = os.path.join(MODELS_DIR, "model.cbm")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "inference_artifacts.pkl")

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

best_threshold = artifacts["best_threshold"]
feature_order = artifacts["feature_order"]

# ---------------- HOME ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- PAGES ----------------
@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(50).all()

    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "predictions": rows
        },
    )


@app.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request):
    metrics = artifacts.get("metrics", {})

    return templates.TemplateResponse(
        "experiments.html",
        {
            "request": request,
            "threshold": best_threshold,
            "roc": metrics.get("roc_auc", 0),
            "f1": metrics.get("f1", 0),
        },
    )

# ---------------- PREDICT SCHEMA (НЕ ТРОГАЕМ) ----------------
class Transaction(BaseModel):
    transaction_id: str = Field(example="T00000001")
    customer_id: str = Field(example="C043212")
    merchant_id: str = Field(example="M009249")
    timestamp: str = Field(example="2025-02-17 17:41:57.841354")
    date: str = Field(example="2025-02-17")
    amount: float = Field(example=3.59)
    currency: str = Field(example="USD")
    is_foreign: int = Field(example=0)
    is_online: int = Field(example=0)
    is_night: int = Field(example=0)
    is_weekend: int = Field(example=0)
    hour: float = Field(example=17.47)
    day_of_week: int = Field(example=0)
    ip_address: Optional[str] = Field(None, example="78.159.28.180")
    device_id: Optional[str] = Field(None, example="2bb48282-ce20-4fc8-8e1f-e3194be99082")
    operating_system: str = Field(example="MacOS")
    age: int = Field(example=37)
    income: int = Field(example=33392)
    credit_score: int = Field(example=730)
    merchant_category: str = Field(example="services")
    risk_score: float = Field(example=0.01)
    merchant_country: str = Field(example="MM")
    amount_deviation: float = Field(example=-0.98)
    time_since_last_tx: float = Field(example=0.91)
    rule_high_amount_night: int = Field(example=0)
    rule_foreign_new_device: int = Field(example=0)
    rule_high_risk_merchant: int = Field(example=0)
    rule_fast_sequence: int = Field(example=0)
    rule_country_change: int = Field(example=0)

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(transaction: Transaction, db: Session = Depends(get_db)):
    try:
        data = transaction.model_dump()
        df = pd.DataFrame([data])

        for col in feature_order:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_order]
        proba = model.predict_proba(df)[:, 1][0]
        pred = int(proba >= best_threshold)

        record = Prediction(
            probability=float(proba),
            prediction=pred,
            anomaly=proba > 0.90,
        )

        db.add(record)
        db.commit()
        return {
            "probability": round(proba, 6),
            "prediction": pred
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


import subprocess
from fastapi import BackgroundTasks

TRAIN_STATUS = {
    "status": "idle",
    "log": "",
    "returncode": None
}

def run_training():
    global TRAIN_STATUS
    TRAIN_STATUS["status"] = "running"
    result = subprocess.run(
        ["python", "/app/src/train.py"],
        capture_output=True,
        text=True
    )
    TRAIN_STATUS["returncode"] = result.returncode
    TRAIN_STATUS["log"] = (result.stdout or "")[-3000:] + "\n" + (result.stderr or "")[-3000:]
    if result.returncode == 0:
        TRAIN_STATUS["status"] = "done"
    else:
        TRAIN_STATUS["status"] = "failed"

@app.get("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training)
    return {"message": "training started"}

@app.get("/train-status")
def train_status():
    return TRAIN_STATUS


@app.get("/drift", response_class=HTMLResponse)
def drift_page(request: Request):
    drift_path = "/app/models/drift.json"
    drift = {
        "drift_detected": False,
        "score": 0.0
    }
    if os.path.exists(drift_path):
        import json
        with open(drift_path) as f:
            drift = json.load(f)
    return templates.TemplateResponse(
        "drift.html",
        {
            "request": request,
            "drift": drift
        }
    )