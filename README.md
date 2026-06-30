# Fraud Detection — MLOps проект

Детекция мошеннических транзакций на CatBoost: версионирование данных (DVC),
трекинг экспериментов (MLflow), CI/CD, сервис на FastAPI с OpenAPI,
мониторинг дрейфа (Prometheus + Grafana) и финальное развёртывание в
Kubernetes через ArgoCD (CD).

> Docker Compose используется только для локальной отладки. Конечная цель
> проекта — работающий сервис в Kubernetes (Minikube), куда изменения
> доезжают автоматически через ArgoCD.

---

## Содержание

- [Структура проекта](#структура-проекта)
- [Быстрый старт (Docker Compose, для отладки)](#быстрый-старт-docker-compose-для-отладки)
- [Финальный запуск (Kubernetes / Minikube)](#финальный-запуск-kubernetes--minikube)
- [Continuous Deployment через ArgoCD](#continuous-deployment-через-argocd)
- [OpenAPI / Swagger](#openapi--swagger)
- [Мониторинг дрейфа](#мониторинг-дрейфа)
- [CI](#ci)
- [Тесты и линтер](#тесты-и-линтер)

---

## Структура проекта

```
.
├── app/                      # FastAPI-сервис: инференс + веб-UI
│   ├── main.py                  # /, /predict, /predictions, /experiments,
│   │                             #   /drift, /retrain, /health
│   ├── database.py
│   └── schemas.py
├── src/
│   ├── train.py               # обучение CatBoost + логирование в MLflow
│   ├── utilits.py             # split_feature_types
│   └── monitoring/            # расчёт дрейфа
│       ├── drift_metrics.py      # PSI, KL, JS, KS-тест, Wasserstein, PR-AUC
│       ├── generate_report.py    # CLI: считает дрейф на случайной выборке
│       └── metrics_exporter.py   # Prometheus-экспортёр, порт 9200
├── templates/, static/        # веб-UI
├── models/                    # model.cb/.cbm + inference_artifacts.pkl + drift.json
├── data/raw/                  # 4 parquet-части датасета (под DVC)
├── reports/drift/             # подробные JSON-отчёты о дрейфе
├── docker/                    # Dockerfile мониторинга, конфиги Prometheus/Grafana
├── k8s/                       # Kubernetes-манифесты + ArgoCD Application
├── tests/                     # pytest
├── Dockerfile                 # образ FastAPI-сервиса
├── docker-compose.yaml        # локальная отладка всего стека
└── requirements.txt
```

---

## Быстрый старт (Docker Compose, для отладки)

Нужны: Docker Desktop, 4 parquet-части датасета в `data/raw/`.

```bash
# 1. данные под DVC — подтянуть, если их нет локально
dvc pull

# 2. поднять весь стек: API, монитор дрейфа, Prometheus, Grafana
docker compose up --build
```

После старта:

| Сервис | Адрес |
|---|---|
| FastAPI UI | http://localhost:8000 |
| OpenAPI / Swagger | http://localhost:8000/docs |
| Страница дрейфа | http://localhost:8000/drift |
| Метрики дрейфа (сырые) | http://localhost:9200/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

Остановить:
```bash
docker compose down
```

**Если меняешь код** (`app/`, `src/`) — нужна пересборка: `docker compose up --build`.
**Если меняешь только `docker-compose.yaml`** (порты, env, volumes) — пересборка не нужна, достаточно `docker compose up`.

---

## Финальный запуск (Kubernetes / Minikube)

Это целевой способ запуска по требованиям проекта. Образы собираются
локально и публикуются прямо в Docker-демон Minikube — внешний registry
не нужен.

```bash
# 1. поднять кластер
minikube start

# 2. переключить docker-cli на демон minikube
eval $(minikube docker-env)            # Linux/macOS
# Windows PowerShell:
# & minikube -p minikube docker-env | Invoke-Expression

# 3. собрать образы (data/ и models/ запекаются внутрь — образ самодостаточен)
docker build -t fraud-api:latest -f Dockerfile .
docker build -t fraud-monitoring:latest -f docker/monitoring.Dockerfile .

# 4. применить манифесты
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/monitoring-deployment.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

# 5. проверить, что всё поднялось
kubectl get pods -n fraud-detection
```

Доступ к сервисам из браузера:

```bash
minikube service -n fraud-detection fraud-api
minikube service -n fraud-detection grafana
minikube service -n fraud-detection prometheus
```

Каждая команда откроет нужный сервис в браузере на NodePort-адресе Minikube.

> Контейнеры в Kubernetes не используют host-volumes с твоего компьютера —
> данные и модель уже встроены в образ на этапе `docker build`. Чтобы
> обновить данные/модель в кластере, пересобери образ и сделай
> `kubectl rollout restart deployment/fraud-api -n fraud-detection`
> (или просто закоммить и пусть это сделает ArgoCD, см. ниже).

---

## Continuous Deployment через ArgoCD

ArgoCD следит за репозиторием и сам синхронизирует кластер с тем, что
описано в папке `k8s/` — пуш в `main` доезжает до кластера без ручного
`kubectl apply`.

```bash
# установка ArgoCD в кластер
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# подключение приложения (поправь repoURL в файле под свой репозиторий)
kubectl apply -f k8s/argocd-application.yaml

# доступ к UI ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Открой `https://localhost:8080`, залогинься (логин `admin`, пароль —
команда `kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d`),
и увидишь приложение `fraud-detection` со статусом **Synced / Healthy** —
дерево всех развёрнутых ресурсов.

---

## OpenAPI / Swagger

FastAPI генерирует OpenAPI-схему автоматически. После запуска (любым
способом из вариантов выше) доступна по адресу `/docs` (Swagger UI) или
`/redoc`.

---

## Мониторинг дрейфа

`src/monitoring/generate_report.py` на каждой итерации (`--mode serve`,
по умолчанию каждые 5 минут) берёт **новую случайную выборку** из
датасета — поэтому метрики дрейфа в Prometheus/Grafana реально меняются
со временем, а не остаются одним и тем же значением.

Считается три вида дрейфа:
- **Data drift** — PSI/KS-тест/Wasserstein по каждому признаку
- **Target drift** — дрейф распределения предсказанных вероятностей
- **Concept drift** (proxy) — падение PR-AUC модели между батчами

Результаты пишутся в:
- `models/drift.json` — компактный, читает страница `/drift` в UI
- `reports/drift/latest_metrics.json` — подробная сводка по фичам
- `http://localhost:9200/metrics` — Prometheus-формат для Grafana

Метрики для дашборда в Grafana: `fraud_data_drift_share`,
`fraud_target_drift_psi`, `fraud_model_pr_auc`, `fraud_concept_drift_detected`.

---

## CI

`.github/workflows/ci.yml` на каждый push/PR в `main`:
1. ставит зависимости;
2. линтер: `ruff check src/ tests/`;
3. тесты: `pytest`.

---

## Тесты и линтер

```bash
pip install -r requirements.txt
pip install pytest ruff

ruff check src/ tests/
python -m pytest tests/ -v
```
