# Контейнер сервиса мониторинга дрейфа.
# Независим от FastAPI-сервиса — отдельный образ, свой порт 9200 для Prometheus.
#
# data/ и models/ копируются ВНУТРЬ образа, чтобы контейнер был самодостаточным
# и одинаково запускался и в docker-compose (для отладки), и в Kubernetes/Minikube
# (финальная цель по ТЗ), где нет общего host-volume с твоим компьютером.
# В docker-compose volumes из docker-compose.yaml при необходимости переопределят
# эти файлы своими (для быстрой локальной разработки без пересборки образа).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/

EXPOSE 9200

# По умолчанию — serve-режим: пересчёт дрейфа на новой случайной выборке
# каждые 5 минут, метрики живут на /metrics постоянно.
ENTRYPOINT ["python", "-m", "src.monitoring.generate_report"]
CMD ["--mode", "serve", "--sample", "50000", "--interval", "300"]
