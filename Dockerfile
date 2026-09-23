FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
COPY requirements-ci.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-ci.txt

COPY app ./app
COPY src ./src
COPY knowledge_base ./knowledge_base

COPY models/checkpoints/efficientnet_b0_best.pth \
     ./models/checkpoints/efficientnet_b0_best.pth

COPY reports/results/efficientnet_b0_calibration.json \
     ./reports/results/efficientnet_b0_calibration.json

COPY reports/results/efficientnet_b0_ood.json \
     ./reports/results/efficientnet_b0_ood.json

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]