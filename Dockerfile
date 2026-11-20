# Unified Potato/Tomato V2 + Grapes V3 inference image.
# Default CPU wheels work with Docker Desktop on Windows without NVIDIA setup.
# Optional CUDA image: --build-arg TORCH_REQUIREMENTS=requirements-gpu.txt
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# OpenMP runtime used by scientific Python packages; no source build tools needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ci.txt requirements-gpu.txt ./
ARG TORCH_REQUIREMENTS=requirements-ci.txt
RUN case "$TORCH_REQUIREMENTS" in \
      requirements-ci.txt|requirements-gpu.txt) ;; \
      *) echo "Unsupported TORCH_REQUIREMENTS: $TORCH_REQUIREMENTS"; exit 2 ;; \
    esac \
    && python -m pip install --upgrade pip \
    && python -m pip install -r "$TORCH_REQUIREMENTS" \
    && python -m pip check

# Source code and expert-review-pending knowledge base.
COPY app ./app
COPY src ./src
COPY v3 ./v3
COPY knowledge_base ./knowledge_base

# V2 checkpoint, temperature calibration, and OOD statistics are versioned
# artifacts already present in the repository. The Grapes checkpoint is NOT
# baked into the image; docker-compose mounts models/grapes from the host.
COPY models/checkpoints/efficientnet_b0_best.pth ./models/checkpoints/efficientnet_b0_best.pth
COPY reports/results/efficientnet_b0_calibration.json ./reports/results/efficientnet_b0_calibration.json
COPY reports/results/efficientnet_b0_ood.json ./reports/results/efficientnet_b0_ood.json

# Leave the image non-root and writable cache/home for Streamlit.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin plantapp \
    && mkdir -p /app/models/grapes \
    && chown -R plantapp:plantapp /app
USER plantapp

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
