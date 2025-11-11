FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN python -c "import torch; \
    from transformers import AutoTokenizer, AutoModel; \
    model_name = 'facebook/esm2_t33_650M_UR50D'; \
    print('Descargando modelo ESM2...'); \
    tokenizer = AutoTokenizer.from_pretrained(model_name); \
    model = AutoModel.from_pretrained(model_name); \
    print('Modelo ESM descargado exitosamente')"

COPY src/ ./src/
COPY run_prediction.py .
COPY mpnn_results.csv .
COPY models/ ./models/

RUN mkdir -p /app/output

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV BATCH_SIZE=16
ENV MAX_SEQ_LENGTH=1024

# Punto de entrada
ENTRYPOINT ["python", "run_prediction.py"]

CMD ["--help"]

