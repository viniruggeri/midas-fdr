# 🧠 Midas FDR v2 — Full Cognitive Reasoning Stack
FROM python:3.10-slim AS base

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema (necessárias pro PyTorch, FAISS e PostgreSQL)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    libomp-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip e instala wheel (pra builds mais rápidos)
RUN pip install --upgrade pip wheel setuptools

# Copia requirements
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Instala dependências (com flags que garantem fallback pro CPU)
RUN pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copia código do projeto
COPY . /app

# Expõe porta padrão do FastAPI
EXPOSE 8080

# Comando padrão pra subir a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
