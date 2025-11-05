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
    gcc \
    g++ && rm -rf /var/lib/apt/lists/*

# Atualiza pip e instala wheel (pra builds mais rápidos)
RUN pip install --upgrade pip wheel setuptools

# Instalar o PyTorch com suporte para CPU (garantindo que esteja no ambiente)
RUN pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copia o arquivo requirements.txt
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Instalar dependências do projeto (já inclui o 'torch-scatter' no requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . /app

# Expõe a porta padrão do FastAPI
EXPOSE 8080

# Comando padrão para subir a API com o Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]