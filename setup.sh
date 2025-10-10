#!/bin/bash

# 🏛️ Midas AI Service - Setup Script
echo "🚀 Configurando Midas AI Service..."

# Create virtual environment
echo "📦 Criando ambiente virtual..."
python -m venv venv

# Activate virtual environment
echo "🔧 Ativando ambiente virtual..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📥 Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Criando diretórios..."
mkdir -p data/faiss_index
mkdir -p data/tfidf_index
mkdir -p logs

# Copy environment file
echo "⚙️ Configurando variáveis de ambiente..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Arquivo .env criado. Configure suas variáveis de ambiente!"
fi

echo "🎉 Setup concluído!"
echo ""
echo "📋 Próximos passos:"
echo "1. Configure o arquivo .env com suas credenciais"
echo "2. Configure PostgreSQL com pgvector"
echo "3. Configure Oracle Database"
echo "4. Execute: python -m app.main"