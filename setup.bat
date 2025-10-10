@echo off
REM 🏛️ Midas AI Service - Setup Script (Windows)
echo 🚀 Configurando Midas AI Service...

REM Create virtual environment
echo 📦 Criando ambiente virtual...
python -m venv venv

REM Activate virtual environment
echo 🔧 Ativando ambiente virtual...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Instalando dependências...
pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Criando diretórios...
if not exist "data\faiss_index" mkdir data\faiss_index
if not exist "data\tfidf_index" mkdir data\tfidf_index
if not exist "logs" mkdir logs

REM Copy environment file
echo ⚙️ Configurando variáveis de ambiente...
if not exist ".env" (
    copy .env.example .env
    echo ✅ Arquivo .env criado. Configure suas variáveis de ambiente!
)

echo 🎉 Setup concluído!
echo.
echo 📋 Próximos passos:
echo 1. Configure o arquivo .env com suas credenciais
echo 2. Configure PostgreSQL com pgvector
echo 3. Configure Oracle Database
echo 4. Execute: python -m app.main

pause