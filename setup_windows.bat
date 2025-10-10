@echo off
echo 🏛️ Midas AI Service - Setup Windows
echo ===================================

echo.
echo 📦 Criando ambiente virtual...
python -m venv venv

echo.
echo 🔧 Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo.
echo 📥 Atualizando pip...
python -m pip install --upgrade pip

echo.
echo 📚 Instalando dependências...
pip install -r requirements.txt

echo.
echo 📁 Criando diretórios...
if not exist "data\faiss_index" mkdir data\faiss_index
if not exist "data\tfidf_index" mkdir data\tfidf_index
if not exist "logs" mkdir logs

echo.
echo ⚙️ Configurando variáveis de ambiente...
if not exist ".env" (
    copy .env.example .env
    echo ✅ Arquivo .env criado. Configure suas variáveis de ambiente!
) else (
    echo ⚠️ Arquivo .env já existe.
)

echo.
echo 🎉 Setup concluído!
echo.
echo 📋 Próximos passos:
echo 1. Configure o arquivo .env com suas credenciais
echo 2. Para testar localmente: test_dummy.bat  
echo 3. Para executar o serviço: python -m app.main

pause