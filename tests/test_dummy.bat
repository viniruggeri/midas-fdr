@echo off
echo 🏛️ Midas AI Service - Teste Local com Dados Dummy
echo ===============================================

echo.
echo 📦 Ativando ambiente virtual...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ❌ Ambiente virtual não encontrado!
    echo Execute primeiro: setup.bat
    pause
    exit /b 1
)

echo.
echo 🧪 Executando teste local do RAG...
python test_local.py

echo.
echo 🎉 Teste concluído!
pause