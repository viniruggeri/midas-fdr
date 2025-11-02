# 🚀 MIDAS FDR v2 - Setup Automatizado
# Execute: .\setup.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  MIDAS FDR v2 - Auto Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar Docker
Write-Host "1️⃣  Verificando Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "   ✅ $dockerVersion" -ForegroundColor Green
}
catch {
    Write-Host "   ❌ Docker não encontrado. Instale: https://www.docker.com/products/docker-desktop" -ForegroundColor Red
    exit 1
}

# Testar se Docker Desktop está rodando
Write-Host ""
Write-Host "2️⃣  Verificando Docker Desktop..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "   ✅ Docker Desktop está rodando" -ForegroundColor Green
}
catch {
    Write-Host "   ⚠️  Docker Desktop NÃO está rodando!" -ForegroundColor Red
    Write-Host "   📌 Abra o Docker Desktop e aguarde iniciar (ícone da baleia azul)" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "   Pressione ENTER após iniciar o Docker Desktop..."
    
    # Aguardar Docker ficar disponível
    $maxRetries = 30
    $retries = 0
    while ($retries -lt $maxRetries) {
        try {
            docker ps | Out-Null
            Write-Host "   ✅ Docker Desktop conectado!" -ForegroundColor Green
            break
        }
        catch {
            $retries++
            Write-Host "   ⏳ Aguardando Docker... ($retries/$maxRetries)" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    
    if ($retries -eq $maxRetries) {
        Write-Host "   ❌ Timeout aguardando Docker. Reinicie o Docker Desktop." -ForegroundColor Red
        exit 1
    }
}

# 3. Verificar Python
Write-Host ""
Write-Host "3️⃣  Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "   ✅ $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "   ❌ Python não encontrado. Instale Python 3.10+: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# 4. Criar/Ativar ambiente virtual
Write-Host ""
Write-Host "4️⃣  Configurando ambiente virtual..." -ForegroundColor Yellow
$venvPath = if (Test-Path ".venv") { ".venv" } elseif (Test-Path "venv") { "venv" } else { $null }

if (!$venvPath) {
    Write-Host "   📦 Criando .venv..." -ForegroundColor Cyan
    python -m venv .venv
    $venvPath = ".venv"
}
Write-Host "   ✅ Ambiente virtual pronto: $venvPath" -ForegroundColor Green

# 5. Verificar .env
Write-Host ""
Write-Host "5️⃣  Configurando .env..." -ForegroundColor Yellow
if (!(Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "   ⚠️  Arquivo .env criado!" -ForegroundColor Yellow
        Write-Host "   📝 AÇÃO NECESSÁRIA: Edite o .env e adicione sua OPENAI_API_KEY" -ForegroundColor Red
        Write-Host "   🔑 Pegue sua key em: https://platform.openai.com/api-keys" -ForegroundColor Cyan
        Write-Host ""
        $response = Read-Host "   Deseja abrir o .env agora? (s/n)"
        if ($response -eq "s") {
            notepad .env
            Write-Host ""
            Read-Host "   Pressione ENTER após salvar a API key..."
        }
    }
    else {
        Write-Host "   ⚠️  .env.example não encontrado. Criando .env básico..." -ForegroundColor Yellow
        @"
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=midas123

# OpenAI
OPENAI_API_KEY=your-api-key-here

# App
ENVIRONMENT=development
"@ | Out-File -FilePath ".env" -Encoding UTF8
        Write-Host "   ⚠️  Edite o .env e adicione sua OPENAI_API_KEY!" -ForegroundColor Red
    }
}
else {
    Write-Host "   ✅ .env já existe" -ForegroundColor Green
}

# 6. Instalar dependências
Write-Host ""
Write-Host "6️⃣  Instalando dependências Python..." -ForegroundColor Yellow
Write-Host "   ⏱️  Isso pode levar 5-8 minutos (PyTorch é pesado)..." -ForegroundColor Cyan
Write-Host ""

# Ativar venv
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "   ✅ Ambiente virtual ativado" -ForegroundColor Green
}
else {
    Write-Host "   ⚠️  Não consegui ativar o venv automaticamente" -ForegroundColor Yellow
}

$pipInstall = Read-Host "   Instalar dependências agora? (s/n)"
if ($pipInstall -eq "s") {
    & (Join-Path $venvPath "Scripts\pip.exe") install -r requirements.txt
    Write-Host "   ✅ Dependências instaladas" -ForegroundColor Green
}
else {
    Write-Host "   ⏭️  Pulado. Execute manualmente: pip install -r requirements.txt" -ForegroundColor Yellow
}

# 7. Iniciar Neo4j
Write-Host ""
Write-Host "7️⃣  Iniciando Neo4j (Docker)..." -ForegroundColor Yellow
docker-compose up -d

Write-Host "   ⏳ Aguardando Neo4j iniciar (30s)..." -ForegroundColor Cyan
Start-Sleep -Seconds 30

# Verificar se Neo4j está rodando
$neo4jRunning = docker ps --filter "name=neo4j" --format "{{.Names}}"
if ($neo4jRunning) {
    Write-Host "   ✅ Neo4j rodando: $neo4jRunning" -ForegroundColor Green
    Write-Host "   🌐 Browser: http://localhost:7474 (neo4j / midas123)" -ForegroundColor Cyan
}
else {
    Write-Host "   ⚠️  Neo4j pode não estar rodando. Verifique: docker ps" -ForegroundColor Yellow
}

# 8. Resumo final
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  ✅ SETUP COMPLETO!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 PRÓXIMOS PASSOS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1️⃣  Inicie o servidor FastAPI:" -ForegroundColor White
Write-Host "   .$venvPath\Scripts\Activate" -ForegroundColor Gray
Write-Host "   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Gray
Write-Host ""
Write-Host "2️⃣  Popule o grafo:" -ForegroundColor White
Write-Host "   curl -X POST http://localhost:8000/graph/populate" -ForegroundColor Gray
Write-Host ""
Write-Host "3️⃣  Treine a GNN:" -ForegroundColor White
Write-Host "   python train_gnn.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4️⃣  Teste o sistema:" -ForegroundColor White
Write-Host "   python demo_mvp.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Guia completo: QUICK_START.md" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bora revolucionar o reasoning em LLMs!" -ForegroundColor Green
Write-Host ""
