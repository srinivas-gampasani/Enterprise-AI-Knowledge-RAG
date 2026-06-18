#!/bin/bash
# ============================================================
#  Ascension Via Christi RAG System — Quick Start Script
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "============================================================"
echo "  Ascension Via Christi — RAG System Setup"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}ERROR: python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYVER=$(python3 -c "import sys; print(sys.version_info.minor)")
echo -e "${GREEN}Python 3.${PYVER} found.${NC}"

# Check .env
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}No .env found. Copying from .env.example...${NC}"
    cp backend/.env.example backend/.env
    echo -e "${RED}ACTION REQUIRED: Edit backend/.env and set OPENAI_API_KEY${NC}"
    echo "  Open backend/.env and replace: sk-your-openai-api-key-here"
    echo ""
    read -p "Press Enter once you've added your API key..."
fi

# Install deps
echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"
cd backend
pip install -r requirements.txt -q
cd ..
echo -e "${GREEN}Dependencies installed.${NC}"

# Ingest documents
echo ""
echo -e "${BLUE}Ingesting documents into FAISS index...${NC}"
cd backend
python ../scripts/ingest_documents.py
cd ..
echo -e "${GREEN}Documents indexed.${NC}"

# Start server
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Starting FastAPI server on http://localhost:8000${NC}"
echo -e "${GREEN}  Open frontend/index.html in your browser to use the UI${NC}"
echo -e "${GREEN}  API docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
