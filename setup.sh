#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Physics ML - Local Setup${NC}"
echo "================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo -e "\n${YELLOW}Step 1: Checking Python version...${NC}"
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
else
    echo -e "${RED}✗ Python 3.10+ required, but $PYTHON_VERSION found${NC}"
    exit 1
fi

# Check for uv
echo -e "\n${YELLOW}Step 1b: Checking for uv package manager...${NC}"
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ uv is installed${NC}"
else
    echo -e "${YELLOW}Installing uv...${NC}"
    pip install uv >/dev/null 2>&1
    echo -e "${GREEN}✓ uv installed${NC}"
fi

# Create virtual environment
echo -e "\n${YELLOW}Step 2: Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Step 3: Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies using uv
echo -e "\n${YELLOW}Step 4: Installing dependencies from pyproject.toml...${NC}"
if [ -f "pyproject.toml" ]; then
    uv pip install -e .
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${RED}✗ pyproject.toml not found${NC}"
    exit 1
fi

# Create directories
echo -e "\n${YELLOW}Step 5: Creating necessary directories...${NC}"
mkdir -p raw_data
mkdir -p artifacts
mkdir -p plots
echo -e "${GREEN}✓ Directories created${NC}"

# Verify installations
echo -e "\n${YELLOW}Step 6: Verifying installations...${NC}"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" || echo "  PyTorch: Failed"
python3 -c "import pytorch_lightning; print(f'  PyTorch Lightning: {pytorch_lightning.__version__}')" || echo "  PyTorch Lightning: Failed"
python3 -c "import mlflow; print(f'  MLflow: {mlflow.__version__}')" || echo "  MLflow: Failed"
python3 -c "import scipy; print(f'  SciPy: {scipy.__version__}')" || echo "  SciPy: Failed"
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" || echo "  NumPy: Failed"

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo ""
echo "1. Generate training data:"
echo "   python data_processing/generate_data.py"
echo ""
echo "2. Organize data into the expected structure:"
echo "   mkdir -p raw_data/lotka_volterra_trajectories_T50_beta05_20_IC/{train_val,test}"
echo "   # Move generated parquet files accordingly"
echo ""
echo "3. Start MLflow tracking server (in a new terminal):"
echo "   source .venv/bin/activate"
echo "   mlflow ui"
echo ""
echo "4. Train a model (in another terminal):"
echo "   source .venv/bin/activate"
echo "   python main.py"
echo ""
echo -e "${YELLOW}For detailed instructions, see SETUP.md${NC}"
