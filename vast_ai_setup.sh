#!/bin/bash
# vast_ai_setup.sh — One-click установка всех зависимостей на vast.ai

set -e  # Exit on error

echo "=========================================="
echo "🚀 VAST.AI SETUP - Установка зависимостей"
echo "=========================================="
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ==================== Проверка GPU ====================
echo -e "${YELLOW}🔍 Проверка GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✅ GPU обнаружен${NC}"
else
    echo -e "${RED}❌ NVIDIA GPU не найден!${NC}"
    echo "   Проверьте что вы арендовали правильный instance"
    exit 1
fi

echo ""

# ==================== Python версия ====================
echo -e "${YELLOW}🐍 Проверка Python...${NC}"
python3 --version

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 не установлен!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python OK${NC}"
echo ""

# ==================== pip обновление ====================
echo -e "${YELLOW}📦 Обновление pip...${NC}"
python3 -m pip install --upgrade pip setuptools wheel

echo ""

# ==================== PyTorch ====================
echo -e "${YELLOW}🔥 Установка PyTorch с CUDA...${NC}"

# Определяем версию CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "   Обнаружена CUDA: $CUDA_VERSION"
    
    # PyTorch для CUDA 12.x
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # PyTorch для CUDA 11.x
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        # Fallback на последнюю версию
        pip3 install torch torchvision torchaudio
    fi
else
    pip3 install torch torchvision torchaudio
fi

echo -e "${GREEN}✅ PyTorch установлен${NC}"
echo ""

# ==================== OpenCLIP ====================
echo -e "${YELLOW}🎨 Установка OpenCLIP...${NC}"
pip3 install open-clip-torch safetensors

echo -e "${GREEN}✅ OpenCLIP установлен${NC}"
echo ""

# ==================== HNSW ====================
echo -e "${YELLOW}🔗 Установка hnswlib...${NC}"
pip3 install hnswlib

echo -e "${GREEN}✅ hnswlib установлен${NC}"
echo ""

# ==================== Computer Vision ====================
echo -e "${YELLOW}📷 Установка Computer Vision библиотек...${NC}"
pip3 install opencv-python-headless pillow

echo -e "${GREEN}✅ CV библиотеки установлены${NC}"
echo ""

# ==================== Data Science ====================
echo -e "${YELLOW}📊 Установка Data Science библиотек...${NC}"
pip3 install numpy pandas scipy scikit-learn tqdm joblib

echo -e "${GREEN}✅ DS библиотеки установлены${NC}"
echo ""

# ==================== OCR (опционально) ====================
echo -e "${YELLOW}📝 Установка OCR (EasyOCR)...${NC}"
pip3 install easyocr

echo -e "${GREEN}✅ OCR установлен${NC}"
echo ""

# ==================== Monitoring ====================
echo -e "${YELLOW}📈 Установка мониторинга...${NC}"
pip3 install py3nvml psutil

echo -e "${GREEN}✅ Мониторинг установлен${NC}"
echo ""

# ==================== Streetlevel (для скачивания панорам) ====================
echo -e "${YELLOW}🗺️  Установка streetlevel...${NC}"
pip3 install streetlevel aiohttp

echo -e "${GREEN}✅ streetlevel установлен${NC}"
echo ""

# ==================== Проверка установки ====================
echo ""
echo "=========================================="
echo "🔬 ПРОВЕРКА УСТАНОВКИ"
echo "=========================================="
echo ""

echo -e "${YELLOW}Проверка PyTorch CUDA...${NC}"
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("⚠️  CUDA не доступна!")
EOF

echo ""
echo -e "${YELLOW}Проверка OpenCLIP...${NC}"
python3 -c "import open_clip; print(f'OpenCLIP: OK')"

echo ""
echo -e "${YELLOW}Проверка hnswlib...${NC}"
python3 -c "import hnswlib; print(f'hnswlib: OK')"

echo ""
echo -e "${YELLOW}Проверка OpenCV...${NC}"
python3 -c "import cv2; print(f'OpenCV {cv2.__version__}: OK')"

echo ""
echo -e "${YELLOW}Проверка EasyOCR...${NC}"
python3 -c "import easyocr; print(f'EasyOCR: OK')"

echo ""
echo "=========================================="
echo -e "${GREEN}✅ ✅ ✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!${NC}"
echo "=========================================="
echo ""
echo "🎯 Теперь можно запускать индексацию:"
echo "   python scripts/04_build_index_production.py --clip-model \"ViT-L-14\" --ocr"
echo ""
echo "📊 Или готовый скрипт для центра Москвы:"
echo "   bash scripts/run_moscow_center.sh"
echo ""