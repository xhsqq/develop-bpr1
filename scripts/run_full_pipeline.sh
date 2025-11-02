#!/bin/bash
# Full pipeline script: Download -> Preprocess -> Extract Features -> Train -> Evaluate
# 完整流程：下载数据 -> 预处理 -> 提取多模态特征 -> 训练 -> 评估

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo "用法: $0 [category] [batch_size] [epochs] [device]"
    echo ""
    echo "参数:"
    echo "  category    数据集类别 (默认: beauty)"
    echo "              可选: beauty, games, sports"
    echo "  batch_size  批次大小 (默认: 256)"
    echo "  epochs      训练轮数 (默认: 50)"
    echo "  device      设备 (默认: auto)"
    echo "              可选: cuda, cpu, auto"
    echo ""
    echo "示例:"
    echo "  $0                           # 使用默认参数"
    echo "  $0 beauty 128 100           # 指定category, batch_size, epochs"
    echo "  $0 beauty 256 50 cuda       # 指定所有参数"
    echo ""
    exit 0
}

# 检查帮助标志
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# Default parameters
CATEGORY=${1:-beauty}
BATCH_SIZE=${2:-256}
EPOCHS=${3:-50}
DEVICE=${4:-auto}

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  QMCSR - 完整训练流程${NC}"
echo -e "${GREEN}  Quantum Multimodal Causal Sequential Recommender${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "${BLUE}配置:${NC}"
echo -e "  数据集: ${YELLOW}$CATEGORY${NC}"
echo -e "  批次大小: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  训练轮数: ${YELLOW}$EPOCHS${NC}"
echo -e "  设备: ${YELLOW}$DEVICE${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# Step 1: Download data
echo -e "${YELLOW}[Step 1/6] 下载原始数据...${NC}"
echo "-------------------------------------------------------------"
if [ -f "data/raw/${CATEGORY}_reviews.json" ] && [ -f "data/raw/${CATEGORY}_meta.json" ]; then
    echo -e "${GREEN}✓ 原始数据已存在，跳过下载${NC}"
else
    python data/download_amazon.py --category $CATEGORY
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 数据下载失败！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 数据下载完成${NC}"
fi
echo ""

# Step 2: Preprocess data
echo -e "${YELLOW}[Step 2/6] 数据预处理...${NC}"
echo "-------------------------------------------------------------"
if [ -f "data/processed/${CATEGORY}/train_sequences.pkl" ] && [ -f "data/processed/${CATEGORY}/item_features.pkl" ]; then
    echo -e "${GREEN}✓ 预处理数据已存在，跳过此步骤${NC}"
else
    python data/preprocess_amazon.py --category $CATEGORY
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 数据预处理失败！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 数据预处理完成${NC}"
fi
echo ""

# Step 3: Extract text features
echo -e "${YELLOW}[Step 3/6] 提取BERT文本特征...${NC}"
echo "-------------------------------------------------------------"
if [ -f "data/processed/${CATEGORY}/text_features.pkl" ]; then
    echo -e "${GREEN}✓ 文本特征已存在，跳过提取${NC}"
else
    if [ "$DEVICE" = "auto" ]; then
        python scripts/extract_text_features.py --category $CATEGORY
    else
        python scripts/extract_text_features.py --category $CATEGORY --device $DEVICE
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 文本特征提取失败！${NC}"
        echo "请检查是否已安装 transformers:"
        echo "  pip install transformers"
        exit 1
    fi
    echo -e "${GREEN}✓ 文本特征提取完成${NC}"
fi
echo ""

# Step 4: Extract image features (真实ResNet特征)
echo -e "${YELLOW}[Step 4/6] 提取真实ResNet图像特征（从URL下载）...${NC}"
echo "-------------------------------------------------------------"
if [ -f "data/processed/${CATEGORY}/image_features.pkl" ]; then
    echo -e "${GREEN}✓ 图像特征已存在，跳过提取${NC}"
    echo -e "${BLUE}  如需重新提取真实特征，请删除: data/processed/${CATEGORY}/image_features.pkl${NC}"
else
    echo -e "${BLUE}  注意: 将从Amazon URL下载真实图片并提取ResNet特征${NC}"
    echo -e "${BLUE}  这可能需要较长时间（约5-30分钟，取决于网速）${NC}"
    
    if [ "$DEVICE" = "auto" ]; then
        python scripts/extract_image_features.py \
            --category $CATEGORY \
            --batch_size 32 \
            --timeout 10 \
            --num_workers 16
    else
        python scripts/extract_image_features.py \
            --category $CATEGORY \
            --device $DEVICE \
            --batch_size 32 \
            --timeout 10 \
            --num_workers 16
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 图像特征提取失败！${NC}"
        echo "请检查是否已安装 torchvision, pillow, requests:"
        echo "  pip install torchvision pillow requests"
        exit 1
    fi
    
    # 检查提取结果
    SUCCESS_RATE=$(python -c "
import pickle
import sys
try:
    with open('data/processed/${CATEGORY}/image_features.pkl', 'rb') as f:
        pass
    print('done')
except:
    print('error')
" 2>/dev/null)
    
    if [ "$SUCCESS_RATE" = "done" ]; then
        echo -e "${GREEN}✓ 图像特征提取完成${NC}"
    else
        echo -e "${RED}✗ 图像特征文件损坏${NC}"
        exit 1
    fi
fi
echo ""

# Step 5: Quick test
echo -e "${YELLOW}[Step 5/6] 运行快速测试...${NC}"
echo "-------------------------------------------------------------"
python scripts/quick_test.py
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 快速测试失败！${NC}"
    exit 1
fi
echo ""

# Step 6: Training
echo -e "${YELLOW}[Step 6/6] 开始模型训练...${NC}"
echo "-------------------------------------------------------------"
# ⭐ 使用配置文件 + 命令行参数覆盖（只在显式指定时覆盖）
TRAIN_CMD="python train_amazon.py --config config_example.yaml --category $CATEGORY --filter_train_items"

# 只在用户明确指定时才覆盖config
if [ $# -ge 2 ]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi
if [ $# -ge 3 ]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi

echo "执行命令: $TRAIN_CMD"
eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 训练失败！${NC}"
    exit 1
fi
echo ""

# Results
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  训练完成！${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

if [ -f "checkpoints/${CATEGORY}/results.json" ]; then
    echo -e "${BLUE}最终结果:${NC}"
    cat checkpoints/${CATEGORY}/results.json | python -m json.tool
else
    echo -e "${YELLOW}⚠ 未找到结果文件${NC}"
fi

echo ""
echo -e "${GREEN}生成的文件:${NC}"
echo "  数据文件:"
echo "    - data/processed/${CATEGORY}/train_sequences.pkl"
echo "    - data/processed/${CATEGORY}/val_sequences.pkl"
echo "    - data/processed/${CATEGORY}/test_sequences.pkl"
echo "  特征文件:"
echo "    - data/processed/${CATEGORY}/text_features.pkl (BERT, 768维)"
echo "    - data/processed/${CATEGORY}/image_features.pkl (ResNet, 2048维)"
echo "  模型文件:"
echo "    - checkpoints/${CATEGORY}/best_model.pt"
echo "    - checkpoints/${CATEGORY}/results.json"
echo ""
echo -e "${GREEN}================================================================${NC}"
