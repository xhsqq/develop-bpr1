#!/bin/bash
# ============================================
# 完整流程脚本：一键运行数据准备 + 特征提取 + 训练 + 评估
# 适配改进版模型：维度特定融合 + 量子编码 + SCM因果推断
# ============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_msg() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# ============================================
# 配置参数
# ============================================

# 数据目录
DATA_DIR="data"
RAW_DIR="${DATA_DIR}/raw"
FEATURE_DIR="${DATA_DIR}/features"

# 模型配置
CONFIG_FILE="config.yaml"

# GPU配置
GPU_ID=0
USE_GPU=true

# ============================================
# Step 1: 环境检查
# ============================================

print_header "Step 1: 环境检查"

# 检查Python
if ! command -v python &> /dev/null; then
    print_error "Python未找到，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_msg "Python版本: ${PYTHON_VERSION}"

# 检查必要的Python包
print_msg "检查依赖包..."
python -c "import torch; import transformers; import torchvision" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "依赖包缺失，正在安装..."
    pip install -r requirements.txt
fi

# 检查GPU
if [ "$USE_GPU" = true ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'" 2>/dev/null
    if [ $? -eq 0 ]; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        print_msg "检测到 ${GPU_COUNT} 个GPU: ${GPU_NAME}"
    else
        print_warning "GPU不可用，将使用CPU训练（较慢）"
        USE_GPU=false
    fi
fi

print_msg "✓ 环境检查完成"

# ============================================
# Step 2: 数据准备
# ============================================

print_header "Step 2: 数据准备"

# 创建目录
mkdir -p ${RAW_DIR}
mkdir -p ${FEATURE_DIR}

# 检查原始数据是否存在
if [ ! -f "${RAW_DIR}/reviews_Fashion.json.gz" ]; then
    print_msg "下载Amazon Fashion数据集..."
    python data/download_amazon.py --output_dir ${RAW_DIR}

    if [ $? -ne 0 ]; then
        print_error "数据下载失败"
        exit 1
    fi
else
    print_msg "原始数据已存在，跳过下载"
fi

# 预处理数据
if [ ! -f "${RAW_DIR}/processed_data.pkl" ]; then
    print_msg "预处理数据..."
    python data/preprocess_amazon.py \
        --input_dir ${RAW_DIR} \
        --output_dir ${RAW_DIR}

    if [ $? -ne 0 ]; then
        print_error "数据预处理失败"
        exit 1
    fi
else
    print_msg "预处理数据已存在，跳过预处理"
fi

print_msg "✓ 数据准备完成"

# ============================================
# Step 3: 特征提取
# ============================================

print_header "Step 3: 多模态特征提取"

# 提取文本特征（BERT）
if [ ! -f "${FEATURE_DIR}/text_features.pkl" ]; then
    print_msg "提取文本特征（BERT）..."
    python scripts/extract_text_features.py \
        --data_dir ${RAW_DIR} \
        --output_dir ${FEATURE_DIR} \
        --model_name "bert-base-uncased" \
        --batch_size 64

    if [ $? -ne 0 ]; then
        print_error "文本特征提取失败"
        exit 1
    fi
else
    print_msg "文本特征已存在，跳过提取"
fi

# 提取图像特征（ResNet）
if [ ! -f "${FEATURE_DIR}/image_features.pkl" ]; then
    print_msg "提取图像特征（ResNet）..."
    python scripts/extract_image_features.py \
        --data_dir ${RAW_DIR} \
        --output_dir ${FEATURE_DIR} \
        --model_name "resnet50" \
        --batch_size 64

    if [ $? -ne 0 ]; then
        print_error "图像特征提取失败"
        exit 1
    fi
else
    print_msg "图像特征已存在，跳过提取"
fi

print_msg "✓ 特征提取完成"

# ============================================
# Step 4: 模型训练
# ============================================

print_header "Step 4: 训练改进版推荐模型"

print_msg "模型配置："
print_msg "  - 维度特定多模态融合: ✓"
print_msg "  - 量子编码器（16个量子态）: ✓"
print_msg "  - SCM因果推断（Pearl三步法）: ✓"

# 构建训练命令
TRAIN_CMD="python train.py --config ${CONFIG_FILE}"

if [ "$USE_GPU" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --gpu_ids ${GPU_ID}"
else
    TRAIN_CMD="${TRAIN_CMD} --device cpu"
fi

print_msg "开始训练..."
print_msg "命令: ${TRAIN_CMD}"

# 运行训练
eval ${TRAIN_CMD}

if [ $? -ne 0 ]; then
    print_error "训练失败"
    exit 1
fi

print_msg "✓ 训练完成"

# ============================================
# Step 5: 模型评估
# ============================================

print_header "Step 5: 模型评估"

# 查找最新的检查点
CHECKPOINT_DIR="checkpoints"
LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.pth 2>/dev/null | head -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    print_error "未找到模型检查点"
    exit 1
fi

print_msg "使用检查点: ${LATEST_CHECKPOINT}"

# 运行评估
print_msg "评估模型性能..."
python train.py \
    --config ${CONFIG_FILE} \
    --mode eval \
    --checkpoint ${LATEST_CHECKPOINT}

if [ $? -ne 0 ]; then
    print_error "评估失败"
    exit 1
fi

print_msg "✓ 评估完成"

# ============================================
# Step 6: 生成报告
# ============================================

print_header "Step 6: 生成实验报告"

# 创建报告目录
REPORT_DIR="reports"
mkdir -p ${REPORT_DIR}

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
REPORT_FILE="${REPORT_DIR}/experiment_${TIMESTAMP}.txt"

# 写入报告
cat > ${REPORT_FILE} <<EOF
========================================
实验报告 - 改进版时尚推荐系统
========================================

运行时间: $(date +'%Y-%m-%d %H:%M:%S')

模型配置:
---------
- 配置文件: ${CONFIG_FILE}
- 检查点: ${LATEST_CHECKPOINT}

架构改进:
---------
1. ✓ 维度特定多模态融合
   - 先解耦再融合
   - 跨模态注意力
   - 可解释性提升

2. ✓ 量子编码器优化
   - 16个量子态（从4增加）
   - 相位编码
   - 幺正干涉矩阵
   - 严格的量子测量

3. ✓ SCM因果推断
   - Pearl三步反事实推理
   - Abduction: 推断外生变量
   - Action: 干预操作
   - Prediction: ITE计算

数据统计:
---------
- 数据目录: ${FEATURE_DIR}
- 文本特征: 768维 (BERT)
- 图像特征: 2048维 (ResNet)

训练配置:
---------
$(cat ${CONFIG_FILE} | grep -A 10 "training:")

评估结果:
---------
$(tail -n 20 logs/train.log 2>/dev/null || echo "日志文件不存在")

========================================
EOF

print_msg "✓ 报告已生成: ${REPORT_FILE}"

# ============================================
# 完成
# ============================================

print_header "完整流程执行完毕！"

echo -e "${GREEN}总结：${NC}"
echo -e "  1. ✓ 环境检查"
echo -e "  2. ✓ 数据准备"
echo -e "  3. ✓ 特征提取"
echo -e "  4. ✓ 模型训练"
echo -e "  5. ✓ 模型评估"
echo -e "  6. ✓ 报告生成"
echo ""
echo -e "${GREEN}模型检查点:${NC} ${LATEST_CHECKPOINT}"
echo -e "${GREEN}实验报告:${NC} ${REPORT_FILE}"
echo ""
echo -e "${BLUE}下一步建议：${NC}"
echo -e "  - 查看TensorBoard: ${YELLOW}tensorboard --logdir runs${NC}"
echo -e "  - 运行消融实验: ${YELLOW}bash scripts/run_ablation_study.sh${NC}"
echo -e "  - 可视化结果: ${YELLOW}python scripts/visualize_results.py${NC}"
echo ""

exit 0
