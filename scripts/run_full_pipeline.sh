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

# ⭐ 数据集类别（可选：beauty, games, sports, all）
CATEGORY="${1:-all}"  # 默认处理所有数据集

# 数据目录
DATA_DIR="data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"

# 模型配置
CONFIG_FILE="config.yaml"

# GPU配置
GPU_ID=0
USE_GPU=true

# 打印数据集配置
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}数据集配置: ${CATEGORY}${NC}"
echo -e "${BLUE}========================================${NC}"

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
mkdir -p ${PROCESSED_DIR}

# ⭐ 下载数据集（支持beauty/games/sports/all）
print_msg "下载Amazon数据集（${CATEGORY}）..."
python data/download_amazon.py \
    --category ${CATEGORY} \
    --data_dir ${RAW_DIR}

if [ $? -ne 0 ]; then
    print_error "数据下载失败"
    exit 1
fi

# ⭐ 预处理数据（支持beauty/games/sports/all）
print_msg "预处理数据（${CATEGORY}）..."
python data/preprocess_amazon.py \
    --category ${CATEGORY} \
    --raw_dir ${RAW_DIR} \
    --processed_dir ${PROCESSED_DIR} \
    --min_interactions 5 \
    --max_seq_length 50

if [ $? -ne 0 ]; then
    print_error "数据预处理失败"
    exit 1
fi

print_msg "✓ 数据准备完成"

# ============================================
# Step 3: 特征提取
# ============================================

print_header "Step 3: 多模态特征提取"

# ⭐ 确定要处理的数据集
if [ "${CATEGORY}" = "all" ]; then
    CATEGORIES=("beauty" "games" "sports")
else
    CATEGORIES=("${CATEGORY}")
fi

# ⭐ 对每个数据集提取特征
for CAT in "${CATEGORIES[@]}"; do
    print_msg "处理数据集: ${CAT}"

    CAT_DIR="${PROCESSED_DIR}/${CAT}"

    # 提取文本特征（BERT）
    print_msg "提取文本特征（BERT）..."
    python scripts/extract_text_features.py \
        --category ${CAT} \
        --data_dir ${PROCESSED_DIR} \
        --model_name "bert-base-uncased" \
        --batch_size 64

    if [ $? -ne 0 ]; then
        print_error "${CAT} 文本特征提取失败"
        exit 1
    fi

    # 提取图像特征（ResNet）
    print_msg "提取图像特征（ResNet）..."
    python scripts/extract_image_features.py \
        --category ${CAT} \
        --data_dir ${PROCESSED_DIR} \
        --model_name "resnet50" \
        --batch_size 64

    if [ $? -ne 0 ]; then
        print_error "${CAT} 图像特征提取失败"
        exit 1
    fi

    print_msg "✓ ${CAT} 特征提取完成"
done

print_msg "✓ 所有特征提取完成"

# ============================================
# Step 4: 模型训练
# ============================================

print_header "Step 4: 训练改进版推荐模型"

print_msg "模型配置："
print_msg "  - 维度特定多模态融合: ✓"
print_msg "  - 量子编码器（16个量子态）: ✓"
print_msg "  - SCM因果推断（Pearl三步法）: ✓"

# ⭐ 对每个数据集训练模型
for CAT in "${CATEGORIES[@]}"; do
    print_msg "训练数据集: ${CAT}"

    # 构建训练命令（注意：train.py需要支持--category参数）
    TRAIN_CMD="python train.py --config ${CONFIG_FILE}"

    if [ "$USE_GPU" = true ]; then
        TRAIN_CMD="${TRAIN_CMD} --device cuda:${GPU_ID}"
    else
        TRAIN_CMD="${TRAIN_CMD} --device cpu"
    fi

    # 添加数据集特定参数
    TRAIN_CMD="${TRAIN_CMD} --category ${CAT}"
    TRAIN_CMD="${TRAIN_CMD} --save_dir checkpoints/${CAT}"

    print_msg "开始训练 ${CAT}..."
    print_msg "命令: ${TRAIN_CMD}"

    # 运行训练
    eval ${TRAIN_CMD}

    if [ $? -ne 0 ]; then
        print_warning "${CAT} 训练失败，继续下一个数据集"
        continue
    fi

    print_msg "✓ ${CAT} 训练完成"
done

print_msg "✓ 所有训练完成"

# ============================================
# Step 5: 模型评估
# ============================================

print_header "Step 5: 模型评估"

# ⭐ 对每个数据集评估模型
for CAT in "${CATEGORIES[@]}"; do
    print_msg "评估数据集: ${CAT}"

    # 查找最新的检查点
    CHECKPOINT_DIR="checkpoints/${CAT}"
    LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.pth 2>/dev/null | head -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        print_warning "未找到 ${CAT} 的模型检查点，跳过评估"
        continue
    fi

    print_msg "使用检查点: ${LATEST_CHECKPOINT}"

    # 运行评估
    print_msg "评估模型性能..."
    python train.py \
        --config ${CONFIG_FILE} \
        --category ${CAT} \
        --mode eval \
        --checkpoint ${LATEST_CHECKPOINT}

    if [ $? -ne 0 ]; then
        print_warning "${CAT} 评估失败，继续下一个数据集"
        continue
    fi

    print_msg "✓ ${CAT} 评估完成"
done

print_msg "✓ 所有评估完成"

# ============================================
# Step 6: 生成报告
# ============================================

print_header "Step 6: 生成实验报告"

# 创建报告目录
REPORT_DIR="reports"
mkdir -p ${REPORT_DIR}

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
REPORT_FILE="${REPORT_DIR}/experiment_${CATEGORY}_${TIMESTAMP}.txt"

# 写入报告
cat > ${REPORT_FILE} <<EOF
========================================
实验报告 - 改进版推荐系统（多数据集）
========================================

运行时间: $(date +'%Y-%m-%d %H:%M:%S')
数据集: ${CATEGORY}
处理的数据集: ${CATEGORIES[@]}

模型配置:
---------
- 配置文件: ${CONFIG_FILE}

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

数据集统计:
---------
EOF

# ⭐ 为每个数据集添加统计信息
for CAT in "${CATEGORIES[@]}"; do
    cat >> ${REPORT_FILE} <<EOF

### ${CAT} 数据集
- 处理目录: ${PROCESSED_DIR}/${CAT}
- 文本特征: 768维 (BERT)
- 图像特征: 2048维 (ResNet)
- 检查点目录: checkpoints/${CAT}
EOF

    # 添加统计信息（如果存在）
    STATS_FILE="${PROCESSED_DIR}/${CAT}/statistics.json"
    if [ -f "${STATS_FILE}" ]; then
        echo "- 数据统计:" >> ${REPORT_FILE}
        cat ${STATS_FILE} >> ${REPORT_FILE}
    fi
done

cat >> ${REPORT_FILE} <<EOF

训练配置:
---------
$(cat ${CONFIG_FILE} | grep -A 15 "training:" || echo "配置文件读取失败")

========================================
EOF

print_msg "✓ 报告已生成: ${REPORT_FILE}"

# ============================================
# 完成
# ============================================

print_header "完整流程执行完毕！"

echo -e "${GREEN}总结：${NC}"
echo -e "  1. ✓ 环境检查"
echo -e "  2. ✓ 数据准备（${CATEGORY}）"
echo -e "  3. ✓ 特征提取"
echo -e "  4. ✓ 模型训练"
echo -e "  5. ✓ 模型评估"
echo -e "  6. ✓ 报告生成"
echo ""
echo -e "${GREEN}处理的数据集:${NC} ${CATEGORIES[@]}"
echo -e "${GREEN}实验报告:${NC} ${REPORT_FILE}"
echo ""
echo -e "${BLUE}下一步建议：${NC}"
echo -e "  - 查看TensorBoard: ${YELLOW}tensorboard --logdir runs${NC}"
echo -e "  - 运行消融实验: ${YELLOW}bash scripts/run_ablation_study.sh <category>${NC}"
echo -e "  - 查看数据统计: ${YELLOW}cat data/processed/<category>/statistics.json${NC}"
echo ""
echo -e "${BLUE}使用方法：${NC}"
echo -e "  - 处理所有数据集: ${YELLOW}bash scripts/run_full_pipeline.sh all${NC}"
echo -e "  - 处理beauty数据集: ${YELLOW}bash scripts/run_full_pipeline.sh beauty${NC}"
echo -e "  - 处理games数据集: ${YELLOW}bash scripts/run_full_pipeline.sh games${NC}"
echo -e "  - 处理sports数据集: ${YELLOW}bash scripts/run_full_pipeline.sh sports${NC}"
echo ""

exit 0
