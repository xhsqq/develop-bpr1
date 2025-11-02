#!/bin/bash
# ============================================
# 消融实验脚本：验证各个模块的贡献
# 适配改进版模型：维度特定融合 + 量子编码 + SCM因果推断
# ============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}\n"
}

# ============================================
# 配置参数
# ============================================

# ⭐ 数据集类别（可选：beauty, games, sports）
CATEGORY="${1:-beauty}"  # 默认使用beauty数据集

# 基础配置
BASE_CONFIG="config.yaml"
DATA_DIR="data/processed/${CATEGORY}"
CHECKPOINT_DIR="checkpoints/${CATEGORY}_ablation"
RESULTS_DIR="ablation_results/${CATEGORY}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}消融实验 - 数据集: ${CATEGORY}${NC}"
echo -e "${BLUE}========================================${NC}"

# GPU配置
GPU_ID=0

# 训练配置
NUM_EPOCHS=30  # 消融实验使用较少的epoch
BATCH_SIZE=256

# 创建结果目录
mkdir -p ${RESULTS_DIR}

# 时间戳
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# ============================================
# 消融实验配置
# ============================================

# 实验1: 完整模型（所有改进启用）
EXP1_NAME="full_model"
EXP1_DESC="完整模型（所有改进）"
EXP1_ARGS="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE}"

# 实验2: 无维度特定融合
EXP2_NAME="no_disentangled_fusion"
EXP2_DESC="移除维度特定融合"
EXP2_ARGS="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --alpha_recon 0.0"

# 实验3: 无量子编码（降级为普通MLP）
EXP3_NAME="no_quantum"
EXP3_DESC="移除量子编码器（使用4个兴趣）"
EXP3_ARGS="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_interests 4 --alpha_diversity 0.0 --alpha_orthogonality 0.0"

# 实验4: 无SCM因果推断
EXP4_NAME="no_causal"
EXP4_DESC="移除SCM因果推断"
EXP4_ARGS="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --alpha_causal 0.0"

# 实验5: 基线模型（所有改进禁用）
EXP5_NAME="baseline"
EXP5_DESC="基线模型（无任何改进）"
EXP5_ARGS="--num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --alpha_recon 0.0 --alpha_causal 0.0 --alpha_diversity 0.0 --alpha_orthogonality 0.0 --num_interests 4"

# ============================================
# 实验函数
# ============================================

run_experiment() {
    local exp_name=$1
    local exp_desc=$2
    local exp_args=$3

    print_header "实验: ${exp_desc}"

    # 创建实验目录
    local exp_dir="${RESULTS_DIR}/${exp_name}_${TIMESTAMP}"
    mkdir -p ${exp_dir}

    # 日志文件
    local log_file="${exp_dir}/train.log"
    local result_file="${exp_dir}/results.txt"

    print_msg "实验配置: ${exp_args}"
    print_msg "日志文件: ${log_file}"

    # 运行训练
    print_msg "开始训练..."
    python train.py \
        --config ${BASE_CONFIG} \
        --category ${CATEGORY} \
        --data_dir ${DATA_DIR} \
        --checkpoint_dir ${exp_dir}/checkpoints \
        --log_dir ${exp_dir}/logs \
        ${exp_args} \
        --gpu_ids ${GPU_ID} \
        2>&1 | tee ${log_file}

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        print_error "实验 ${exp_name} 失败"
        echo "FAILED" > ${result_file}
        return 1
    fi

    # 查找最新的检查点
    local checkpoint=$(ls -t ${exp_dir}/checkpoints/*.pth 2>/dev/null | head -n 1)

    if [ -z "$checkpoint" ]; then
        print_warning "未找到检查点，跳过评估"
        echo "NO_CHECKPOINT" > ${result_file}
        return 0
    fi

    # 运行评估
    print_msg "评估模型..."
    python train.py \
        --config ${BASE_CONFIG} \
        --category ${CATEGORY} \
        --mode eval \
        --checkpoint ${checkpoint} \
        --gpu_ids ${GPU_ID} \
        2>&1 | tee -a ${log_file}

    # 提取评估结果
    extract_results ${log_file} ${result_file}

    print_msg "✓ 实验 ${exp_name} 完成"
    echo ""
}

# ============================================
# 提取结果函数
# ============================================

extract_results() {
    local log_file=$1
    local result_file=$2

    # 提取Recall@10, NDCG@10, HR@10
    local recall10=$(grep "Recall@10" ${log_file} | tail -n 1 | awk '{print $NF}')
    local ndcg10=$(grep "NDCG@10" ${log_file} | tail -n 1 | awk '{print $NF}')
    local hr10=$(grep "HR@10" ${log_file} | tail -n 1 | awk '{print $NF}')

    # 写入结果文件
    cat > ${result_file} <<EOF
Recall@10: ${recall10:-N/A}
NDCG@10: ${ndcg10:-N/A}
HR@10: ${hr10:-N/A}
EOF
}

# ============================================
# 主流程
# ============================================

print_header "消融实验开始"

echo -e "${BLUE}实验计划：${NC}"
echo -e "  1. ${EXP1_DESC}"
echo -e "  2. ${EXP2_DESC}"
echo -e "  3. ${EXP3_DESC}"
echo -e "  4. ${EXP4_DESC}"
echo -e "  5. ${EXP5_DESC}"
echo ""

read -p "是否继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_msg "取消实验"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)

# 运行所有实验
run_experiment "${EXP1_NAME}" "${EXP1_DESC}" "${EXP1_ARGS}"
run_experiment "${EXP2_NAME}" "${EXP2_DESC}" "${EXP2_ARGS}"
run_experiment "${EXP3_NAME}" "${EXP3_DESC}" "${EXP3_ARGS}"
run_experiment "${EXP4_NAME}" "${EXP4_DESC}" "${EXP4_ARGS}"
run_experiment "${EXP5_NAME}" "${EXP5_DESC}" "${EXP5_ARGS}"

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

# ============================================
# 生成汇总报告
# ============================================

print_header "生成汇总报告"

SUMMARY_FILE="${RESULTS_DIR}/ablation_summary_${TIMESTAMP}.txt"

cat > ${SUMMARY_FILE} <<EOF
========================================
消融实验汇总报告
========================================

运行时间: $(date +'%Y-%m-%d %H:%M:%S')
总耗时: ${HOURS}小时 ${MINUTES}分钟

实验配置:
---------
- 训练轮数: ${NUM_EPOCHS}
- 批大小: ${BATCH_SIZE}
- GPU: ${GPU_ID}

========================================
实验结果
========================================

EOF

# 读取每个实验的结果
for exp_name in "${EXP1_NAME}" "${EXP2_NAME}" "${EXP3_NAME}" "${EXP4_NAME}" "${EXP5_NAME}"; do
    exp_dir=$(ls -td ${RESULTS_DIR}/${exp_name}_* 2>/dev/null | head -n 1)

    if [ -n "$exp_dir" ]; then
        result_file="${exp_dir}/results.txt"

        case $exp_name in
            "${EXP1_NAME}") exp_desc="${EXP1_DESC}" ;;
            "${EXP2_NAME}") exp_desc="${EXP2_DESC}" ;;
            "${EXP3_NAME}") exp_desc="${EXP3_DESC}" ;;
            "${EXP4_NAME}") exp_desc="${EXP4_DESC}" ;;
            "${EXP5_NAME}") exp_desc="${EXP5_DESC}" ;;
        esac

        echo "----------------------------------------" >> ${SUMMARY_FILE}
        echo "${exp_desc}" >> ${SUMMARY_FILE}
        echo "----------------------------------------" >> ${SUMMARY_FILE}

        if [ -f "$result_file" ]; then
            cat ${result_file} >> ${SUMMARY_FILE}
        else
            echo "结果文件不存在" >> ${SUMMARY_FILE}
        fi

        echo "" >> ${SUMMARY_FILE}
    fi
done

# 生成Markdown表格
cat >> ${SUMMARY_FILE} <<EOF

========================================
Markdown表格（可直接用于论文）
========================================

| 实验 | Recall@10 | NDCG@10 | HR@10 |
|------|-----------|---------|-------|
EOF

for exp_name in "${EXP1_NAME}" "${EXP2_NAME}" "${EXP3_NAME}" "${EXP4_NAME}" "${EXP5_NAME}"; do
    exp_dir=$(ls -td ${RESULTS_DIR}/${exp_name}_* 2>/dev/null | head -n 1)

    if [ -n "$exp_dir" ]; then
        result_file="${exp_dir}/results.txt"

        case $exp_name in
            "${EXP1_NAME}") exp_desc="完整模型" ;;
            "${EXP2_NAME}") exp_desc="无维度融合" ;;
            "${EXP3_NAME}") exp_desc="无量子编码" ;;
            "${EXP4_NAME}") exp_desc="无SCM因果" ;;
            "${EXP5_NAME}") exp_desc="基线模型" ;;
        esac

        if [ -f "$result_file" ]; then
            recall=$(grep "Recall@10" ${result_file} | awk '{print $2}')
            ndcg=$(grep "NDCG@10" ${result_file} | awk '{print $2}')
            hr=$(grep "HR@10" ${result_file} | awk '{print $2}')

            echo "| ${exp_desc} | ${recall:-N/A} | ${ndcg:-N/A} | ${hr:-N/A} |" >> ${SUMMARY_FILE}
        fi
    fi
done

cat >> ${SUMMARY_FILE} <<EOF

========================================
分析与结论
========================================

1. 维度特定融合的贡献:
   - 比较完整模型与"无维度融合"实验
   - 预期提升: 提高可解释性，减少模态偏差

2. 量子编码器的贡献:
   - 比较完整模型与"无量子编码"实验
   - 预期提升: 更丰富的多兴趣建模（16态 vs 4态）

3. SCM因果推断的贡献:
   - 比较完整模型与"无SCM因果"实验
   - 预期提升: 理论严谨的反事实推理

4. 整体改进:
   - 比较完整模型与基线模型
   - 预期提升: 三大改进的协同效应

========================================
EOF

print_msg "✓ 汇总报告已生成: ${SUMMARY_FILE}"

# ============================================
# 生成对比图表脚本（Python）
# ============================================

PLOT_SCRIPT="${RESULTS_DIR}/plot_ablation.py"

cat > ${PLOT_SCRIPT} <<'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验结果可视化脚本
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import sys

def parse_results(summary_file):
    """解析汇总文件"""
    experiments = []
    recalls = []
    ndcgs = []
    hrs = []

    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取表格内容
    table_match = re.search(r'\| 实验 \| Recall@10.*?\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if not table_match:
        print("未找到结果表格")
        return None

    table_lines = table_match.group(1).strip().split('\n')

    for line in table_lines:
        if line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) == 4:
                exp_name, recall, ndcg, hr = parts

                if recall != 'N/A' and ndcg != 'N/A' and hr != 'N/A':
                    experiments.append(exp_name)
                    recalls.append(float(recall))
                    ndcgs.append(float(ndcg))
                    hrs.append(float(hr))

    return experiments, recalls, ndcgs, hrs

def plot_results(experiments, recalls, ndcgs, hrs, output_file):
    """绘制对比图"""
    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, recalls, width, label='Recall@10', color='#2E86AB')
    bars2 = ax.bar(x, ndcgs, width, label='NDCG@10', color='#A23B72')
    bars3 = ax.bar(x + width, hrs, width, label='HR@10', color='#F18F01')

    ax.set_xlabel('实验配置', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('消融实验结果对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python plot_ablation.py <summary_file>")
        sys.exit(1)

    summary_file = sys.argv[1]
    output_file = summary_file.replace('.txt', '.png')

    result = parse_results(summary_file)
    if result:
        experiments, recalls, ndcgs, hrs = result
        plot_results(experiments, recalls, ndcgs, hrs, output_file)
    else:
        print("解析结果失败")
EOF

chmod +x ${PLOT_SCRIPT}

print_msg "可视化脚本已生成: ${PLOT_SCRIPT}"
print_msg "运行可视化: python ${PLOT_SCRIPT} ${SUMMARY_FILE}"

# ============================================
# 完成
# ============================================

print_header "消融实验完成！"

echo -e "${GREEN}总结：${NC}"
echo -e "  - 总耗时: ${HOURS}小时 ${MINUTES}分钟"
echo -e "  - 实验数量: 5个"
echo -e "  - 结果目录: ${RESULTS_DIR}"
echo ""
echo -e "${GREEN}输出文件：${NC}"
echo -e "  - 汇总报告: ${YELLOW}${SUMMARY_FILE}${NC}"
echo -e "  - 可视化脚本: ${YELLOW}${PLOT_SCRIPT}${NC}"
echo ""
echo -e "${BLUE}下一步建议：${NC}"
echo -e "  1. 查看汇总报告: ${YELLOW}cat ${SUMMARY_FILE}${NC}"
echo -e "  2. 生成对比图: ${YELLOW}python ${PLOT_SCRIPT} ${SUMMARY_FILE}${NC}"
echo -e "  3. 将表格复制到论文中"
echo ""

exit 0
