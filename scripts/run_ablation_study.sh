#!/bin/bash
# Ablation Study Runner
# 批量运行消融实验
# 用于系统性评估各个模块的贡献

set -e

# 配置
CATEGORY=${1:-beauty}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-256}
DEVICE=${4:-cuda}

echo "=================================================================="
echo "  Ablation Study Runner"
echo "=================================================================="
echo "Category: $CATEGORY"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=================================================================="
echo ""

# 创建消融实验结果目录
RESULTS_DIR="ablation_results/${CATEGORY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# 基础训练命令
BASE_CMD="python train_amazon.py \
    --category $CATEGORY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --use_tensorboard \
    --filter_train_items"

# 定义消融实验列表
declare -a experiments=(
    "full_model:完整模型（基线）:"
    "no_disentangled:移除解耦表征:--ablation_no_disentangled"
    "no_causal:移除因果推断:--ablation_no_causal"
    "no_quantum:移除量子编码器:--ablation_no_quantum"
    "no_multimodal:移除多模态特征:--ablation_no_multimodal"
    "text_only:仅文本特征:--ablation_text_only"
    "image_only:仅图像特征:--ablation_image_only"
    "no_dis_causal:移除解耦+因果:--ablation_no_disentangled --ablation_no_causal"
    "no_dis_quantum:移除解耦+量子:--ablation_no_disentangled --ablation_no_quantum"
    "no_causal_quantum:移除因果+量子:--ablation_no_causal --ablation_no_quantum"
    "minimal:最简模型:--ablation_no_disentangled --ablation_no_causal --ablation_no_quantum --ablation_no_multimodal"
)

# 运行实验
total=${#experiments[@]}
current=0

for exp in "${experiments[@]}"; do
    IFS=':' read -ra PARTS <<< "$exp"
    exp_name="${PARTS[0]}"
    exp_desc="${PARTS[1]}"
    exp_args="${PARTS[2]}"
    
    current=$((current + 1))
    
    echo ""
    echo "=================================================================="
    echo "  Experiment $current/$total: $exp_desc"
    echo "  Name: $exp_name"
    echo "=================================================================="
    echo ""
    
    # 运行实验
    exp_log="$RESULTS_DIR/${exp_name}.log"
    
    if [ -z "$exp_args" ]; then
        # 基线模型（无消融）
        $BASE_CMD --exp_name "${CATEGORY}_${exp_name}" 2>&1 | tee $exp_log
    else
        # 消融实验
        $BASE_CMD $exp_args --exp_name "${CATEGORY}_${exp_name}" 2>&1 | tee $exp_log
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment completed: $exp_name"
    else
        echo "✗ Experiment failed: $exp_name"
        echo "See log: $exp_log"
    fi
    
    echo ""
done

# 收集结果
echo "=================================================================="
echo "  Collecting Results"
echo "=================================================================="
echo ""

python - <<EOF
import json
import os
import pandas as pd
from pathlib import Path

results_dir = "$RESULTS_DIR"
checkpoints_dir = "checkpoints"

# 收集所有结果
all_results = []

for exp_dir in Path(checkpoints_dir).glob("${CATEGORY}_*"):
    results_file = exp_dir / "results.json"
    
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        
        exp_name = data.get('exp_name', exp_dir.name)
        test_metrics = data.get('test_metrics', {})
        ablation = data.get('ablation_settings', {})
        
        # 提取关键指标
        result = {
            'Experiment': exp_name,
            'NDCG@10': test_metrics.get('NDCG@10', 0),
            'HR@10': test_metrics.get('HR@10', 0),
            'MRR': test_metrics.get('MRR', 0),
            'Recall@10': test_metrics.get('Recall@10', 0),
            'Precision@10': test_metrics.get('Precision@10', 0),
            'No_Disentangled': ablation.get('no_disentangled', False),
            'No_Causal': ablation.get('no_causal', False),
            'No_Quantum': ablation.get('no_quantum', False),
            'No_Multimodal': ablation.get('no_multimodal', False),
            'Text_Only': ablation.get('text_only', False),
            'Image_Only': ablation.get('image_only', False)
        }
        
        all_results.append(result)

if all_results:
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 按NDCG@10排序
    df = df.sort_values('NDCG@10', ascending=False)
    
    # 保存CSV
    csv_path = os.path.join(results_dir, 'ablation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # 打印表格
    print("\n" + "="*100)
    print("Ablation Study Results")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # 计算各模块贡献
    print("\nModule Contributions (vs Full Model):")
    print("-"*100)
    
    full_model = df[df['Experiment'].str.contains('full_model')]
    if not full_model.empty:
        baseline_ndcg = full_model['NDCG@10'].values[0]
        
        for _, row in df.iterrows():
            if 'full_model' not in row['Experiment']:
                diff = (row['NDCG@10'] - baseline_ndcg) * 100
                print(f"{row['Experiment']:30s}: NDCG@10={row['NDCG@10']:.4f} ({diff:+.2f}%)")
    
    print("-"*100)
else:
    print("✗ No results found")

EOF

echo ""
echo "=================================================================="
echo "  Ablation Study Completed!"
echo "=================================================================="
echo ""
echo "Results directory: $RESULTS_DIR"
echo "TensorBoard logs: logs/"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=logs/"
echo ""
echo "To compare experiments:"
echo "  cat $RESULTS_DIR/ablation_results.csv"
echo ""

