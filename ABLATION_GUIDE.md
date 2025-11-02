# 消融实验指南

## 📋 概述

本指南说明如何使用新增的TensorBoard日志、YAML配置文件和消融实验功能。

---

## 🎯 新增功能

### 1. TensorBoard 日志记录

**功能**: 实时可视化训练过程，对比不同实验

**启用方式**:
```bash
python train_amazon.py \
    --category beauty \
    --use_tensorboard \
    --exp_name my_experiment
```

**查看日志**:
```bash
# 启动TensorBoard
tensorboard --logdir=logs/

# 访问 http://localhost:6006
```

**记录的指标**:
- 训练损失（总损失、推荐损失、解耦损失、因果损失等）
- 验证指标（NDCG、HR、MRR、Recall、Precision）
- 学习率变化
- 超参数和最终测试结果

---

### 2. YAML 配置文件

**功能**: 统一管理超参数，便于实验复现

**使用方式**:
```bash
python train_amazon.py --config config_example.yaml
```

**配置文件结构**:
```yaml
model:
  hidden_dim: 256
  item_embed_dim: 128
  # ...

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  loss_weights:
    recon: 0.5
    causal: 0.1
    diversity: 0.05
    orthogonality: 0.1

ablation:
  no_disentangled: false
  no_causal: false
  # ...
```

**优先级**: 命令行参数 > 配置文件 > 默认值

**示例**:
```bash
# 使用配置文件，但覆盖epochs参数
python train_amazon.py \
    --config config_example.yaml \
    --epochs 100
```

---

### 3. 消融实验

**功能**: 系统性评估各模块的贡献

**可用的消融选项**:

| 参数 | 说明 | 效果 |
|------|------|------|
| `--ablation_no_disentangled` | 禁用解耦表征学习 | 评估解耦学习的贡献 |
| `--ablation_no_causal` | 禁用因果推断模块 | 评估因果推断的贡献 |
| `--ablation_no_quantum` | 禁用量子启发编码器 | 评估多兴趣建模的贡献 |
| `--ablation_no_multimodal` | 禁用多模态特征 | 仅使用物品嵌入 |
| `--ablation_text_only` | 仅使用文本特征 | 评估文本特征的贡献 |
| `--ablation_image_only` | 仅使用图像特征 | 评估图像特征的贡献 |

**单个实验示例**:
```bash
# 移除因果推断模块
python train_amazon.py \
    --category beauty \
    --ablation_no_causal \
    --use_tensorboard \
    --exp_name beauty_no_causal
```

**组合消融**:
```bash
# 移除解耦学习和因果推断
python train_amazon.py \
    --category beauty \
    --ablation_no_disentangled \
    --ablation_no_causal \
    --use_tensorboard
```

---

## 🚀 快速开始

### 方式一：单个实验

```bash
# 1. 基线模型（完整模型）
python train_amazon.py \
    --category beauty \
    --epochs 50 \
    --use_tensorboard \
    --exp_name beauty_full_model

# 2. 消融实验：移除解耦表征
python train_amazon.py \
    --category beauty \
    --epochs 50 \
    --ablation_no_disentangled \
    --use_tensorboard \
    --exp_name beauty_no_dis

# 3. 消融实验：仅文本特征
python train_amazon.py \
    --category beauty \
    --epochs 50 \
    --ablation_text_only \
    --use_tensorboard \
    --exp_name beauty_text_only
```

### 方式二：批量运行（推荐）

```bash
# 自动运行所有消融实验
./scripts/run_ablation_study.sh beauty 50 256 cuda
```

**运行内容**:
1. 完整模型（基线）
2. 移除解耦表征
3. 移除因果推断
4. 移除量子编码器
5. 移除多模态特征
6. 仅文本特征
7. 仅图像特征
8. 移除解耦+因果
9. 移除解耦+量子
10. 移除因果+量子
11. 最简模型

**输出结果**:
- 每个实验的详细日志
- 汇总的CSV结果表
- TensorBoard日志（可视化对比）

---

## 📊 结果分析

### 1. 查看TensorBoard

```bash
tensorboard --logdir=logs/
```

**对比实验**:
- 在TensorBoard中选择多个实验
- 对比训练曲线
- 对比最终指标
- 分析超参数影响

### 2. CSV结果

```bash
# 查看消融实验结果汇总
cat ablation_results/beauty_YYYYMMDD_HHMMSS/ablation_results.csv
```

**CSV包含**:
- 实验名称
- 主要指标（NDCG@10, HR@10, MRR等）
- 消融设置标记
- 按NDCG@10排序

### 3. 分析模块贡献

**示例分析**:
```
完整模型: NDCG@10 = 0.0850

移除解耦表征: NDCG@10 = 0.0820 (-3.5%)
移除因果推断: NDCG@10 = 0.0840 (-1.2%)
移除量子编码: NDCG@10 = 0.0830 (-2.4%)
仅文本特征: NDCG@10 = 0.0800 (-5.9%)
仅图像特征: NDCG@10 = 0.0790 (-7.1%)
```

**结论**: 
- 解耦表征贡献最大（+3.5%）
- 文本和图像互补（组合比单独好）
- 因果推断提供稳定增益（+1.2%）

---

## 💡 最佳实践

### 1. 实验命名

使用有意义的实验名称：
```bash
--exp_name beauty_baseline_20241031
--exp_name beauty_no_dis_lr001
--exp_name beauty_text_only_bs512
```

### 2. 记录超参数

始终启用TensorBoard并记录配置：
```bash
--use_tensorboard
--config my_config.yaml
```

### 3. 多次运行

对关键实验运行多次（不同随机种子）：
```bash
for seed in 42 123 456; do
    python train_amazon.py \
        --seed $seed \
        --exp_name beauty_full_seed${seed} \
        --use_tensorboard
done
```

### 4. 保存配置

每个实验的配置会自动保存到 `checkpoints/{exp_name}/config.json`

### 5. GPU内存管理

如果运行多个消融实验，注意GPU内存：
```bash
# 顺序运行
./scripts/run_ablation_study.sh beauty 50 128 cuda

# 或手动控制batch_size
--batch_size 128
```

---

## 📝 配置文件示例

### config_baseline.yaml
```yaml
# 基线配置
model:
  hidden_dim: 256
  item_embed_dim: 128
  disentangled_dim: 64
  num_interests: 4

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  
  loss_weights:
    recon: 0.5
    causal: 0.1
    diversity: 0.05
    orthogonality: 0.1

ablation:
  no_disentangled: false
  no_causal: false
  no_quantum: false
  no_multimodal: false
```

### config_ablation_no_causal.yaml
```yaml
# 移除因果推断
model:
  hidden_dim: 256
  item_embed_dim: 128
  disentangled_dim: 64
  num_interests: 4

training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  
  loss_weights:
    recon: 0.5
    causal: 0.0  # 设为0
    diversity: 0.05
    orthogonality: 0.1

ablation:
  no_disentangled: false
  no_causal: true  # 禁用
  no_quantum: false
  no_multimodal: false
```

---

## 🔍 常见问题

### Q1: TensorBoard不显示数据？
```bash
# 检查日志目录
ls -la logs/

# 确保使用了 --use_tensorboard
python train_amazon.py --use_tensorboard ...
```

### Q2: 如何对比两个实验？
在TensorBoard中：
1. 勾选要对比的实验
2. 切换到对比视图
3. 查看指标差异

### Q3: 消融实验运行时间？
- 单个实验：约30-40分钟（beauty数据集，50 epochs）
- 完整消融研究（11个实验）：约6-7小时

### Q4: 如何提前终止消融研究？
按 `Ctrl+C`，已完成的实验结果会保留。

### Q5: 配置文件和命令行冲突？
命令行参数优先级更高，会覆盖配置文件。

---

## 📈 示例工作流程

### 完整的消融实验流程

```bash
# 1. 准备数据和特征
./scripts/run_full_pipeline.sh beauty 256 50 cuda

# 2. 运行基线模型
python train_amazon.py \
    --category beauty \
    --epochs 50 \
    --use_tensorboard \
    --filter_train_items \
    --exp_name beauty_baseline

# 3. 运行所有消融实验
./scripts/run_ablation_study.sh beauty 50 256 cuda

# 4. 查看TensorBoard
tensorboard --logdir=logs/ &

# 5. 分析结果
cat ablation_results/*/ablation_results.csv

# 6. 根据结果调整模型
# 例如，如果发现某个模块贡献小，可以简化模型
```

---

## 🎯 预期结果模板

| 实验 | NDCG@10 | HR@10 | MRR | 变化 |
|------|---------|-------|-----|------|
| 完整模型 | 0.0850 | 0.1500 | 0.0680 | 基线 |
| 无解耦 | 0.0820 | 0.1450 | 0.0660 | -3.5% |
| 无因果 | 0.0840 | 0.1480 | 0.0670 | -1.2% |
| 无量子 | 0.0830 | 0.1465 | 0.0665 | -2.4% |
| 无多模态 | 0.0750 | 0.1320 | 0.0600 | -11.8% |
| 仅文本 | 0.0800 | 0.1400 | 0.0640 | -5.9% |
| 仅图像 | 0.0790 | 0.1380 | 0.0635 | -7.1% |

*注：实际结果会因数据集和随机种子而异*

---

## 🔧 高级用法

### 1. 自定义消融组合

```python
# 创建自定义配置
config = {
    'model': {...},
    'training': {...},
    'ablation': {
        'no_disentangled': True,
        'no_causal': True,
        # 移除多个模块
    }
}

# 保存为YAML
with open('my_ablation.yaml', 'w') as f:
    yaml.dump(config, f)

# 运行
python train_amazon.py --config my_ablation.yaml
```

### 2. 超参数搜索

```bash
# 搜索学习率
for lr in 0.0001 0.001 0.01; do
    python train_amazon.py \
        --lr $lr \
        --exp_name beauty_lr${lr} \
        --use_tensorboard
done
```

### 3. 多数据集对比

```bash
# 在不同数据集上运行相同实验
for cat in beauty games sports; do
    python train_amazon.py \
        --category $cat \
        --exp_name ${cat}_baseline \
        --use_tensorboard
done
```

---

## 📚 参考

- **TensorBoard文档**: https://www.tensorflow.org/tensorboard
- **YAML语法**: https://yaml.org/
- **消融研究最佳实践**: 见论文相关章节

---

**更新日期**: 2025-10-31  
**版本**: v0.3.0

