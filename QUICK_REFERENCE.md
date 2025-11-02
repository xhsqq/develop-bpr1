# 🚀 快速参考

## 常用命令

### 基础训练
```bash
# 标准训练
python train_amazon.py --category beauty --epochs 50

# 带TensorBoard
python train_amazon.py --category beauty --use_tensorboard

# 使用配置文件
python train_amazon.py --config config_example.yaml
```

### 消融实验
```bash
# 单个消融实验
python train_amazon.py --ablation_no_causal --use_tensorboard

# 批量运行所有消融实验
./scripts/run_ablation_study.sh beauty 50 256 cuda
```

### TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir=logs/

# 指定端口
tensorboard --logdir=logs/ --port=6007
```

---

## 消融选项

| 参数 | 说明 |
|------|------|
| `--ablation_no_disentangled` | 移除解耦表征 |
| `--ablation_no_causal` | 移除因果推断 |
| `--ablation_no_quantum` | 移除量子编码器 |
| `--ablation_no_multimodal` | 移除多模态特征 |
| `--ablation_text_only` | 仅文本特征 |
| `--ablation_image_only` | 仅图像特征 |

---

## 重要参数

### 模型参数
```bash
--hidden_dim 256                # 隐藏层维度
--item_embed_dim 128            # 物品嵌入维度
--disentangled_dim 64           # 解耦维度
--num_interests 4               # 兴趣数量
```

### 训练参数
```bash
--batch_size 256                # 批次大小
--epochs 50                     # 训练轮数
--lr 0.001                      # 学习率
--eval_interval 5               # 评估间隔
```

### 损失权重
```bash
--alpha_recon 0.5               # 重构损失
--alpha_causal 0.1              # 因果损失
--alpha_diversity 0.05          # 多样性损失
--alpha_orthogonality 0.1       # 正交性损失
```

---

## 文件结构

```
develop/
├── train_amazon.py             # 主训练脚本
├── config_example.yaml         # 配置示例
├── config_ablation.yaml        # 消融配置
├── requirements.txt            # 依赖列表
│
├── data/                       # 数据处理
│   ├── download_amazon.py
│   ├── preprocess_amazon.py
│   └── dataset.py
│
├── models/                     # 模型定义
│   ├── multimodal_recommender.py
│   ├── disentangled_representation.py
│   ├── causal_inference.py
│   └── quantum_inspired_encoder.py
│
├── scripts/                    # 脚本工具
│   ├── run_full_pipeline.sh
│   ├── run_ablation_study.sh
│   ├── extract_text_features.py
│   └── extract_image_features.py
│
├── logs/                       # TensorBoard日志
├── checkpoints/                # 模型检查点
└── ablation_results/           # 消融实验结果
```

---

## 新功能速查

### ✅ TensorBoard
- **启用**: `--use_tensorboard`
- **查看**: `tensorboard --logdir=logs/`
- **记录**: 训练/验证/测试指标 + 超参数

### ✅ YAML配置
- **使用**: `--config config.yaml`
- **优先级**: 命令行 > 配置文件 > 默认值
- **包含**: 模型/训练/消融配置

### ✅ 消融实验
- **单个**: `--ablation_*`
- **批量**: `./scripts/run_ablation_study.sh`
- **结果**: `ablation_results/*/ablation_results.csv`

---

## 文档导航

| 文档 | 内容 |
|------|------|
| `README.md` | 项目概述 |
| `DATA_GUIDE.md` | 数据说明 |
| `IMPROVEMENTS_SUMMARY.md` | 训练优化 |
| `ABLATION_GUIDE.md` | 消融实验详细指南 |
| `UPDATE_SUMMARY.md` | 最新功能更新 |
| `QUICK_REFERENCE.md` | 本文档 |

---

## 典型工作流

```bash
# 1. 准备数据
./scripts/run_full_pipeline.sh beauty

# 2. 基线训练
python train_amazon.py \
    --category beauty \
    --use_tensorboard \
    --exp_name baseline

# 3. 消融实验
./scripts/run_ablation_study.sh beauty 50 256 cuda

# 4. 查看结果
tensorboard --logdir=logs/
cat ablation_results/*/ablation_results.csv
```

---

**版本**: v0.3.0  
**更新**: 2025-10-31

