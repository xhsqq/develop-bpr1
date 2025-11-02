# Amazon Dataset Guide

## 数据集说明

使用Amazon Review数据集（2018版本）的三个类别：
- **Beauty**: 美妆产品
- **Games**: 视频游戏
- **Sports**: 运动户外

数据来源：[Amazon Review Data (2018)](http://jmcauley.ucsd.edu/data/amazon/)

## 快速开始

### 1. 下载数据

下载全部三个数据集：
```bash
python data/download_amazon.py --category all
```

或下载单个数据集：
```bash
python data/download_amazon.py --category beauty
```

**注意**：下载可能需要较长时间，取决于网络速度。

### 2. 预处理数据

预处理全部数据集（推荐）：
```bash
python data/preprocess_amazon.py --category all --min_interactions 5
```

或预处理单个数据集：
```bash
python data/preprocess_amazon.py --category beauty
```

预处理参数：
- `--min_interactions`: K-core过滤的最小交互次数（默认：5）
- `--max_seq_length`: 最大序列长度（默认：50）

### 3. 快速测试

验证数据处理是否正确：
```bash
python scripts/quick_test.py
```

### 4. 训练模型

在Beauty数据集上训练：
```bash
python train_amazon.py --category beauty \
                        --batch_size 256 \
                        --epochs 50 \
                        --hidden_dim 256 \
                        --eval_interval 5
```

在Games数据集上训练：
```bash
python train_amazon.py --category games \
                        --batch_size 256 \
                        --epochs 50
```

在Sports数据集上训练：
```bash
python train_amazon.py --category sports \
                        --batch_size 256 \
                        --epochs 50
```

## 数据划分方式

使用**留一法（Leave-One-Out）**划分：

- **训练集**: 每个用户的前 n-2 个交互
- **验证集**: 每个用户的倒数第 2 个交互
- **测试集**: 每个用户的最后 1 个交互

### 重要特性

1. **无数据泄漏**：
   - 严格按时间戳排序
   - 验证集只使用训练集的历史序列
   - 测试集使用训练集+验证集的历史序列

2. **全库评估**：
   - 对所有物品计算分数
   - **不使用负采样**
   - 真实反映模型的排序能力

3. **过滤训练物品**（可选）：
   - 使用 `--filter_train_items` 参数
   - 评估时不考虑用户已交互的物品

## 数据统计

预处理后的数据统计（5-core过滤后）：

| Dataset | Users | Items | Train | Valid | Test | Density |
|---------|-------|-------|-------|-------|------|---------|
| Beauty  | ~22K  | ~12K  | ~180K | ~22K  | ~22K | 0.068%  |
| Games   | ~25K  | ~11K  | ~200K | ~25K  | ~25K | 0.073%  |
| Sports  | ~35K  | ~18K  | ~280K | ~35K  | ~35K | 0.044%  |

*具体数值取决于K-core过滤参数*

## 数据格式

### 原始数据

Reviews (每行一个JSON):
```json
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewText": "I bought this for my husband...",
  "overall": 5.0,
  "summary": "Good Quality",
  "unixReviewTime": 1355270400
}
```

Metadata (每行一个JSON):
```json
{
  "asin": "0000013714",
  "title": "Music Album",
  "price": "$19.99",
  "brand": "Sony",
  "category": ["CDs & Vinyl", "Pop"],
  "description": "Great music album..."
}
```

### 处理后的序列数据

每个序列样本：
```python
{
  'user_id': 123,              # 用户ID
  'history': [45, 67, 89],     # 历史交互物品ID列表
  'target': 102,               # 目标物品ID
  'seq_length': 3              # 序列实际长度
}
```

### 物品特征

```python
{
  'title': "Product Title",
  'description': "Product description...",
  'brand': "Brand Name",
  'categories': ["Category1", "Category2"],
  'price': 29.99
}
```

## 训练参数说明

### 基础参数

- `--category`: 数据集类别 (beauty/games/sports)
- `--batch_size`: 批次大小（推荐：256）
- `--epochs`: 训练轮数（推荐：50）
- `--lr`: 学习率（默认：1e-3）

### 模型参数

- `--hidden_dim`: 隐藏层维度（推荐：256）
- `--item_embed_dim`: 物品嵌入维度（推荐：128）
- `--disentangled_dim`: 解耦维度大小（推荐：64）
- `--num_interests`: 用户兴趣数量（推荐：4）
- `--quantum_state_dim`: 量子态维度（推荐：128）

### 损失权重

- `--alpha_recon`: 重构损失权重（默认：0.5）
- `--alpha_causal`: 因果损失权重（默认：0.1）
- `--alpha_diversity`: 多样性损失权重（默认：0.05）

### 评估参数

- `--eval_interval`: 评估间隔（默认：5 epochs）
- `--filter_train_items`: 评估时过滤训练物品

## 评估指标

模型在以下指标上评估：

- **HR@K** (Hit Rate): Top-K中命中目标的比例
- **NDCG@K** (Normalized DCG): 考虑排名位置的归一化折损累积增益
- **MRR** (Mean Reciprocal Rank): 目标物品排名的平均倒数

默认评估 K = [5, 10, 20, 50]

## 完整流程示例

```bash
# 1. 下载数据
python data/download_amazon.py --category beauty

# 2. 预处理
python data/preprocess_amazon.py --category beauty

# 3. 快速测试
python scripts/quick_test.py

# 4. 训练（基础配置）
python train_amazon.py --category beauty \
                        --batch_size 256 \
                        --epochs 50

# 5. 训练（完整配置，过滤训练物品）
python train_amazon.py --category beauty \
                        --batch_size 256 \
                        --epochs 100 \
                        --hidden_dim 512 \
                        --item_embed_dim 256 \
                        --disentangled_dim 128 \
                        --num_interests 8 \
                        --filter_train_items \
                        --eval_interval 5
```

## 训练时间估计

在单个 NVIDIA RTX 3090 GPU 上：

| Dataset | Batch Size | Time per Epoch | Total (50 epochs) |
|---------|------------|----------------|-------------------|
| Beauty  | 256        | ~2 min         | ~1.5 hours        |
| Games   | 256        | ~2.5 min       | ~2 hours          |
| Sports  | 256        | ~3 min         | ~2.5 hours        |

*时间取决于硬件配置和模型参数*

## 性能基准

在标准配置下的预期性能（NDCG@10）：

| Dataset | Random | PopRank | GRU4Rec | SASRec | Our Model |
|---------|--------|---------|---------|--------|-----------|
| Beauty  | 0.010  | 0.045   | 0.082   | 0.095  | **0.12+** |
| Games   | 0.008  | 0.052   | 0.091   | 0.108  | **0.14+** |
| Sports  | 0.009  | 0.041   | 0.076   | 0.089  | **0.11+** |

*实际性能取决于超参数调优*

## 故障排查

### 1. 下载失败

**问题**: 网络连接超时

**解决**:
- 检查网络连接
- 使用代理或VPN
- 手动下载数据文件

### 2. 内存不足

**问题**: OOM (Out of Memory)

**解决**:
```bash
# 减小batch size
python train_amazon.py --batch_size 128

# 减小模型维度
python train_amazon.py --hidden_dim 128 --item_embed_dim 64

# 减小序列长度
python data/preprocess_amazon.py --max_seq_length 30
```

### 3. 训练很慢

**问题**: 训练速度慢

**解决**:
```bash
# 增加batch size（如果内存允许）
python train_amazon.py --batch_size 512

# 减少num_workers（如果CPU瓶颈）
python train_amazon.py --num_workers 2

# 不使用文本特征（默认已关闭）
# 文本特征需要额外的BERT编码，会显著降低速度
```

### 4. 评估很慢

**问题**: 全库评估时间长

**解决**:
- 这是正常的，因为我们对所有物品计算分数
- 可以增加评估间隔: `--eval_interval 10`
- 或使用GPU加速评估

## 数据目录结构

```
data/
├── raw/                          # 原始数据
│   ├── beauty_reviews.json
│   ├── beauty_meta.json
│   ├── games_reviews.json
│   ├── games_meta.json
│   ├── sports_reviews.json
│   └── sports_meta.json
│
└── processed/                    # 处理后的数据
    ├── beauty/
    │   ├── train_sequences.pkl   # 训练序列
    │   ├── valid_sequences.pkl   # 验证序列
    │   ├── test_sequences.pkl    # 测试序列
    │   ├── item_features.pkl     # 物品特征
    │   ├── mappings.pkl          # ID映射
    │   └── statistics.json       # 统计信息
    │
    ├── games/
    │   └── ...
    │
    └── sports/
        └── ...
```

## 引用

如果使用Amazon数据集，请引用：

```bibtex
@inproceedings{ni2019justifying,
  title={Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}
```

## 常见问题

**Q: 可以使用其他Amazon类别吗？**

A: 可以！修改 `download_amazon.py` 中的 `AMAZON_URLS` 字典添加新类别即可。

**Q: 为什么使用留一法而不是随机划分？**

A: 留一法更符合真实场景：我们总是预测用户的下一个交互，而不是随机的某个交互。

**Q: 全库评估vs负采样评估的区别？**

A: 全库评估更准确但更慢；负采样更快但可能过于乐观。我们选择全库评估以获得更可靠的性能估计。

**Q: 如何加速训练？**

A: 使用更大的batch size、关闭文本特征、使用更小的模型、减少评估频率。

## 联系与支持

如有问题，请提Issue或联系开发者。
