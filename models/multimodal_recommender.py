"""
Multimodal Sequential Recommender
整合解耦表征学习、因果推断和量子启发编码器的完整推荐系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .disentangled_representation import DisentangledRepresentation
from .causal_inference import CausalInferenceModule
from .quantum_inspired_encoder import QuantumInspiredMultiInterestEncoder


class SequenceEncoder(nn.Module):
    """序列编码器，用于处理用户历史行为序列"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 使用GRU或Transformer编码序列
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Linear(output_dim, input_dim)

    def forward(self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence: (batch, seq_len, input_dim)
            lengths: (batch,) 每个序列的实际长度

        Returns:
            编码后的序列 (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = sequence.size()
        
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.gru(packed)
            # ⭐ 修复：指定total_length以确保输出维度正确
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len
            )
        else:
            output, _ = self.gru(sequence)

        output = self.output_projection(output)
        return output


class MultimodalRecommender(nn.Module):
    """
    多模态序列推荐系统
    整合三个核心组件：
    1. 解耦表征学习 (Disentangled Representation)
    2. 因果推断模块 (Causal Inference)
    3. 量子启发多兴趣编码器 (Quantum-Inspired Multi-Interest Encoder)
    """

    def __init__(
        self,
        # 多模态输入配置
        modality_dims: Dict[str, int] = None,
        # 解耦表征配置
        disentangled_dim: int = 128,
        num_disentangled_dims: int = 3,
        # 量子编码器配置
        num_interests: int = 4,
        quantum_state_dim: int = 256,
        # 通用配置
        hidden_dim: int = 512,
        item_embed_dim: int = 256,
        num_items: int = 10000,
        # 序列配置
        max_seq_length: int = 50,
        # 损失权重 (v0.8.1 - 平衡方案：rec主导但辅助损失仍有意义)
        alpha_recon: float = 0.1,       # 从0.5→0.1 (降低5倍)
        alpha_causal: float = 0.2,      # 从0.5→0.2 (降低2.5倍)
        alpha_diversity: float = 0.05,  # 保持不变
        alpha_orthogonality: float = 0.05,  # 从0.1→0.05 (降低2倍)
        # 其他
        use_quantum_computing: bool = False,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: 各模态的输入维度，例如 {'text': 768, 'image': 2048}
            disentangled_dim: 每个解耦维度的大小
            num_disentangled_dims: 解耦维度数量（功能、美学、情感）
            num_interests: 用户兴趣数量
            quantum_state_dim: 量子态维度
            hidden_dim: 隐藏层维度
            item_embed_dim: 物品嵌入维度
            num_items: 物品总数
            max_seq_length: 最大序列长度
            alpha_recon: 重构损失权重
            alpha_causal: 因果损失权重
            alpha_diversity: 多样性损失权重
            use_quantum_computing: 是否使用真实量子计算
            dropout: Dropout比率
        """
        super().__init__()

        # 默认模态配置
        if modality_dims is None:
            modality_dims = {
                'text': 768,  # BERT/RoBERTa
                'image': 2048,  # ResNet
                'audio': 128   # Audio features
            }
        
        # ⭐ 添加item embedding作为一个模态
        modality_dims_with_item = modality_dims.copy()
        modality_dims_with_item['item'] = item_embed_dim  # item embedding也是一个模态

        self.modality_dims = modality_dims
        self.disentangled_dim = disentangled_dim
        self.num_disentangled_dims = num_disentangled_dims
        self.num_interests = num_interests
        self.quantum_state_dim = quantum_state_dim
        self.hidden_dim = hidden_dim
        self.item_embed_dim = item_embed_dim
        self.num_items = num_items
        self.max_seq_length = max_seq_length

        self.alpha_recon = alpha_recon
        self.alpha_causal = alpha_causal
        self.alpha_diversity = alpha_diversity
        self.alpha_orthogonality = alpha_orthogonality

        # ==================== 核心组件 ====================

        # 1. 物品嵌入（+1 因为ID从1开始，0用于padding）
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_dim, padding_idx=0)

        # 2. 解耦表征学习模块
        self.disentangled_module = DisentangledRepresentation(
            input_dims=modality_dims_with_item,  # ⭐ 包含item embedding
            hidden_dim=hidden_dim,
            shared_dim=item_embed_dim,
            disentangled_dim=disentangled_dim,
            beta=0.5,  # ⭐ v0.7.0: 降低从1.0
            gamma=1.0   # ⭐⭐⭐ v0.7.0: 大幅降低从10.0
        )

        # 3. 序列编码器
        total_disentangled_dim = disentangled_dim * num_disentangled_dims
        self.sequence_encoder = SequenceEncoder(
            input_dim=total_disentangled_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            bidirectional=True
        )

        # 4. 量子启发多兴趣编码器
        self.quantum_encoder = QuantumInspiredMultiInterestEncoder(
            input_dim=total_disentangled_dim,
            state_dim=quantum_state_dim,
            num_interests=num_interests,
            hidden_dim=hidden_dim,
            output_dim=item_embed_dim,
            use_quantum_computing=use_quantum_computing
        )

        # 5. 因果推断模块
        self.causal_module = CausalInferenceModule(
            disentangled_dim=disentangled_dim,
            num_dimensions=num_disentangled_dims,
            hidden_dim=hidden_dim,
            num_ensembles=5
        )

        # ==================== 预测头 ====================

        # ⭐⭐⭐ P0核心修复: 权重共享点积打分（不再用独立分类器）
        # 推荐打分 = user_repr @ item_embedding.weight.T + item_bias
        # 这样rec监督直接作用到item_embedding，加速收敛、提升效果！
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))  # 每个物品的偏置
        
        # ⭐ 温度参数：固定值（不可学习），避免训练中变成极端值导致NaN
        self.register_buffer('temperature', torch.tensor(0.2))  # 固定为0.2
        
        # ⭐ KL退火因子（训练时动态调整）
        self.kl_anneal_factor = 1.0  # 默认1.0，在训练初期会设为0

        # 维度重要性预测（用于可解释性）
        self.dimension_importance_head = nn.Sequential(
            nn.Linear(total_disentangled_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_disentangled_dims)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 正态分布初始化embedding
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    # padding位置初始化为0
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm初始化
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _compute_interest_orthogonality_loss(
        self,
        interests_real: torch.Tensor,
        interests_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        计算兴趣正交性损失
        确保不同兴趣向量尽可能正交（相互独立）
        
        Args:
            interests_real: 兴趣向量的实部 (batch, num_interests, dim)
            interests_imag: 兴趣向量的虚部 (batch, num_interests, dim)
        
        Returns:
            正交性损失标量
        """
        batch_size, num_interests, dim = interests_real.shape
        
        # 计算复数兴趣向量的Gram矩阵
        # 对于复数向量 z_i 和 z_j，内积定义为：<z_i, z_j*> = Σ(a_i*a_j + b_i*b_j) + i*Σ(b_i*a_j - a_i*b_j)
        
        # 实部的内积贡献
        real_gram = torch.bmm(interests_real, interests_real.transpose(1, 2))  # (batch, n, n)
        real_gram += torch.bmm(interests_imag, interests_imag.transpose(1, 2))
        
        # 理想情况下，Gram矩阵应该是单位矩阵（对角线为1，其他为0）
        # 我们只惩罚非对角线元素
        identity = torch.eye(num_interests, device=interests_real.device).unsqueeze(0)  # (1, n, n)
        
        # 归一化：将每个兴趣向量归一化，使对角线接近1
        interests_norm = torch.sqrt(
            (interests_real ** 2).sum(dim=-1, keepdim=True) + 
            (interests_imag ** 2).sum(dim=-1, keepdim=True) + 1e-8
        )
        interests_real_norm = interests_real / interests_norm
        interests_imag_norm = interests_imag / interests_norm
        
        # 重新计算归一化后的Gram矩阵
        real_gram_norm = torch.bmm(interests_real_norm, interests_real_norm.transpose(1, 2))
        real_gram_norm += torch.bmm(interests_imag_norm, interests_imag_norm.transpose(1, 2))
        
        # 非对角线元素的平方和（越小越好）
        mask = 1 - identity  # 非对角线掩码
        off_diagonal = (real_gram_norm * mask).pow(2).sum(dim=[1, 2])  # (batch,)
        
        return off_diagonal.mean()

    def forward(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None,
        target_items: Optional[torch.Tensor] = None,
        candidate_items: Optional[torch.Tensor] = None,  # ⭐ 新增：候选物品（负采样）
        labels: Optional[torch.Tensor] = None,  # ⭐ 新增：标签（0/1）
        return_loss: bool = True,
        return_explanations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            item_ids: 物品ID序列 (batch, seq_len)
            multimodal_features: 多模态特征字典，每个值的形状为 (batch, seq_len, feature_dim)
            seq_lengths: 序列实际长度 (batch,)
            target_items: 目标物品ID (batch,)，用于训练（全库模式）
            candidate_items: 候选物品 (batch, num_candidates)，用于训练（采样模式）⭐
            labels: 候选物品的标签 (batch, num_candidates)，0/1 ⭐
            return_loss: 是否计算损失
            return_explanations: 是否返回可解释性信息

        Returns:
            包含预测、损失和其他信息的字典
        """
        batch_size, seq_len = item_ids.size()

        # ==================== 1. 物品嵌入 + 解耦表征学习 ====================
        
        # ⭐ 关键修复：使用item_embedding！
        item_embeddings = self.item_embedding(item_ids)  # (batch, seq_len, item_embed_dim)

        # 处理每个时间步的多模态特征
        all_disentangled_features = []
        disentangled_losses = []

        for t in range(seq_len):
            # ⭐ 结合item embedding和multimodal features
            multimodal_t = {
                modality: features[:, t, :]
                for modality, features in multimodal_features.items()
            }
            # 将item embedding作为一个模态加入
            multimodal_t['item'] = item_embeddings[:, t, :]

            # 应用解耦表征学习
            disentangled_output = self.disentangled_module(
                multimodal_t,
                return_loss=return_loss
            )

            # 拼接解耦特征
            z_concat = disentangled_output['z_concat']
            all_disentangled_features.append(z_concat)

            if return_loss:
                disentangled_losses.append(disentangled_output['loss'])

        # Stack所有时间步的解耦特征
        disentangled_sequence = torch.stack(all_disentangled_features, dim=1)  # (batch, seq_len, total_dim)

        # ==================== 2. 序列编码 ====================

        # 使用序列编码器捕获时序依赖
        encoded_sequence = self.sequence_encoder(disentangled_sequence, seq_lengths)  # (batch, seq_len, total_dim)

        # 取最后一个时间步作为用户表示
        if seq_lengths is not None:
            # 获取每个序列的最后一个有效位置
            last_indices = (seq_lengths - 1).long()
            user_representation = encoded_sequence[
                torch.arange(batch_size, device=encoded_sequence.device),
                last_indices
            ]
        else:
            user_representation = encoded_sequence[:, -1, :]  # (batch, total_dim)

        # ==================== 3. 量子启发多兴趣编码 ====================

        # 使用量子编码器建模用户的多样化兴趣
        quantum_output = self.quantum_encoder(
            user_representation,
            return_all_interests=True
        )

        user_interest_representation = quantum_output['output']  # (batch, item_embed_dim)

        # ==================== 4. 因果推断 ====================

        # ⭐ 渐进式训练：根据alpha_causal决定是否调用因果模块
        # Phase 1时alpha_causal=0.0，跳过因果推断以加速收敛
        if self.alpha_causal > 0:
            # 准备解耦特征字典（使用最后一个时间步）
            last_disentangled_features = {
                'function': disentangled_sequence[:, -1, :self.disentangled_dim],
                'aesthetics': disentangled_sequence[:, -1, self.disentangled_dim:2*self.disentangled_dim],
                'emotion': disentangled_sequence[:, -1, 2*self.disentangled_dim:]
            }

            # 应用因果推断
            causal_output = self.causal_module(
                last_disentangled_features,
                return_uncertainty=True,
                num_mc_samples=10
            )
        else:
            # Phase 1: 跳过因果推断，返回空字典
            causal_output = {
                'counterfactuals': {},
                'causal_effects': {},
                'uncertainty': torch.zeros(batch_size, device=item_ids.device),
                'original_features': disentangled_sequence[:, -1, :]
            }

        # ==================== 5. 推荐预测 ====================

        # ⭐⭐⭐ 关键修复: L2归一化 + 温度缩放，防止范数失衡
        # 用户表征归一化（添加epsilon防止零向量导致NaN）
        user_repr_norm = F.normalize(user_interest_representation + 1e-8, p=2, dim=-1)  # (batch, embed_dim)
        
        # 物品embedding归一化（添加epsilon防止零向量导致NaN）
        item_emb_norm = F.normalize(self.item_embedding.weight + 1e-8, p=2, dim=-1)  # (num_items+1, embed_dim)
        
        # 使用固定的温度参数（已在__init__中定义为buffer）
        temperature = self.temperature  # 固定值0.2，不再裁剪
        
        # ⭐⭐⭐ P0核心修复: 统一用点积打分（权重共享）
        if candidate_items is not None:
            # 采样模式：只对候选物品打分（训练时可选）
            candidate_embeddings = self.item_embedding(candidate_items)  # (batch, num_candidates, embed_dim)
            candidate_emb_norm = F.normalize(candidate_embeddings, p=2, dim=-1)
            user_repr_expanded = user_repr_norm.unsqueeze(1)  # (batch, 1, embed_dim)
            recommendation_logits = (user_repr_expanded * candidate_emb_norm).sum(dim=-1) / temperature  # (batch, num_candidates)
            # 添加候选物品的bias
            candidate_bias = self.item_bias[candidate_items]  # (batch, num_candidates)
            recommendation_logits = recommendation_logits + candidate_bias
        else:
            # 全库模式：对所有物品打分（评估/训练默认）
            # ⭐ 归一化点积打分: logits = (user_norm @ item_norm.T) / temperature + bias
            recommendation_logits = torch.matmul(
                user_repr_norm,  # (batch, embed_dim)
                item_emb_norm.T   # (num_items+1, embed_dim).T
            ) / temperature + self.item_bias  # (batch, num_items+1)
            
            # ⭐⭐⭐ P0关键: 屏蔽padding (item_id=0)，防止污染Top-K
            recommendation_logits[:, 0] = -1e9

        # 预测维度重要性（用于可解释性）
        dimension_importance = self.dimension_importance_head(user_representation)
        dimension_importance = F.softmax(dimension_importance, dim=-1)

        # ==================== 结果汇总 ====================

        results = {
            'recommendation_logits': recommendation_logits,
            'user_representation': user_interest_representation,
            'dimension_importance': dimension_importance,
            'quantum_output': quantum_output,
            'causal_output': causal_output,
            'disentangled_sequence': disentangled_sequence
        }

        # ==================== 6. 损失计算 ====================

        if return_loss:
            # ⭐⭐⭐ 使用BPR损失（更适合隐式反馈Top-K推荐）
            if target_items is not None:
                # 计算正样本得分
                batch_indices = torch.arange(batch_size, device=target_items.device)
                pos_scores = recommendation_logits[batch_indices, target_items]  # (batch,)
                
                # BPR损失：高效采样负样本
                num_negatives = 4  # 每个正样本采样4个负样本
                
                # ⭐ 高效负采样（向量化操作，避免循环）
                # 采样更多候选，然后过滤（容忍小概率碰撞）
                neg_items = torch.randint(
                    1, self.num_items + 1,  # 从1到num_items（排除padding=0）
                    (batch_size, num_negatives),
                    device=target_items.device
                )
                
                # 简单策略：如果碰到正样本就+1（循环避免）
                # 这种情况极少（1/12101），对训练影响可忽略
                target_expanded = target_items.unsqueeze(1)  # (batch, 1)
                collision_mask = (neg_items == target_expanded)
                neg_items = torch.where(
                    collision_mask,
                    (neg_items % self.num_items) + 1,  # 碰撞时换一个
                    neg_items
                )
                
                # 计算负样本得分
                neg_scores = recommendation_logits[batch_indices.unsqueeze(1), neg_items]  # (batch, num_negatives)
                
                # ⭐ BPR损失: -log(sigmoid(pos_score - neg_score))
                # 展开为: log(1 + exp(neg_score - pos_score))
                pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch, 1)
                bpr_loss = -F.logsigmoid(pos_scores_expanded - neg_scores).mean()  # 鼓励pos > neg
                
                recommendation_loss = bpr_loss
            else:
                recommendation_loss = torch.tensor(0.0, device=item_ids.device)

            # 解耦表征损失
            avg_disentangled_loss = torch.stack(disentangled_losses).mean() if disentangled_losses else 0.0

            # 多样性损失（鼓励兴趣多样性）
            diversity_loss = 0.0
            orthogonality_loss = 0.0
            
            if 'individual_interests_real' in quantum_output:
                interests_real = quantum_output['individual_interests_real']
                interests_imag = quantum_output['individual_interests_imag']
                
                # 多样性损失：鼓励不同兴趣之间的差异
                # diversity_score 范围 [0,1]，越大越多样
                # 我们希望最大化diversity，即最小化 (1 - diversity)
                diversity_score = self.quantum_encoder.get_interest_diversity(
                    interests_real, interests_imag
                )
                diversity_loss = (1.0 - diversity_score).mean()  # 鼓励diversity接近1
                
                # 正交性损失：确保不同兴趣向量相互正交（独立）
                orthogonality_loss = self._compute_interest_orthogonality_loss(
                    interests_real, interests_imag
                )

            # 因果损失 - 极简版 v0.7.0 ⭐⭐⭐
            # 核心原则: 简单有效，避免过度复杂
            causal_loss = 0.0
            
            if 'counterfactuals' in causal_output and len(causal_output['counterfactuals']) > 0:
                # 收集所有反事实预测
                all_cf_logits = []
                
                for cf_key, cf_features in causal_output['counterfactuals'].items():
                    if cf_key == 'original' or cf_features is None:
                        continue
                    
                    if isinstance(cf_features, torch.Tensor) and cf_features.size(0) > 0:
                        # 获取原始特征并重构反事实特征
                        original_features = causal_output['original_features']
                        
                        # 确定哪个维度被干预
                        if 'function' in cf_key:
                            dim_idx = 0
                        elif 'aesthetics' in cf_key:
                            dim_idx = 1
                        elif 'emotion' in cf_key:
                            dim_idx = 2
                        else:
                            continue
                        
                        # 重构完整特征
                        cf_full_features = original_features.clone()
                        start_idx = dim_idx * self.disentangled_dim
                        end_idx = start_idx + self.disentangled_dim
                        cf_full_features[:, start_idx:end_idx] = cf_features
                        
                        # 生成反事实预测
                        cf_quantum_output = self.quantum_encoder(
                            cf_full_features.unsqueeze(1),
                            return_all_interests=False
                        )
                        cf_user_representation = cf_quantum_output['output']
                        
                        # ⭐⭐⭐ P0: 反事实预测也用归一化点积打分（一致性）
                        # 添加epsilon防止零向量导致NaN
                        cf_user_norm = F.normalize(cf_user_representation + 1e-8, p=2, dim=-1)
                        cf_logits = torch.matmul(
                            cf_user_norm,
                            item_emb_norm.T  # 使用前面计算的归一化embedding
                        ) / temperature + self.item_bias
                        cf_logits[:, 0] = -1e9  # 屏蔽padding
                        
                        all_cf_logits.append(cf_logits)
                
                if len(all_cf_logits) > 0:
                    # ⭐ 核心损失1: 预测差异（简单直接，让模型自然学习）
                    pred_diff_loss = 0.0
                    
                    # ⭐⭐⭐ P0: 用归一化点积计算原始logits做比较
                    if labels is not None and target_items is not None:
                        # 混合模式：重新计算全库logits用于因果比较
                        original_all_logits = torch.matmul(
                            user_repr_norm,  # 使用归一化的用户表征
                            item_emb_norm.T
                        ) / temperature + self.item_bias
                        original_all_logits[:, 0] = -1e9
                        for cf_logits in all_cf_logits:
                            pred_diff = (cf_logits - original_all_logits).abs().mean()
                            pred_diff_loss += pred_diff
                    else:
                        # 纯模式：直接比较
                        for cf_logits in all_cf_logits:
                            pred_diff = (cf_logits - recommendation_logits).abs().mean()
                            pred_diff_loss += pred_diff
                    
                    causal_loss += pred_diff_loss / len(all_cf_logits)
                    
                    # ⭐ 核心损失2: ITE幅度（因果效应强度）
                    if 'causal_effects' in causal_output:
                        effects = causal_output['causal_effects']
                        if 'ite' in effects:
                            ite = effects['ite']  # (batch, num_treatments)
                            # 鼓励ITE有合理的幅度
                            ite_magnitude = ite.abs().mean()
                            target_ite = torch.tensor(0.4, device=ite.device)
                            ite_loss = F.smooth_l1_loss(ite_magnitude, target_ite)
                            causal_loss += ite_loss

            # ⭐ 总损失 - 应用KL退火
            # KL退火：前20个epoch逐渐增加KL权重，避免后验坍塌
            total_loss = (
                recommendation_loss +
                self.alpha_recon * self.kl_anneal_factor * avg_disentangled_loss +
                self.alpha_diversity * diversity_loss +
                self.alpha_orthogonality * orthogonality_loss +
                self.alpha_causal * causal_loss
            )

            results.update({
                'loss': total_loss,
                'recommendation_loss': recommendation_loss,
                'disentangled_loss': avg_disentangled_loss,
                'diversity_loss': diversity_loss,
                'orthogonality_loss': orthogonality_loss,
                'causal_loss': causal_loss
            })

        # ==================== 7. 可解释性信息 ====================

        if return_explanations:
            explanations = {
                'dimension_importance': dimension_importance,
                'uncertainty': causal_output['uncertainty'],
                'interference_strength': quantum_output['interference_strength'],
            }

            if 'dimension_importance' in causal_output:
                explanations['causal_dimension_importance'] = causal_output['dimension_importance']

            results['explanations'] = explanations

        return results

    def predict(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测Top-K推荐

        Args:
            item_ids: 物品ID序列
            multimodal_features: 多模态特征
            seq_lengths: 序列长度
            top_k: 返回Top-K结果

        Returns:
            (top_k_items, top_k_scores)
        """
        with torch.no_grad():
            outputs = self.forward(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=False
            )

            logits = outputs['recommendation_logits']
            scores = F.softmax(logits, dim=-1)

            top_k_scores, top_k_items = torch.topk(scores, k=top_k, dim=-1)

            return top_k_items, top_k_scores

    def explain_recommendation(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        为推荐结果提供可解释性分析

        Returns:
            包含各种可解释性指标的字典
        """
        with torch.no_grad():
            outputs = self.forward(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=True
            )

            explanations = outputs['explanations']

            # 格式化输出
            dimension_names = ['Function (功能)', 'Aesthetics (美学)', 'Emotion (情感)']

            explanation_text = {
                'dimension_importance': {
                    dimension_names[i]: float(explanations['dimension_importance'][0, i])
                    for i in range(self.num_disentangled_dims)
                },
                'uncertainty': {
                    'aleatoric': float(explanations['uncertainty']['refined_aleatoric'].mean()),
                    'epistemic': float(explanations['uncertainty']['refined_epistemic'].mean()),
                    'confidence': float(explanations['uncertainty']['confidence'].mean())
                },
                'interest_diversity': 'High' if outputs['quantum_output']['interference_strength'].var() > 0.1 else 'Low'
            }

            if 'causal_dimension_importance' in explanations:
                explanation_text['causal_importance'] = {
                    dimension_names[i]: float(explanations['causal_dimension_importance'][i])
                    for i in range(self.num_disentangled_dims)
                }

            return explanation_text

    def get_user_interests(
        self,
        item_ids: torch.Tensor,
        multimodal_features: Dict[str, torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取用户的多个兴趣表示

        Returns:
            包含各个兴趣向量的字典
        """
        with torch.no_grad():
            outputs = self.forward(
                item_ids,
                multimodal_features,
                seq_lengths,
                target_items=None,
                return_loss=False,
                return_explanations=False
            )

            quantum_output = outputs['quantum_output']

            # 返回各个兴趣的复数表示
            interests = {
                f'interest_{i}': {
                    'real': quantum_output['individual_interests_real'][:, i, :],
                    'imag': quantum_output['individual_interests_imag'][:, i, :],
                }
                for i in range(self.num_interests)
            }

            # 也返回叠加后的状态
            interests['superposed'] = {
                'real': quantum_output['superposed_state_real'],
                'imag': quantum_output['superposed_state_imag']
            }

            return interests
