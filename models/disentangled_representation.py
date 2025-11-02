"""
Disentangled Representation Learning Module
解耦表征学习：将多模态特征解耦为功能、美学、情感三个独立维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class MultimodalEncoder(nn.Module):
    """多模态特征编码器"""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int, output_dim: int):
        """
        Args:
            input_dims: 每个模态的输入维度，例如 {'text': 768, 'image': 2048, 'audio': 128}
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()
        self.modalities = list(input_dims.keys())

        # 为每个模态创建独立的编码器
        self.encoders = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
            for modality, dim in input_dims.items()
        })

        # 注意力融合
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, multimodal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multimodal_features: 字典，包含各模态特征 {modality: tensor of shape (batch, feature_dim)}

        Returns:
            融合后的特征 (batch, output_dim)
        """
        encoded_features = []

        for modality in self.modalities:
            if modality in multimodal_features:
                encoded = self.encoders[modality](multimodal_features[modality])
                encoded_features.append(encoded)

        if not encoded_features:
            raise ValueError("No valid modality features provided")

        # Stack features for attention
        stacked_features = torch.stack(encoded_features, dim=1)  # (batch, num_modalities, output_dim)

        # Self-attention fusion
        fused_features, _ = self.attention_fusion(
            stacked_features, stacked_features, stacked_features
        )

        # Mean pooling over modalities
        fused_features = fused_features.mean(dim=1)  # (batch, output_dim)
        fused_features = self.layer_norm(fused_features)

        return fused_features


class DisentangledHead(nn.Module):
    """解耦维度的独立头部"""

    def __init__(self, input_dim: int, disentangled_dim: int, dimension_name: str):
        """
        Args:
            input_dim: 输入特征维度
            disentangled_dim: 解耦维度大小
            dimension_name: 维度名称（'function', 'aesthetics', 'emotion'）
        """
        super().__init__()
        self.dimension_name = dimension_name

        # 使用多层网络提取特定维度的特征
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, disentangled_dim)
        )

        # 参数化的高斯分布（用于变分推断）
        self.mu_head = nn.Linear(disentangled_dim, disentangled_dim)
        self.logvar_head = nn.Linear(disentangled_dim, disentangled_dim)
        
        # ⭐ 初始化：确保初始logvar接近0（std接近1）
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 输入特征 (batch, input_dim)

        Returns:
            z: 采样的解耦特征 (batch, disentangled_dim)
            mu: 均值 (batch, disentangled_dim)
            logvar: 对数方差 (batch, disentangled_dim)
        """
        h = self.projector(features)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        # ⭐⭐⭐ 关键修复: 约束logvar防止方差爆炸
        logvar = torch.clamp(logvar, min=-10, max=2)  # std在[0.007, 2.7]范围

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class DisentangledRepresentation(nn.Module):
    """
    解耦表征学习模块
    将多模态特征解耦为功能（Function）、美学（Aesthetics）、情感（Emotion）三个独立维度
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 512,
        shared_dim: int = 256,
        disentangled_dim: int = 128,
        beta: float = 0.5,
        gamma: float = 1.0
    ):
        """
        Args:
            input_dims: 每个模态的输入维度
            hidden_dim: 编码器隐藏层维度
            shared_dim: 共享特征维度
            disentangled_dim: 每个解耦维度的大小
            beta: KL散度权重（β-VAE）⭐ 降低从1.0到0.5
            gamma: 总相关性惩罚权重 ⭐⭐⭐ 大幅降低从10.0到1.0
        """
        super().__init__()

        self.shared_dim = shared_dim
        self.disentangled_dim = disentangled_dim
        self.beta = beta
        self.gamma = gamma

        # 多模态编码器
        self.multimodal_encoder = MultimodalEncoder(input_dims, hidden_dim, shared_dim)

        # 三个解耦维度的独立头部
        self.function_head = DisentangledHead(shared_dim, disentangled_dim, 'function')
        self.aesthetics_head = DisentangledHead(shared_dim, disentangled_dim, 'aesthetics')
        self.emotion_head = DisentangledHead(shared_dim, disentangled_dim, 'emotion')

        # 重构解码器
        self.decoder = nn.Sequential(
            nn.Linear(disentangled_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, shared_dim)
        )

        # 用于计算总相关性的判别器
        self.discriminator = nn.Sequential(
            nn.Linear(disentangled_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        multimodal_features: Dict[str, torch.Tensor],
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            multimodal_features: 多模态特征字典
            return_loss: 是否返回损失

        Returns:
            包含解耦特征、重构特征和损失的字典
        """
        batch_size = next(iter(multimodal_features.values())).size(0)

        # 1. 编码多模态特征
        shared_features = self.multimodal_encoder(multimodal_features)

        # 2. 解耦为三个独立维度
        z_function, mu_function, logvar_function = self.function_head(shared_features)
        z_aesthetics, mu_aesthetics, logvar_aesthetics = self.aesthetics_head(shared_features)
        z_emotion, mu_emotion, logvar_emotion = self.emotion_head(shared_features)

        # 3. 拼接所有解耦特征
        z_concat = torch.cat([z_function, z_aesthetics, z_emotion], dim=-1)

        # 4. 重构
        reconstructed = self.decoder(z_concat)

        # 5. 计算损失
        results = {
            'z_function': z_function,
            'z_aesthetics': z_aesthetics,
            'z_emotion': z_emotion,
            'z_concat': z_concat,
            'reconstructed': reconstructed,
            'shared_features': shared_features
        }

        if return_loss:
            # 重构损失
            recon_loss = F.mse_loss(reconstructed, shared_features)

            # KL散度损失（β-VAE）
            kl_function = self._kl_divergence(mu_function, logvar_function)
            kl_aesthetics = self._kl_divergence(mu_aesthetics, logvar_aesthetics)
            kl_emotion = self._kl_divergence(mu_emotion, logvar_emotion)
            kl_loss = kl_function + kl_aesthetics + kl_emotion

            # 总相关性惩罚（TC - Total Correlation）
            # 确保维度之间相互独立
            tc_loss = self._total_correlation_loss(z_concat, batch_size)

            # 维度独立性损失（通过协方差矩阵）
            independence_loss = self._independence_loss(z_function, z_aesthetics, z_emotion)

            # ⭐ 修复：为每个损失项添加权重平衡和clip，防止某项爆炸
            # 降低内部权重，让解耦损失更温和
            total_loss = (
                recon_loss +
                self.beta * torch.clamp(kl_loss, 0, 10.0) +  # ⭐ clip KL loss
                self.gamma * torch.clamp(tc_loss, 0, 10.0) +  # ⭐ clip TC loss
                torch.clamp(independence_loss, 0, 1.0)  # ⭐ clip independence loss
            )

            results.update({
                'loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'tc_loss': tc_loss,
                'independence_loss': independence_loss
            })

        return results

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """计算KL散度 KL(q(z|x) || p(z))，其中 p(z) = N(0, I)"""
        # ⭐ 修复: 正确归一化 - 除以batch_size和特征维度
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl / (mu.size(0) * mu.size(1))

    def _total_correlation_loss(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        计算总相关性损失，确保不同维度之间相互独立
        使用对抗训练方法
        """
        # 真实样本
        real_scores = self.discriminator(z)

        # 创建置换样本（permuted samples）来破坏维度间的依赖关系
        # 对每个维度独立进行随机置换
        z_split = torch.split(z, self.disentangled_dim, dim=-1)
        z_permuted = []
        for z_dim in z_split:
            perm_idx = torch.randperm(batch_size, device=z.device)
            z_permuted.append(z_dim[perm_idx])
        z_permuted = torch.cat(z_permuted, dim=-1)

        # 置换样本
        fake_scores = self.discriminator(z_permuted)

        # 对抗损失：真实样本得分高，置换样本得分低
        tc_loss = F.binary_cross_entropy_with_logits(
            real_scores,
            torch.ones_like(real_scores)
        ) + F.binary_cross_entropy_with_logits(
            fake_scores,
            torch.zeros_like(fake_scores)
        )

        return tc_loss

    def _independence_loss(
        self,
        z_function: torch.Tensor,
        z_aesthetics: torch.Tensor,
        z_emotion: torch.Tensor
    ) -> torch.Tensor:
        """
        计算维度间的独立性损失
        通过最小化不同维度间的协方差来确保独立性
        """
        # 标准化特征
        z_function = F.normalize(z_function, dim=-1)
        z_aesthetics = F.normalize(z_aesthetics, dim=-1)
        z_emotion = F.normalize(z_emotion, dim=-1)

        # 计算不同维度间的相关性（应该接近0）
        corr_func_aesth = torch.abs(torch.sum(z_function * z_aesthetics, dim=-1)).mean()
        corr_func_emot = torch.abs(torch.sum(z_function * z_emotion, dim=-1)).mean()
        corr_aesth_emot = torch.abs(torch.sum(z_aesthetics * z_emotion, dim=-1)).mean()

        independence_loss = corr_func_aesth + corr_func_emot + corr_aesth_emot

        return independence_loss

    def get_disentangled_features(
        self,
        multimodal_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        获取解耦后的特征（用于推理）

        Returns:
            包含三个解耦维度特征的字典
        """
        with torch.no_grad():
            results = self.forward(multimodal_features, return_loss=False)
            return {
                'function': results['z_function'],
                'aesthetics': results['z_aesthetics'],
                'emotion': results['z_emotion'],
                'concat': results['z_concat']
            }
