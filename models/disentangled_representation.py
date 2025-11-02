"""
Improved Disentangled Representation Learning Module
改进的解耦表征学习：先让每个模态独立解耦，再在维度内进行跨模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ModalityDisentangler(nn.Module):
    """
    单模态解耦编码器
    将单个模态的特征解耦为功能、美学、情感三个维度
    """

    def __init__(self, input_dim: int, latent_dim: int = 64):
        """
        Args:
            input_dim: 输入模态的维度
            latent_dim: 每个解耦维度的大小
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

        # 三个解耦头（VAE）
        self.function_head = VAEHead(128, latent_dim, 'function')
        self.aesthetics_head = VAEHead(128, latent_dim, 'aesthetics')
        self.emotion_head = VAEHead(128, latent_dim, 'emotion')

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            x: 输入特征 (batch, input_dim)

        Returns:
            包含三个维度的字典，每个维度包含z, mu, logvar
        """
        shared = self.encoder(x)  # (batch, 128)

        # 解耦为3个维度
        func_out = self.function_head(shared)
        aes_out = self.aesthetics_head(shared)
        emo_out = self.emotion_head(shared)

        return {
            'function': func_out,     # {'z', 'mu', 'logvar', 'epsilon'}
            'aesthetics': aes_out,
            'emotion': emo_out
        }


class VAEHead(nn.Module):
    """VAE头部，用于生成解耦维度"""

    def __init__(self, input_dim: int, latent_dim: int, dimension_name: str):
        super().__init__()

        self.dimension_name = dimension_name
        self.latent_dim = latent_dim

        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, latent_dim)
        )

        # mu和logvar头
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

        # 初始化
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, input_dim)

        Returns:
            字典包含z, mu, logvar, epsilon
        """
        h = self.projector(features)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # 约束logvar防止爆炸
        logvar = torch.clamp(logvar, min=-10, max=2)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'epsilon': epsilon  # ⭐ 保存epsilon用于因果推断
        }


class CrossModalAttention(nn.Module):
    """
    维度内的跨模态注意力融合
    在同一维度内融合来自不同模态的信息
    """

    def __init__(self, num_modalities: int = 3, dim: int = 64):
        super().__init__()

        self.num_modalities = num_modalities
        self.dim = dim

        # 可学习的query（融合目标）
        self.query = nn.Parameter(torch.randn(1, dim))

        # Key和Value投影
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, modality_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_features: (batch, num_modalities, dim)

        Returns:
            fused: (batch, dim)
            attention_weights: (batch, num_modalities)
        """
        batch_size = modality_features.size(0)

        # Q: (batch, 1, dim)
        Q = self.query.expand(batch_size, -1, -1)

        # K, V: (batch, num_modalities, dim)
        K = self.key_proj(modality_features)
        V = self.value_proj(modality_features)

        # Attention scores: (batch, 1, num_modalities)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)  # (batch, 1, num_modalities)

        # Weighted sum: (batch, 1, dim) → (batch, dim)
        fused = torch.bmm(attention_weights, V).squeeze(1)

        return fused, attention_weights.squeeze(1)


class DimensionSpecificMultimodalFusion(nn.Module):
    """
    维度特定的多模态融合
    核心创新：每个模态先独立解耦，然后在同一维度内跨模态融合
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        latent_dim: int = 64,
        beta: float = 0.5,
        gamma: float = 1.0
    ):
        """
        Args:
            input_dims: 每个模态的输入维度，例如 {'text': 768, 'image': 2048, 'item': 256}
            latent_dim: 解耦维度大小
            beta: KL散度权重
            gamma: 总相关性惩罚权重
        """
        super().__init__()

        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.modality_names = list(input_dims.keys())

        # ===每个模态独立的解耦编码器===
        self.modality_disentanglers = nn.ModuleDict({
            modality: ModalityDisentangler(input_dim=dim, latent_dim=latent_dim)
            for modality, dim in input_dims.items()
        })

        # ===维度内跨模态注意力融合===
        self.dimension_fusion = nn.ModuleDict({
            'function': CrossModalAttention(num_modalities=len(input_dims), dim=latent_dim),
            'aesthetics': CrossModalAttention(num_modalities=len(input_dims), dim=latent_dim),
            'emotion': CrossModalAttention(num_modalities=len(input_dims), dim=latent_dim)
        })

        # ===重构解码器（用于VAE损失）===
        # 为每个模态创建重构解码器
        self.decoders = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(latent_dim * 3, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, latent_dim)  # 重构到一个共享空间
            )
            for modality in input_dims.keys()
        })

    def forward(
        self,
        multimodal_features: Dict[str, torch.Tensor],
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            multimodal_features: 多模态特征字典 {modality: (batch, feature_dim)}
            return_loss: 是否计算损失

        Returns:
            包含融合特征和损失的字典
        """
        batch_size = next(iter(multimodal_features.values())).size(0)

        # ===Step 1: 每个模态独立解耦===
        modality_disentangled = {}
        for modality, features in multimodal_features.items():
            if modality in self.modality_disentanglers:
                disentangled = self.modality_disentanglers[modality](features)
                modality_disentangled[modality] = disentangled

        # ===Step 2: 维度内跨模态融合===
        fused = {}
        attention_maps = {}

        for dim_name in ['function', 'aesthetics', 'emotion']:
            # 收集该维度下所有模态的特征
            modality_features_list = []
            for modality in self.modality_names:
                if modality in modality_disentangled:
                    modality_features_list.append(
                        modality_disentangled[modality][dim_name]['z']
                    )

            if modality_features_list:
                modality_features = torch.stack(modality_features_list, dim=1)  # (batch, num_modalities, latent_dim)

                # 跨模态注意力融合
                fused_z, attention_weights = self.dimension_fusion[dim_name](
                    modality_features
                )

                fused[dim_name] = {
                    'z': fused_z,
                    'attention': attention_weights,
                    'modality_contributions': {
                        modality: modality_disentangled[modality][dim_name]
                        for modality in self.modality_names
                        if modality in modality_disentangled
                    }
                }
                attention_maps[dim_name] = attention_weights

        # 拼接所有融合后的解耦特征
        z_concat = torch.cat([
            fused['function']['z'],
            fused['aesthetics']['z'],
            fused['emotion']['z']
        ], dim=-1)

        results = {
            'z_function': fused['function']['z'],
            'z_aesthetics': fused['aesthetics']['z'],
            'z_emotion': fused['emotion']['z'],
            'z_concat': z_concat,
            'attention_maps': attention_maps,
            'modality_disentangled': modality_disentangled
        }

        # ===Step 3: 计算损失===
        if return_loss:
            total_loss = 0.0
            recon_loss = 0.0
            kl_loss = 0.0

            # 对每个模态计算VAE损失
            for modality, disentangled in modality_disentangled.items():
                # 拼接该模态的三个维度
                z_mod_concat = torch.cat([
                    disentangled['function']['z'],
                    disentangled['aesthetics']['z'],
                    disentangled['emotion']['z']
                ], dim=-1)

                # 重构（重构到共享空间）
                reconstructed = self.decoders[modality](z_mod_concat)

                # 重构损失（与原始特征的降维版本比较）
                target = F.adaptive_avg_pool1d(
                    multimodal_features[modality].unsqueeze(1),
                    self.latent_dim
                ).squeeze(1)
                recon_loss += F.mse_loss(reconstructed, target)

                # KL散度损失
                for dim_name in ['function', 'aesthetics', 'emotion']:
                    mu = disentangled[dim_name]['mu']
                    logvar = disentangled[dim_name]['logvar']
                    kl = self._kl_divergence(mu, logvar)
                    kl_loss += kl

            # 独立性损失（确保三个维度相互独立）
            independence_loss = self._independence_loss(
                fused['function']['z'],
                fused['aesthetics']['z'],
                fused['emotion']['z']
            )

            # 总损失
            total_loss = (
                recon_loss +
                self.beta * torch.clamp(kl_loss, 0, 10.0) +
                self.gamma * torch.clamp(independence_loss, 0, 1.0)
            )

            results.update({
                'loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'independence_loss': independence_loss
            })

        return results

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """计算KL散度"""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl / (mu.size(0) * mu.size(1))

    def _independence_loss(
        self,
        z_function: torch.Tensor,
        z_aesthetics: torch.Tensor,
        z_emotion: torch.Tensor
    ) -> torch.Tensor:
        """
        计算维度间的独立性损失
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


# 保持向后兼容性的别名
DisentangledRepresentation = DimensionSpecificMultimodalFusion
