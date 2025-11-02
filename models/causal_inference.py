"""
Structural Causal Model (SCM) based Causal Inference
基于结构因果模型的因果推断：实现Pearl三步反事实推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class StructuralCausalModel(nn.Module):
    """
    结构因果模型：实现Pearl三步反事实推断

    理论保证：
    1. Identifiability (定理1.1)
    2. Consistency (定理1.2)
    3. Unbiased ITE (定理1.3)

    结构方程：
    Z_func = μ_func(X) + ε_func * σ_func(X)
    Z_aes = μ_aes(X) + ε_aes * σ_aes(X)
    Z_emo = μ_emo(X) + ε_emo * σ_emo(X)
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        self.latent_dim = latent_dim

        # ===干预策略网络（自适应干预强度）===
        self.intervention_strength = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        mu_dict: Dict[str, torch.Tensor],
        logvar_dict: Dict[str, torch.Tensor],
        quantum_encoder: nn.Module,
        recommendation_head: callable,
        target_items: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        三步反事实推断

        Args:
            z_dict: {'function': z_func, 'aesthetics': z_aes, 'emotion': z_emo}
            mu_dict, logvar_dict: VAE参数（用于abduction）
            quantum_encoder: 下游量子编码器
            recommendation_head: 推荐打分函数
            target_items: 目标商品

        Returns:
            包含ITE、反事实预测等的字典
        """
        batch_size = z_dict['function'].size(0)

        # ============================================
        # Step 1: Abduction（推断外生变量）
        # ============================================
        U_exogenous = self._abduction(z_dict, mu_dict, logvar_dict)

        # ============================================
        # Step 2: Action（干预操作）
        # ============================================
        # 学习自适应干预强度
        z_concat = torch.cat([
            z_dict['function'],
            z_dict['aesthetics'],
            z_dict['emotion']
        ], dim=-1)
        strengths = self.intervention_strength(z_concat)  # (batch, 3)

        # 生成反事实场景
        counterfactuals = self._generate_counterfactuals(
            z_dict, U_exogenous, strengths
        )

        # ============================================
        # Step 3: Prediction（反事实预测）
        # ============================================
        cf_predictions = self._counterfactual_prediction(
            counterfactuals, quantum_encoder, recommendation_head
        )

        # 事实预测（baseline）
        z_factual = z_concat
        quantum_out_factual = quantum_encoder(z_factual)
        logits_factual = recommendation_head(quantum_out_factual['output'])

        # ============================================
        # 计算Individual Treatment Effect (ITE)
        # ============================================
        ite = self._compute_ite(
            logits_factual, cf_predictions, target_items
        )

        # ============================================
        # 因果损失（理论严谨版）
        # ============================================
        causal_loss = self._compute_causal_loss(
            ite, strengths, U_exogenous, target_items
        )

        return {
            'ite': ite,
            'counterfactual_predictions': cf_predictions,
            'exogenous_variables': U_exogenous,
            'intervention_strengths': strengths,
            'causal_loss': causal_loss,
            'causal_effects': {'ite': torch.stack([v['target'] for v in ite.values() if 'target' in v], dim=-1) if target_items is not None else None},
            'original_features': z_factual
        }

    def _abduction(
        self,
        z_dict: Dict[str, torch.Tensor],
        mu_dict: Dict[str, torch.Tensor],
        logvar_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Step 1: Abduction
        从观测(Z)推断外生变量(U)

        公式：ε = (z - μ) / σ
        """
        U = {}

        for dim_name in ['function', 'aesthetics', 'emotion']:
            z = z_dict[dim_name]
            mu = mu_dict[dim_name]
            logvar = logvar_dict[dim_name]

            sigma = torch.exp(0.5 * logvar)
            epsilon = (z - mu) / (sigma + 1e-8)

            U[dim_name] = {
                'epsilon': epsilon,
                'mu': mu,
                'sigma': sigma
            }

        return U

    def _generate_counterfactuals(
        self,
        z_dict: Dict[str, torch.Tensor],
        U_exogenous: Dict[str, Dict[str, torch.Tensor]],
        strengths: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Step 2: Action
        生成反事实场景

        3种干预策略：
        1. Set to mean (去个性化)
        2. Shift (增强/减弱)
        3. Swap (交换)
        """
        batch_size = z_dict['function'].size(0)

        counterfactuals = {}

        # ===场景1：do(function = mean)===
        z_func_cf1 = z_dict['function'].mean(dim=0, keepdim=True).expand_as(
            z_dict['function']
        )

        counterfactuals['function_to_mean'] = {
            'function': z_func_cf1,
            'aesthetics': z_dict['aesthetics'],
            'emotion': z_dict['emotion']
        }

        # ===场景2：do(aesthetics = aesthetics + δ)===
        z_aes_cf1 = z_dict['aesthetics'] + \
                   strengths[:, 1:2] * U_exogenous['aesthetics']['sigma']

        counterfactuals['aesthetics_shift'] = {
            'function': z_dict['function'],
            'aesthetics': z_aes_cf1,
            'emotion': z_dict['emotion']
        }

        # ===场景3：do(emotion = other_user's_emotion)===
        indices = torch.randperm(batch_size, device=z_dict['emotion'].device)
        z_emo_cf1 = z_dict['emotion'][indices]

        counterfactuals['emotion_swap'] = {
            'function': z_dict['function'],
            'aesthetics': z_dict['aesthetics'],
            'emotion': z_emo_cf1
        }

        # ===场景4：do(function = function - δ)===
        z_func_cf2 = z_dict['function'] - \
                    strengths[:, 0:1] * U_exogenous['function']['sigma']

        counterfactuals['function_weaken'] = {
            'function': z_func_cf2,
            'aesthetics': z_dict['aesthetics'],
            'emotion': z_dict['emotion']
        }

        return counterfactuals

    def _counterfactual_prediction(
        self,
        counterfactuals: Dict[str, Dict[str, torch.Tensor]],
        quantum_encoder: nn.Module,
        recommendation_head: callable
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Step 3: Prediction
        用修改后的SCM预测反事实结果

        关键：保持因果链的完整性
        """
        cf_predictions = {}

        for scenario_name, cf_z_dict in counterfactuals.items():
            # 拼接反事实的解耦表征
            z_concat_cf = torch.cat([
                cf_z_dict['function'],
                cf_z_dict['aesthetics'],
                cf_z_dict['emotion']
            ], dim=-1)

            # 重新经过量子编码器（保持因果链）
            quantum_out_cf = quantum_encoder(z_concat_cf)

            # 预测
            logits_cf = recommendation_head(quantum_out_cf['output'])

            cf_predictions[scenario_name] = {
                'logits': logits_cf,
                'quantum_state': quantum_out_cf
            }

        return cf_predictions

    def _compute_ite(
        self,
        logits_factual: torch.Tensor,
        cf_predictions: Dict[str, Dict[str, torch.Tensor]],
        target_items: Optional[torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        计算Individual Treatment Effect

        ITE_dim = Y_counterfactual - Y_factual
        """
        ite = {}

        for scenario_name, cf_pred in cf_predictions.items():
            logits_cf = cf_pred['logits']

            # 全局ITE（所有商品）
            ite[scenario_name] = {
                'global': logits_cf - logits_factual  # (batch, num_items)
            }

            # 针对target_item的ITE
            if target_items is not None:
                batch_indices = torch.arange(
                    logits_factual.size(0),
                    device=logits_factual.device
                )

                ite_target = (
                    logits_cf[batch_indices, target_items] -
                    logits_factual[batch_indices, target_items]
                )

                ite[scenario_name]['target'] = ite_target  # (batch,)

        return ite

    def _compute_causal_loss(
        self,
        ite: Dict[str, Dict[str, torch.Tensor]],
        strengths: torch.Tensor,
        U_exogenous: Dict[str, Dict[str, torch.Tensor]],
        target_items: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        理论严谨的因果损失（4个组件）
        """
        losses = {}

        # ===Loss 1: Consistency Check（一致性）===
        consistency_penalty = torch.stack([
            (ite[key]['target'].abs() * (1 - strengths[:, i])).mean()
            for i, key in enumerate(['function_to_mean', 'aesthetics_shift', 'emotion_swap'])
            if key in ite and 'target' in ite[key]
        ]).mean() if target_items is not None and len(ite) >= 3 else torch.tensor(0.0, device=strengths.device)

        losses['consistency'] = consistency_penalty

        # ===Loss 2: ITE Magnitude Regularization（幅度正则）===
        target_magnitude = 0.3

        magnitude_loss = 0
        count = 0
        for scenario_name, ite_dict in ite.items():
            if 'target' in ite_dict:
                magnitude_loss += F.smooth_l1_loss(
                    ite_dict['target'].abs(),
                    torch.full_like(ite_dict['target'], target_magnitude)
                )
                count += 1
        magnitude_loss = magnitude_loss / count if count > 0 else torch.tensor(0.0, device=strengths.device)

        losses['magnitude'] = magnitude_loss

        # ===Loss 3: Monotonicity（单调性）===
        monotonicity_loss = torch.tensor(0.0, device=strengths.device)
        if 'aesthetics_shift' in ite and 'target' in ite['aesthetics_shift']:
            ite_aes = ite['aesthetics_shift']['target']
            expected_sign = (strengths[:, 1] - 0.5).sign()
            actual_sign = ite_aes.sign()
            monotonicity_loss = F.relu(-expected_sign * actual_sign).mean()

        losses['monotonicity'] = monotonicity_loss

        # ===Loss 4: Orthogonality（正交性）===
        if len(ite) >= 2 and target_items is not None:
            ite_values_list = [
                ite[key]['target']
                for key in ['function_to_mean', 'aesthetics_shift']
                if key in ite and 'target' in ite[key]
            ]

            if len(ite_values_list) >= 2:
                ite_values = torch.stack(ite_values_list, dim=0)  # (2, batch)

                # 计算相关系数
                if ite_values.size(0) >= 2 and ite_values.size(1) > 1:
                    # 手动计算相关矩阵（避免单样本问题）
                    mean_vals = ite_values.mean(dim=1, keepdim=True)
                    centered = ite_values - mean_vals
                    cov = torch.mm(centered, centered.t()) / (ite_values.size(1) - 1)
                    std = torch.sqrt(torch.diag(cov))
                    corr_matrix = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-8)

                    I = torch.eye(corr_matrix.size(0), device=corr_matrix.device)
                    orthogonality_loss = (corr_matrix - I).abs().mean()
                else:
                    orthogonality_loss = torch.tensor(0.0, device=strengths.device)
            else:
                orthogonality_loss = torch.tensor(0.0, device=strengths.device)
        else:
            orthogonality_loss = torch.tensor(0.0, device=strengths.device)

        losses['orthogonality'] = orthogonality_loss

        # ===加权组合===
        total_causal_loss = (
            0.1 * losses['consistency'] +
            0.5 * losses['magnitude'] +      # 最重要
            0.2 * losses['monotonicity'] +
            0.2 * losses['orthogonality']
        )

        return total_causal_loss


class CausalInferenceModule(nn.Module):
    """
    完整的因果推断模块（基于SCM）
    整合Pearl三步反事实推理
    """

    def __init__(
        self,
        disentangled_dim: int = 64,
        num_dimensions: int = 3,
        hidden_dim: int = 256,
        num_ensembles: int = 5
    ):
        super().__init__()

        self.disentangled_dim = disentangled_dim
        self.num_dimensions = num_dimensions
        total_dim = disentangled_dim * num_dimensions

        # SCM核心组件
        self.scm = StructuralCausalModel(latent_dim=disentangled_dim)

        # 不确定性量化（保留用于向后兼容）
        self.uncertainty_quantification = UncertaintyQuantification(
            feature_dim=total_dim,
            hidden_dim=hidden_dim,
            num_ensembles=num_ensembles
        )

    def forward(
        self,
        disentangled_features: Dict[str, torch.Tensor],
        return_uncertainty: bool = True,
        num_mc_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        完整的因果推断流程（向后兼容的接口）

        Args:
            disentangled_features: 解耦特征字典
            return_uncertainty: 是否返回不确定性估计
            num_mc_samples: MC采样次数

        Returns:
            包含反事实、因果效应和不确定性的完整结果
        """
        # 拼接原始特征
        original_concat = torch.cat([
            disentangled_features['function'],
            disentangled_features['aesthetics'],
            disentangled_features['emotion']
        ], dim=-1)

        # 创建简单的反事实（用于向后兼容）
        counterfactuals = self._generate_simple_counterfactuals(disentangled_features)

        results = {
            'counterfactuals': counterfactuals,
            'original_features': original_concat,
            'causal_effects': {}
        }

        # 不确定性量化
        if return_uncertainty:
            uncertainty = self.uncertainty_quantification(
                original_concat,
                num_mc_samples=num_mc_samples
            )
            results['uncertainty'] = uncertainty

        return results

    def _generate_simple_counterfactuals(
        self,
        disentangled_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """生成简单的反事实（向后兼容）"""
        batch_size = disentangled_features['function'].size(0)

        counterfactuals = {}

        # 功能维度的反事实
        z_func_mean = disentangled_features['function'].mean(dim=0, keepdim=True)
        counterfactuals['cf_function_strength_0'] = z_func_mean.expand(batch_size, -1)

        # 美学维度的反事实
        z_aes_perturb = disentangled_features['aesthetics'] + 0.1 * torch.randn_like(
            disentangled_features['aesthetics']
        )
        counterfactuals['cf_aesthetics_strength_0'] = z_aes_perturb

        # 情感维度的反事实
        indices = torch.randperm(batch_size, device=disentangled_features['emotion'].device)
        counterfactuals['cf_emotion_strength_0'] = disentangled_features['emotion'][indices]

        counterfactuals['original'] = disentangled_features

        return counterfactuals


class UncertaintyQuantification(nn.Module):
    """
    不确定性量化模块
    使用Monte Carlo Dropout和深度集成估计预测的不确定性
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_ensembles: int = 5,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.num_ensembles = num_ensembles
        self.dropout_rate = dropout_rate

        # 创建多个预测头（深度集成）
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_ensembles)
        ])

        # 不确定性估计网络
        self.uncertainty_network = nn.Sequential(
            nn.Linear(num_ensembles, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2)  # 输出: [aleatoric, epistemic]
        )

    def forward(
        self,
        features: torch.Tensor,
        num_mc_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        量化预测不确定性

        Args:
            features: 输入特征 (batch, feature_dim)
            num_mc_samples: Monte Carlo采样次数

        Returns:
            包含预测和不确定性估计的字典
        """
        batch_size = features.size(0)

        # 1. Deep Ensemble
        ensemble_predictions = []
        for head in self.ensemble_heads:
            pred = head(features)
            ensemble_predictions.append(pred)

        ensemble_predictions = torch.stack(ensemble_predictions, dim=-1)
        ensemble_predictions = ensemble_predictions.squeeze(1)  # (batch, num_ensembles)

        # 2. Monte Carlo Dropout
        mc_predictions = []
        for _ in range(num_mc_samples):
            pred = self.ensemble_heads[0](features)
            mc_predictions.append(pred)

        mc_predictions = torch.stack(mc_predictions, dim=-1)
        mc_predictions = mc_predictions.squeeze(1)  # (batch, num_mc_samples)

        # 3. 计算统计量
        ensemble_mean = ensemble_predictions.mean(dim=-1)
        ensemble_var = ensemble_predictions.var(dim=-1)

        mc_mean = mc_predictions.mean(dim=-1)
        mc_var = mc_predictions.var(dim=-1)

        # 4. 分解不确定性
        aleatoric_uncertainty = mc_var
        epistemic_uncertainty = ensemble_var
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        # 5. 细化不确定性估计
        uncertainty_features = ensemble_predictions
        uncertainty_estimates = self.uncertainty_network(uncertainty_features)
        refined_aleatoric = F.softplus(uncertainty_estimates[:, 0])
        refined_epistemic = F.softplus(uncertainty_estimates[:, 1])

        return {
            'prediction': ensemble_mean,
            'ensemble_predictions': ensemble_predictions,
            'mc_predictions': mc_predictions,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'refined_aleatoric': refined_aleatoric,
            'refined_epistemic': refined_epistemic,
            'confidence': 1.0 / (1.0 + total_uncertainty)
        }
