"""
Causal Inference Module
因果推断模块：包含个性化反事实生成器、因果效应估计器和不确定性量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class CounterfactualGenerator(nn.Module):
    """
    个性化反事实生成器
    基于解耦特征生成反事实样本，用于因果推断
    """

    def __init__(
        self,
        disentangled_dim: int,
        num_dimensions: int = 3,
        hidden_dim: int = 256,
        num_interventions: int = 5
    ):
        """
        Args:
            disentangled_dim: 每个解耦维度的大小
            num_dimensions: 解耦维度的数量（功能、美学、情感）
            hidden_dim: 隐藏层维度
            num_interventions: 每个维度的干预强度级别
        """
        super().__init__()

        self.disentangled_dim = disentangled_dim
        self.num_dimensions = num_dimensions
        self.num_interventions = num_interventions

        total_dim = disentangled_dim * num_dimensions

        # 干预网络：学习如何对特定维度进行干预
        self.intervention_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(disentangled_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, disentangled_dim)
            )
            for _ in range(num_dimensions)
        ])

        # 条件生成器：根据用户特征个性化反事实生成
        self.conditional_generator = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_dimensions * num_interventions)
        )

        # 可学习的干预强度
        self.intervention_strengths = nn.Parameter(
            torch.linspace(-1.0, 1.0, num_interventions)
        )

    def forward(
        self,
        disentangled_features: Dict[str, torch.Tensor],
        intervention_dim: Optional[int] = None,
        intervention_strength: Optional[float] = None,
        user_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成反事实样本

        Args:
            disentangled_features: 解耦特征字典 {'function', 'aesthetics', 'emotion'}
            intervention_dim: 要干预的维度 (0=function, 1=aesthetics, 2=emotion)
                             如果为None，则对所有维度生成反事实
            intervention_strength: 干预强度 [-1, 1]，如果为None则使用学习的强度
            user_context: 用户上下文特征，用于个性化反事实生成

        Returns:
            包含反事实特征的字典
        """
        # 提取特征
        z_function = disentangled_features['function']
        z_aesthetics = disentangled_features['aesthetics']
        z_emotion = disentangled_features['emotion']

        z_concat = torch.cat([z_function, z_aesthetics, z_emotion], dim=-1)
        batch_size = z_function.size(0)

        # 如果提供了用户上下文，计算个性化的干预参数
        if user_context is not None:
            intervention_params = self.conditional_generator(user_context)
            intervention_params = intervention_params.view(
                batch_size, self.num_dimensions, self.num_interventions
            )
        else:
            intervention_params = None

        counterfactuals = {}

        # 如果指定了特定维度，只对该维度生成反事实
        if intervention_dim is not None:
            dims_to_intervene = [intervention_dim]
        else:
            dims_to_intervene = list(range(self.num_dimensions))

        dimension_names = ['function', 'aesthetics', 'emotion']
        original_features = [z_function, z_aesthetics, z_emotion]

        for dim_idx in dims_to_intervene:
            dim_name = dimension_names[dim_idx]
            original_feature = original_features[dim_idx]

            # 应用干预
            intervened_feature = self.intervention_networks[dim_idx](original_feature)

            # 应用干预强度
            if intervention_strength is not None:
                # 使用指定的干预强度
                alpha = intervention_strength
                counterfactual_feature = (
                    (1 - abs(alpha)) * original_feature +
                    alpha * intervened_feature
                )
                counterfactuals[f'cf_{dim_name}_single'] = counterfactual_feature
            else:
                # 生成多个不同强度的反事实
                for strength_idx, strength in enumerate(self.intervention_strengths):
                    alpha = strength

                    # 如果有个性化参数，使用它们调整干预强度
                    if intervention_params is not None:
                        personalized_weight = intervention_params[:, dim_idx, strength_idx].unsqueeze(-1)
                        alpha = alpha * torch.sigmoid(personalized_weight)

                    # ⭐ v0.7.0 强干预策略: 80%干预 + 20%原始
                    counterfactual_feature = 0.8 * intervened_feature + 0.2 * original_feature

                    key = f'cf_{dim_name}_strength_{strength_idx}'
                    counterfactuals[key] = counterfactual_feature

        # 也保存原始特征
        counterfactuals['original'] = {
            'function': z_function,
            'aesthetics': z_aesthetics,
            'emotion': z_emotion
        }

        return counterfactuals

    def generate_all_counterfactuals(
        self,
        disentangled_features: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        生成所有可能的反事实组合

        Returns:
            反事实特征列表，每个元素是拼接后的特征
        """
        z_function = disentangled_features['function']
        z_aesthetics = disentangled_features['aesthetics']
        z_emotion = disentangled_features['emotion']

        all_counterfactuals = []

        # 对每个维度生成干预版本
        intervened_features = []
        for dim_idx, original in enumerate([z_function, z_aesthetics, z_emotion]):
            intervened = self.intervention_networks[dim_idx](original)
            intervened_features.append([original, intervened])  # [原始, 干预]

        # 生成所有组合 (2^3 = 8 种组合)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    cf = torch.cat([
                        intervened_features[0][i],
                        intervened_features[1][j],
                        intervened_features[2][k]
                    ], dim=-1)
                    all_counterfactuals.append(cf)

        return all_counterfactuals


class CausalEffectEstimator(nn.Module):
    """
    因果效应估计器
    估计不同维度对用户行为的因果影响
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_treatments: int = 3
    ):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_treatments: 处理（treatment）数量（对应解耦维度数量）
        """
        super().__init__()

        self.num_treatments = num_treatments

        # Propensity score network (倾向得分网络)
        # 估计用户接受某种"处理"的概率
        self.propensity_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_treatments)
        )

        # Outcome prediction networks (结果预测网络)
        # 为每个处理预测潜在结果
        self.outcome_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_treatments)
        ])

        # Treatment effect network (处理效应网络)
        self.treatment_effect_network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_treatments)
        )

    def forward(
        self,
        original_features: torch.Tensor,
        counterfactual_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        估计因果效应

        Args:
            original_features: 原始特征 (batch, feature_dim)
            counterfactual_features: 反事实特征 (batch, feature_dim)

        Returns:
            包含因果效应估计的字典
        """
        batch_size = original_features.size(0)

        # 1. 估计倾向得分
        propensity_scores = F.softmax(
            self.propensity_network(original_features), dim=-1
        )

        # 2. 预测原始和反事实结果
        original_outcomes = torch.stack([
            net(original_features).squeeze(-1)
            for net in self.outcome_networks
        ], dim=-1)  # (batch, num_treatments)

        counterfactual_outcomes = torch.stack([
            net(counterfactual_features).squeeze(-1)
            for net in self.outcome_networks
        ], dim=-1)  # (batch, num_treatments)

        # 3. 计算个体因果效应 (ITE - Individual Treatment Effect)
        # ITE = Y(1) - Y(0)，即接受处理vs不接受处理的结果差异
        ite = counterfactual_outcomes - original_outcomes

        # 4. 使用双重鲁棒估计器 (Doubly Robust Estimator)
        combined_features = torch.cat([original_features, counterfactual_features], dim=-1)
        direct_treatment_effect = self.treatment_effect_network(combined_features)

        # 5. 计算平均因果效应 (ATE - Average Treatment Effect)
        ate = ite.mean(dim=0)

        # 6. 使用倾向得分加权调整 ATE
        # IPW (Inverse Propensity Weighting)
        weights = 1.0 / (propensity_scores + 1e-6)
        weighted_ite = ite * weights
        weighted_ate = weighted_ite.mean(dim=0)

        return {
            'propensity_scores': propensity_scores,
            'original_outcomes': original_outcomes,
            'counterfactual_outcomes': counterfactual_outcomes,
            'ite': ite,  # 个体因果效应
            'ate': ate,  # 平均因果效应
            'weighted_ate': weighted_ate,  # 加权平均因果效应
            'direct_treatment_effect': direct_treatment_effect
        }

    def estimate_dimension_importance(
        self,
        causal_effects: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        估计每个解耦维度的重要性

        Args:
            causal_effects: forward()返回的因果效应字典

        Returns:
            维度重要性分数 (num_treatments,)
        """
        # 使用加权平均因果效应的绝对值作为重要性
        importance = torch.abs(causal_effects['weighted_ate'])
        importance = F.softmax(importance, dim=-1)
        return importance


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
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_ensembles: 集成模型数量
            dropout_rate: Dropout比率（用于MC Dropout）
        """
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

        # 1. Deep Ensemble: 使用多个模型预测
        ensemble_predictions = []
        for head in self.ensemble_heads:
            pred = head(features)
            ensemble_predictions.append(pred)

        ensemble_predictions = torch.stack(ensemble_predictions, dim=-1)  # (batch, 1, num_ensembles)
        ensemble_predictions = ensemble_predictions.squeeze(1)  # (batch, num_ensembles)

        # 2. Monte Carlo Dropout: 多次前向传播
        mc_predictions = []
        for _ in range(num_mc_samples):
            # 即使在eval模式下也启用dropout
            pred = self.ensemble_heads[0](features)
            mc_predictions.append(pred)

        mc_predictions = torch.stack(mc_predictions, dim=-1)  # (batch, 1, num_mc_samples)
        mc_predictions = mc_predictions.squeeze(1)  # (batch, num_mc_samples)

        # 3. 计算预测统计量
        # 集成预测的均值和方差
        ensemble_mean = ensemble_predictions.mean(dim=-1)
        ensemble_var = ensemble_predictions.var(dim=-1)

        # MC Dropout预测的均值和方差
        mc_mean = mc_predictions.mean(dim=-1)
        mc_var = mc_predictions.var(dim=-1)

        # 4. 分解不确定性
        # Aleatoric uncertainty (数据不确定性): MC方差的均值
        aleatoric_uncertainty = mc_var

        # Epistemic uncertainty (模型不确定性): 集成预测的方差
        epistemic_uncertainty = ensemble_var

        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        # 5. 使用神经网络进一步细化不确定性估计
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

    def get_confidence_interval(
        self,
        uncertainty_output: Dict[str, torch.Tensor],
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算预测的置信区间

        Args:
            uncertainty_output: forward()返回的不确定性输出
            confidence_level: 置信水平

        Returns:
            (lower_bound, upper_bound)
        """
        from scipy import stats

        # 计算z分数
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        mean = uncertainty_output['prediction']
        std = torch.sqrt(uncertainty_output['total_uncertainty'])

        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std

        return lower_bound, upper_bound


class CausalInferenceModule(nn.Module):
    """
    完整的因果推断模块
    整合反事实生成、因果效应估计和不确定性量化
    """

    def __init__(
        self,
        disentangled_dim: int = 128,
        num_dimensions: int = 3,
        hidden_dim: int = 256,
        num_ensembles: int = 5
    ):
        """
        Args:
            disentangled_dim: 解耦维度大小
            num_dimensions: 解耦维度数量
            hidden_dim: 隐藏层维度
            num_ensembles: 不确定性量化的集成数量
        """
        super().__init__()

        self.disentangled_dim = disentangled_dim
        self.num_dimensions = num_dimensions
        total_dim = disentangled_dim * num_dimensions

        # 三个核心组件
        self.counterfactual_generator = CounterfactualGenerator(
            disentangled_dim=disentangled_dim,
            num_dimensions=num_dimensions,
            hidden_dim=hidden_dim
        )

        self.causal_effect_estimator = CausalEffectEstimator(
            feature_dim=total_dim,
            hidden_dim=hidden_dim,
            num_treatments=num_dimensions
        )

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
        完整的因果推断流程

        Args:
            disentangled_features: 解耦特征字典
            return_uncertainty: 是否返回不确定性估计
            num_mc_samples: MC采样次数

        Returns:
            包含反事实、因果效应和不确定性的完整结果
        """
        # 1. 生成反事实
        counterfactuals = self.counterfactual_generator(disentangled_features)

        # 2. 拼接原始特征
        original_concat = torch.cat([
            disentangled_features['function'],
            disentangled_features['aesthetics'],
            disentangled_features['emotion']
        ], dim=-1)

        results = {
            'counterfactuals': counterfactuals,
            'original_features': original_concat
        }

        # 3. 对每个反事实估计因果效应
        causal_effects_list = []
        for cf_key, cf_feature in counterfactuals.items():
            if cf_key == 'original':
                continue

            # 如果反事实特征是单个维度，需要重构完整特征
            if cf_feature.size(-1) == self.disentangled_dim:
                # 确定哪个维度被干预了
                if 'function' in cf_key:
                    cf_concat = torch.cat([
                        cf_feature,
                        disentangled_features['aesthetics'],
                        disentangled_features['emotion']
                    ], dim=-1)
                elif 'aesthetics' in cf_key:
                    cf_concat = torch.cat([
                        disentangled_features['function'],
                        cf_feature,
                        disentangled_features['emotion']
                    ], dim=-1)
                elif 'emotion' in cf_key:
                    cf_concat = torch.cat([
                        disentangled_features['function'],
                        disentangled_features['aesthetics'],
                        cf_feature
                    ], dim=-1)
            else:
                cf_concat = cf_feature

            # 估计因果效应
            causal_effect = self.causal_effect_estimator(original_concat, cf_concat)
            causal_effects_list.append(causal_effect)

        # 4. 不确定性量化
        if return_uncertainty:
            uncertainty = self.uncertainty_quantification(
                original_concat,
                num_mc_samples=num_mc_samples
            )
            results['uncertainty'] = uncertainty

        # 5. 汇总因果效应
        if causal_effects_list:
            # 平均所有反事实的因果效应
            avg_causal_effects = {
                'ate': torch.stack([ce['ate'] for ce in causal_effects_list]).mean(dim=0),
                'weighted_ate': torch.stack([ce['weighted_ate'] for ce in causal_effects_list]).mean(dim=0),
                'ite': torch.stack([ce['ite'] for ce in causal_effects_list]).mean(dim=0),  # ⭐ 添加ITE
            }
            results['causal_effects'] = avg_causal_effects

            # 估计维度重要性
            dimension_importance = self.causal_effect_estimator.estimate_dimension_importance(
                avg_causal_effects
            )
            results['dimension_importance'] = dimension_importance

        return results

    def get_causal_explanation(
        self,
        disentangled_features: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """
        获取可解释的因果分析结果

        Returns:
            包含可解释信息的字典
        """
        with torch.no_grad():
            results = self.forward(disentangled_features, return_uncertainty=True)

            explanation = {
                'dimension_importance': results.get('dimension_importance'),
                'confidence': results['uncertainty']['confidence'],
                'aleatoric_uncertainty': results['uncertainty']['aleatoric_uncertainty'],
                'epistemic_uncertainty': results['uncertainty']['epistemic_uncertainty'],
            }

            if 'causal_effects' in results:
                explanation['average_treatment_effect'] = results['causal_effects']['weighted_ate']

            return explanation
