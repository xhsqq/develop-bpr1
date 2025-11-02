"""
Improved Quantum-Inspired Multi-Interest Encoder
改进的量子启发多兴趣编码器：严格的相位、幺正性和量子测量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import numpy as np


class ComplexMultiHeadAttention(nn.Module):
    """严格的复数多头注意力"""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 复数Q/K/V投影
        self.qkv_real = nn.Linear(dim, 3 * dim)
        self.qkv_imag = nn.Linear(dim, 3 * dim)

        self.out_real = nn.Linear(dim, dim)
        self.out_imag = nn.Linear(dim, dim)

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复数注意力：Z = Attention(Q, K, V)
        Args:
            x_real/x_imag: (batch, num_interests, dim)
        Returns:
            (output_real, output_imag): (batch, num_interests, dim)
        """
        B, N, D = x_real.shape

        # QKV投影（复数）
        qkv_real = self.qkv_real(x_real).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_imag = self.qkv_imag(x_imag).reshape(B, N, 3, self.num_heads, self.head_dim)

        q_real, k_real, v_real = qkv_real.permute(2, 0, 3, 1, 4)
        q_imag, k_imag, v_imag = qkv_imag.permute(2, 0, 3, 1, 4)

        # 复数矩阵乘法：(Q_r + iQ_i)(K_r - iK_i)
        attn_real = (torch.matmul(q_real, k_real.transpose(-2, -1)) +
                    torch.matmul(q_imag, k_imag.transpose(-2, -1))) * self.scale
        attn_imag = (torch.matmul(q_imag, k_real.transpose(-2, -1)) -
                    torch.matmul(q_real, k_imag.transpose(-2, -1))) * self.scale

        # Softmax（应用在模长上）
        attn_magnitude = torch.sqrt(attn_real**2 + attn_imag**2)
        attn_weights = F.softmax(attn_magnitude, dim=-1)

        # 加权求和（复数）
        out_real = torch.matmul(attn_weights, v_real)
        out_imag = torch.matmul(attn_weights, v_imag)

        # Reshape
        out_real = out_real.transpose(1, 2).reshape(B, N, D)
        out_imag = out_imag.transpose(1, 2).reshape(B, N, D)

        # Output projection
        out_real = self.out_real(out_real)
        out_imag = self.out_imag(out_imag)

        return out_real, out_imag


class QuantumMeasurement(nn.Module):
    """量子测量算子（投影到经典空间）"""

    def __init__(self, num_interests: int, qubit_dim: int, output_dim: int):
        super().__init__()

        # 测量基（可学习的投影算子）
        self.measurement_basis_real = nn.Parameter(
            torch.randn(num_interests, qubit_dim, output_dim) / math.sqrt(qubit_dim)
        )
        self.measurement_basis_imag = nn.Parameter(
            torch.randn(num_interests, qubit_dim, output_dim) / math.sqrt(qubit_dim)
        )

    def forward(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        测量：⟨M|ψ⟩
        Args:
            states_real/imag: (batch, num_interests, qubit_dim)
        Returns:
            classical_output: (batch, output_dim)
            measurement_probs: (batch, num_interests)
        """
        batch_size, num_interests, qubit_dim = states_real.shape

        # 计算每个兴趣的测量结果：⟨M_i|ψ_i⟩
        measured_real = torch.einsum('bnd,ndo->bno', states_real, self.measurement_basis_real) - \
                       torch.einsum('bnd,ndo->bno', states_imag, self.measurement_basis_imag)
        measured_imag = torch.einsum('bnd,ndo->bno', states_real, self.measurement_basis_imag) + \
                       torch.einsum('bnd,ndo->bno', states_imag, self.measurement_basis_real)

        # 测量概率（Born rule）: P_i = |⟨M_i|ψ_i⟩|²
        measurement_probs = torch.sqrt(measured_real**2 + measured_imag**2).mean(dim=-1)
        measurement_probs = F.softmax(measurement_probs, dim=-1)  # (batch, num_interests)

        # 加权平均（概率性坍缩）
        output_real = (measured_real * measurement_probs.unsqueeze(-1)).sum(dim=1)
        output_imag = (measured_imag * measurement_probs.unsqueeze(-1)).sum(dim=1)

        # 取模长作为经典输出
        classical_output = torch.sqrt(output_real**2 + output_imag**2)

        return classical_output, measurement_probs


class ImprovedQuantumEncoder(nn.Module):
    """
    改进的量子编码器

    核心改进：
    1. 增加量子态数量到16
    2. 引入相位（phase）信息
    3. 幺正干涉矩阵（Cayley变换）
    4. 正确的量子测量
    5. 基于Fidelity的多样性损失
    """

    def __init__(
        self,
        input_dim: int = 192,
        num_interests: int = 16,  # ⭐ 从4增加到16
        qubit_dim: int = 128,
        output_dim: int = 256,
        hidden_dim: int = 512,
        use_quantum_computing: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_interests = num_interests
        self.qubit_dim = qubit_dim
        self.output_dim = output_dim

        # ===1. 量子态初始化（amplitude + phase）===
        self.amplitude_encoder_real = nn.Linear(input_dim, num_interests * qubit_dim)
        self.amplitude_encoder_imag = nn.Linear(input_dim, num_interests * qubit_dim)

        # ⭐ 相位编码（量子力学的关键）
        self.phase_encoder = nn.Sequential(
            nn.Linear(input_dim, num_interests),
            nn.Tanh()  # [-1, 1] → [-π, π]
        )

        # ===2. 幺正干涉矩阵===
        # 使用Cayley变换保证幺正性：U = (I + iA)(I - iA)^{-1}
        self.interference_params = nn.Parameter(
            torch.randn(num_interests, num_interests) * 0.01
        )

        # ===3. 复数注意力（量子纠缠建模）===
        self.complex_attention = ComplexMultiHeadAttention(
            dim=qubit_dim,
            num_heads=4
        )

        # ===4. 量子测量算子===
        self.measurement_operator = QuantumMeasurement(
            num_interests=num_interests,
            qubit_dim=qubit_dim,
            output_dim=output_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_interests: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
            return_all_interests: 是否返回所有兴趣

        Returns:
            包含量子态和测量结果的字典
        """
        batch_size = x.size(0)

        # 如果是序列输入，先聚合
        if x.dim() == 3:
            x = x.mean(dim=1)

        # ===Step 1: 编码为量子态（复数）===
        # Amplitude (幅度)
        amp_real = self.amplitude_encoder_real(x).view(
            batch_size, self.num_interests, self.qubit_dim
        )
        amp_imag = self.amplitude_encoder_imag(x).view(
            batch_size, self.num_interests, self.qubit_dim
        )

        # ⭐ Phase (相位) - 量子力学的核心
        phases = self.phase_encoder(x) * np.pi  # (batch, num_interests)

        # 应用相位：|ψ⟩ = A * e^{iφ}
        # e^{iφ} = cos(φ) + i*sin(φ)
        cos_phase = torch.cos(phases).unsqueeze(-1)  # (batch, num_interests, 1)
        sin_phase = torch.sin(phases).unsqueeze(-1)

        states_real = amp_real * cos_phase - amp_imag * sin_phase
        states_imag = amp_real * sin_phase + amp_imag * cos_phase

        # 归一化（保证 ||ψ||² = 1）
        norm = torch.sqrt((states_real**2 + states_imag**2).sum(dim=-1, keepdim=True) + 1e-8)
        states_real = states_real / norm
        states_imag = states_imag / norm

        # ===Step 2: 幺正干涉（量子态叠加）===
        # 构造幺正矩阵 U = (I + iA)(I - iA)^{-1}
        A = self.interference_params - self.interference_params.T  # 反对称
        I = torch.eye(self.num_interests, device=A.device, dtype=A.dtype)

        # Cayley变换（保证幺正性）
        # 使用复数计算
        numerator = I.unsqueeze(0) + 1j * A.unsqueeze(0)  # (1, n, n)
        denominator = I.unsqueeze(0) - 1j * A.unsqueeze(0)

        # 求解 U * denominator = numerator
        U = torch.linalg.solve(denominator, numerator)  # U是幺正矩阵

        # 应用干涉：|ψ'⟩ = U|ψ⟩
        states_complex = torch.complex(states_real, states_imag)
        # (batch, num_interests, qubit_dim) @ (1, num_interests, num_interests)

        # 转置以便矩阵乘法
        states_complex_t = states_complex.transpose(1, 2)  # (batch, qubit_dim, num_interests)
        interfered_t = torch.matmul(states_complex_t, U.squeeze(0))  # (batch, qubit_dim, num_interests)
        interfered = interfered_t.transpose(1, 2)  # (batch, num_interests, qubit_dim)

        states_real_interfered = interfered.real
        states_imag_interfered = interfered.imag

        # ===Step 3: 复数注意力（量子纠缠建模）===
        attended_real, attended_imag = self.complex_attention(
            states_real_interfered,
            states_imag_interfered
        )

        # ===Step 4: 量子测量（坍缩到经典）===
        classical_output, measurement_probs = self.measurement_operator(
            attended_real, attended_imag
        )

        # ===Step 5: 计算量子度量===
        metrics = self._compute_quantum_metrics(
            states_real_interfered,
            states_imag_interfered,
            phases
        )

        results = {
            'output': classical_output,  # (batch, output_dim)
            'measurement_probabilities': measurement_probs,
            'superposed_state_real': attended_real.mean(dim=1),
            'superposed_state_imag': attended_imag.mean(dim=1),
            'interference_strength': measurement_probs,  # 使用测量概率作为干涉强度
            'metrics': metrics
        }

        if return_all_interests:
            results['individual_interests_real'] = attended_real
            results['individual_interests_imag'] = attended_imag

        return results

    def _compute_quantum_metrics(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor,
        phases: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算量子力学相关度量
        """
        batch_size, num_interests, dim = states_real.shape

        # 1. Purity (纯度): Tr(ρ²)
        purity = self._compute_purity(states_real, states_imag)

        # 2. Entanglement (纠缠度): Von Neumann entropy
        entanglement = self._compute_entanglement(states_real, states_imag)

        # 3. Fidelity between quantum states (保真度)
        fidelity_matrix = self._compute_fidelity_matrix(
            states_real, states_imag
        )  # (batch, num_interests, num_interests)

        # 4. Phase variance (相位方差)
        phase_variance = phases.var(dim=1).mean()

        return {
            'purity': purity.mean(),
            'entanglement': entanglement.mean(),
            'fidelity_matrix': fidelity_matrix,
            'phase_variance': phase_variance
        }

    def _compute_purity(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        计算纯度: Tr(ρ²)
        对于纯态，purity = 1；对于混合态，purity < 1
        """
        batch_size, num_interests, dim = states_real.shape

        # 简化计算：使用迹的性质
        # Tr(ρ²) ≈ Σ|ψ_i|⁴ / (Σ|ψ_i|²)²
        magnitude_sq = states_real**2 + states_imag**2  # (batch, num_interests, dim)

        sum_sq = magnitude_sq.sum(dim=-1)  # (batch, num_interests)
        sum_fourth = (magnitude_sq**2).sum(dim=-1)  # (batch, num_interests)

        purity = sum_fourth / (sum_sq**2 + 1e-8)  # (batch, num_interests)
        purity = purity.mean(dim=-1)  # (batch,)

        return purity

    def _compute_entanglement(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        计算纠缠度（Von Neumann熵）
        S(ρ) = -Tr(ρ log ρ)
        """
        # 简化版本：使用purity估计熵
        # S ≈ -log(purity)
        purity = self._compute_purity(states_real, states_imag)
        entanglement = -torch.log(purity + 1e-8)

        return entanglement

    def _compute_fidelity_matrix(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        计算量子态间的Fidelity (Uhlmann-Jozsa)
        F(ρ, σ) = |⟨ψ|φ⟩|²
        """
        batch_size, num_interests, dim = states_real.shape

        # 归一化每个量子态
        norm = torch.sqrt((states_real**2 + states_imag**2).sum(dim=-1, keepdim=True) + 1e-8)
        states_real_norm = states_real / norm
        states_imag_norm = states_imag / norm

        fidelity = torch.zeros(batch_size, num_interests, num_interests,
                              device=states_real.device)

        for i in range(num_interests):
            for j in range(i, num_interests):
                # ⟨ψ_i|ψ_j⟩ = Σ_k (a_k - ib_k)(c_k + id_k)
                inner_real = (states_real_norm[:, i] * states_real_norm[:, j] +
                             states_imag_norm[:, i] * states_imag_norm[:, j]).sum(dim=-1)
                inner_imag = (states_real_norm[:, i] * states_imag_norm[:, j] -
                             states_imag_norm[:, i] * states_real_norm[:, j]).sum(dim=-1)

                # |⟨ψ_i|ψ_j⟩|²
                fid = inner_real**2 + inner_imag**2
                fid = torch.clamp(fid, 0.0, 1.0)  # 确保在[0,1]范围

                fidelity[:, i, j] = fid
                fidelity[:, j, i] = fid

        return fidelity

    def get_interest_diversity(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        计算兴趣多样性（基于Fidelity）
        """
        fidelity_matrix = self._compute_fidelity_matrix(states_real, states_imag)

        # 非对角线元素的平均（不同兴趣间的保真度，越低越多样）
        mask = 1 - torch.eye(self.num_interests, device=fidelity_matrix.device)
        off_diagonal_fidelity = (fidelity_matrix * mask).sum(dim=[1, 2]) / (mask.sum() + 1e-8)

        # 多样性 = 1 - 平均保真度
        diversity = 1 - off_diagonal_fidelity

        return diversity


def compute_quantum_losses(quantum_output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    基于量子理论的损失函数
    """
    metrics = quantum_output['metrics']

    # Loss 1: 多样性损失（基于Fidelity）
    # 不同兴趣应该正交（F → 0）
    fidelity_matrix = metrics['fidelity_matrix']  # (batch, N, N)

    # 只惩罚非对角线元素（不同态应该正交）
    num_interests = fidelity_matrix.size(1)
    mask = 1 - torch.eye(num_interests, device=fidelity_matrix.device)
    diversity_loss = (fidelity_matrix * mask).sum() / (mask.sum() + 1e-8)

    # Loss 2: 纯度正则（不要过于混合）
    purity = metrics['purity']
    purity_loss = F.relu(0.5 - purity)  # 鼓励纯度 > 0.5

    # Loss 3: 相位多样性（利用相位信息）
    phase_variance = metrics['phase_variance']
    phase_loss = F.relu(1.0 - phase_variance)  # 鼓励相位分散

    total_quantum_loss = (
        0.5 * diversity_loss +
        0.3 * purity_loss +
        0.2 * phase_loss
    )

    return total_quantum_loss


# 保持向后兼容性的别名
QuantumInspiredMultiInterestEncoder = ImprovedQuantumEncoder
