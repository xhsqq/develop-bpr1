"""
Quantum-Inspired Multi-Interest Encoder
量子启发的多兴趣编码器：使用复数表示和干涉机制建模用户多样化兴趣
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
import numpy as np


class ComplexLinear(nn.Module):
    """
    复数线性层
    实现复数域的线性变换: W * z，其中W和z都是复数
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        # 实部和虚部的权重
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input_real: torch.Tensor, input_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复数矩阵乘法: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

        Args:
            input_real: 输入的实部 (batch, in_features)
            input_imag: 输入的虚部 (batch, in_features)

        Returns:
            (output_real, output_imag)
        """
        # 复数乘法
        output_real = F.linear(input_real, self.weight_real) - F.linear(input_imag, self.weight_imag)
        output_imag = F.linear(input_real, self.weight_imag) + F.linear(input_imag, self.weight_real)

        # 添加偏置
        if self.bias_real is not None:
            output_real = output_real + self.bias_real
            output_imag = output_imag + self.bias_imag

        return output_real, output_imag


class ComplexActivation(nn.Module):
    """
    复数激活函数
    分别对幅度和相位应用激活
    """

    def __init__(self, activation_type: str = 'modReLU'):
        super().__init__()
        self.activation_type = activation_type

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            real: 实部
            imag: 虚部

        Returns:
            激活后的 (real, imag)
        """
        if self.activation_type == 'modReLU':
            # modReLU: 对幅度应用ReLU，保持相位
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
            phase = torch.atan2(imag, real)

            # ReLU on magnitude
            activated_magnitude = F.relu(magnitude)

            # 转换回实部和虚部
            activated_real = activated_magnitude * torch.cos(phase)
            activated_imag = activated_magnitude * torch.sin(phase)

            return activated_real, activated_imag

        elif self.activation_type == 'zReLU':
            # zReLU: 如果实部和虚部都大于0才激活
            mask = (real > 0) & (imag > 0)
            return real * mask.float(), imag * mask.float()

        elif self.activation_type == 'CReLU':
            # CReLU: 分别对实部和虚部应用ReLU
            return F.relu(real), F.relu(imag)

        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")


class QuantumState(nn.Module):
    """
    量子态表示
    使用复数表示用户的一个兴趣
    """

    def __init__(self, state_dim: int):
        """
        Args:
            state_dim: 量子态的维度（类似量子比特数对应的希尔伯特空间维度）
        """
        super().__init__()
        self.state_dim = state_dim

        # 初始化为归一化的复数态
        # |ψ⟩ = α|0⟩ + β|1⟩ + ... ，满足 Σ|α_i|^2 = 1
        self.state_real = nn.Parameter(torch.randn(state_dim) / math.sqrt(state_dim))
        self.state_imag = nn.Parameter(torch.randn(state_dim) / math.sqrt(state_dim))

    def get_normalized_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取归一化的量子态"""
        magnitude = torch.sqrt(self.state_real**2 + self.state_imag**2)
        norm = torch.sqrt(torch.sum(magnitude**2) + 1e-8)

        normalized_real = self.state_real / norm
        normalized_imag = self.state_imag / norm

        return normalized_real, normalized_imag

    def get_amplitude_phase(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取幅度和相位表示"""
        real, imag = self.get_normalized_state()
        amplitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
        return amplitude, phase


class QuantumInterference(nn.Module):
    """
    量子干涉机制
    模拟量子态的建设性和破坏性干涉
    """

    def __init__(self, state_dim: int, num_interests: int):
        """
        Args:
            state_dim: 量子态维度
            num_interests: 兴趣数量
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_interests = num_interests

        # 学习干涉矩阵（类似于量子门）
        self.interference_matrix_real = nn.Parameter(
            torch.eye(state_dim).repeat(num_interests, 1, 1)
        )
        self.interference_matrix_imag = nn.Parameter(
            torch.zeros(num_interests, state_dim, state_dim)
        )

    def forward(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor,
        interference_strength: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用量子干涉

        Args:
            states_real: 多个量子态的实部 (batch, num_interests, state_dim)
            states_imag: 多个量子态的虚部 (batch, num_interests, state_dim)
            interference_strength: 干涉强度 (batch, num_interests, num_interests)

        Returns:
            干涉后的量子态 (batch, state_dim)
        """
        batch_size = states_real.size(0)

        # 1. 应用干涉矩阵（类似量子门操作）
        # 对每个兴趣应用其对应的干涉矩阵
        interfered_states_real = []
        interfered_states_imag = []

        for i in range(self.num_interests):
            # 提取第i个兴趣的状态
            state_real_i = states_real[:, i, :]  # (batch, state_dim)
            state_imag_i = states_imag[:, i, :]

            # 应用干涉矩阵（复数矩阵乘法）
            W_real = self.interference_matrix_real[i]  # (state_dim, state_dim)
            W_imag = self.interference_matrix_imag[i]

            # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            interfered_real = torch.matmul(state_real_i, W_real.t()) - torch.matmul(state_imag_i, W_imag.t())
            interfered_imag = torch.matmul(state_real_i, W_imag.t()) + torch.matmul(state_imag_i, W_real.t())

            interfered_states_real.append(interfered_real)
            interfered_states_imag.append(interfered_imag)

        interfered_states_real = torch.stack(interfered_states_real, dim=1)  # (batch, num_interests, state_dim)
        interfered_states_imag = torch.stack(interfered_states_imag, dim=1)

        # 2. 量子叠加：将多个兴趣态叠加
        if interference_strength is not None:
            # 使用学习的干涉强度加权叠加
            # interference_strength: (batch, num_interests, num_interests)
            # 计算每对兴趣之间的干涉
            superposed_real = torch.zeros(batch_size, self.state_dim, device=states_real.device)
            superposed_imag = torch.zeros(batch_size, self.state_dim, device=states_imag.device)

            for i in range(self.num_interests):
                for j in range(self.num_interests):
                    # 兴趣i和j之间的干涉
                    strength = interference_strength[:, i, j].unsqueeze(-1)  # (batch, 1)

                    # 相位差导致的干涉
                    # |ψ_i⟩ + e^(iθ)|ψ_j⟩
                    phase_diff = torch.atan2(
                        interfered_states_imag[:, i, :] * interfered_states_real[:, j, :] -
                        interfered_states_real[:, i, :] * interfered_states_imag[:, j, :],
                        interfered_states_real[:, i, :] * interfered_states_real[:, j, :] +
                        interfered_states_imag[:, i, :] * interfered_states_imag[:, j, :]
                    )

                    # 建设性干涉 (constructive): cos(θ) > 0
                    # 破坏性干涉 (destructive): cos(θ) < 0
                    interference_factor = torch.cos(phase_diff)

                    contribution_real = strength * interference_factor * interfered_states_real[:, j, :]
                    contribution_imag = strength * interference_factor * interfered_states_imag[:, j, :]

                    superposed_real += contribution_real
                    superposed_imag += contribution_imag
        else:
            # 简单平均叠加
            superposed_real = interfered_states_real.mean(dim=1)
            superposed_imag = interfered_states_imag.mean(dim=1)

        # 3. 归一化（保持量子态的归一化条件）
        magnitude = torch.sqrt(superposed_real**2 + superposed_imag**2)
        norm = torch.sqrt(torch.sum(magnitude**2, dim=-1, keepdim=True) + 1e-8)

        superposed_real = superposed_real / norm
        superposed_imag = superposed_imag / norm

        return superposed_real, superposed_imag


class QuantumMeasurement(nn.Module):
    """
    量子测量模块
    将量子态"坍缩"到经典表示，用于最终的推荐预测
    """

    def __init__(self, state_dim: int, output_dim: int):
        """
        Args:
            state_dim: 量子态维度
            output_dim: 输出维度（推荐空间维度）
        """
        super().__init__()

        # 测量算子（类似于量子测量中的可观测量）
        self.measurement_operator_real = nn.Parameter(
            torch.randn(output_dim, state_dim) / math.sqrt(state_dim)
        )
        self.measurement_operator_imag = nn.Parameter(
            torch.randn(output_dim, state_dim) / math.sqrt(state_dim)
        )

        # 后处理层
        self.post_processing = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        state_real: torch.Tensor,
        state_imag: torch.Tensor,
        return_probabilities: bool = False
    ) -> torch.Tensor:
        """
        执行量子测量

        Args:
            state_real: 量子态实部 (batch, state_dim)
            state_imag: 量子态虚部 (batch, state_dim)
            return_probabilities: 是否返回测量概率分布

        Returns:
            测量结果 (batch, output_dim)
        """
        # 1. 应用测量算子（复数矩阵乘法）
        measured_real = (
            torch.matmul(state_real, self.measurement_operator_real.t()) -
            torch.matmul(state_imag, self.measurement_operator_imag.t())
        )
        measured_imag = (
            torch.matmul(state_real, self.measurement_operator_imag.t()) +
            torch.matmul(state_imag, self.measurement_operator_real.t())
        )

        # 2. 计算测量结果的幅度（Born rule: P(x) = |⟨x|ψ⟩|^2）
        measurement_amplitude = torch.sqrt(measured_real**2 + measured_imag**2)

        if return_probabilities:
            # 返回概率分布（归一化的平方幅度）
            probabilities = measurement_amplitude**2
            probabilities = probabilities / (torch.sum(probabilities, dim=-1, keepdim=True) + 1e-8)
            return probabilities
        else:
            # 返回经过后处理的特征
            output = self.post_processing(measurement_amplitude)
            return output


class QuantumInspiredMultiInterestEncoder(nn.Module):
    """
    量子启发的多兴趣编码器
    使用复数表示和量子干涉机制建模用户的多样化兴趣
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 256,
        num_interests: int = 4,
        hidden_dim: int = 512,
        output_dim: int = 256,
        use_quantum_computing: bool = False
    ):
        """
        Args:
            input_dim: 输入特征维度
            state_dim: 量子态维度
            num_interests: 用户兴趣数量
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            use_quantum_computing: 是否使用真实量子计算（需要安装qiskit或pennylane）
        """
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_interests = num_interests
        self.output_dim = output_dim
        self.use_quantum_computing = use_quantum_computing

        # 1. 输入编码：将经典特征编码为复数表示
        self.input_encoder_real = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim * num_interests)
        )

        self.input_encoder_imag = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim * num_interests)
        )

        # 2. 复数自注意力层（建模兴趣间的关系）
        self.complex_attention = ComplexMultiheadAttention(
            embed_dim=state_dim,
            num_heads=8,
            dropout=0.1
        )

        # 3. 量子干涉模块
        self.quantum_interference = QuantumInterference(state_dim, num_interests)

        # 4. 学习干涉强度
        self.interference_strength_network = nn.Sequential(
            nn.Linear(state_dim * num_interests * 2, hidden_dim),  # *2 for real and imag
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_interests * num_interests)
        )

        # 5. 量子测量模块
        self.quantum_measurement = QuantumMeasurement(state_dim, output_dim)

        # 6. 如果使用真实量子计算，初始化量子电路
        if use_quantum_computing:
            try:
                self._init_quantum_circuit()
            except ImportError:
                print("Warning: Quantum computing libraries not available. Falling back to classical simulation.")
                self.use_quantum_computing = False

    def _init_quantum_circuit(self):
        """初始化量子电路（可选）"""
        # 这里可以使用qiskit或pennylane构建量子电路
        # 示例代码（需要安装相应库）:
        # from qiskit import QuantumCircuit
        # num_qubits = int(np.ceil(np.log2(self.state_dim)))
        # self.quantum_circuit = QuantumCircuit(num_qubits)
        pass

    def forward(
        self,
        features: torch.Tensor,
        return_all_interests: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: 输入特征 (batch, input_dim) 或 (batch, seq_len, input_dim)
            return_all_interests: 是否返回所有单独的兴趣表示

        Returns:
            包含量子态和测量结果的字典
        """
        batch_size = features.size(0)

        # 如果是序列输入，先聚合
        if features.dim() == 3:
            features = features.mean(dim=1)  # 简单平均，也可以用注意力

        # 1. 编码为复数表示
        encoded_real = self.input_encoder_real(features)  # (batch, state_dim * num_interests)
        encoded_imag = self.input_encoder_imag(features)

        # Reshape为多个兴趣
        states_real = encoded_real.view(batch_size, self.num_interests, self.state_dim)
        states_imag = encoded_imag.view(batch_size, self.num_interests, self.state_dim)

        # 2. 复数自注意力（兴趣间的交互）
        attended_real, attended_imag = self.complex_attention(
            states_real, states_imag,
            states_real, states_imag,
            states_real, states_imag
        )

        # 3. 计算干涉强度
        # 将实部和虚部拼接作为输入
        interference_input = torch.cat([
            attended_real.reshape(batch_size, -1),
            attended_imag.reshape(batch_size, -1)
        ], dim=-1)

        interference_strength = self.interference_strength_network(interference_input)
        interference_strength = interference_strength.view(batch_size, self.num_interests, self.num_interests)
        interference_strength = torch.softmax(interference_strength, dim=-1)

        # 4. 应用量子干涉
        superposed_real, superposed_imag = self.quantum_interference(
            attended_real, attended_imag, interference_strength
        )

        # 5. 量子测量（坍缩到经典表示）
        measured_output = self.quantum_measurement(superposed_real, superposed_imag)
        measurement_probs = self.quantum_measurement(
            superposed_real, superposed_imag, return_probabilities=True
        )

        results = {
            'output': measured_output,
            'measurement_probabilities': measurement_probs,
            'superposed_state_real': superposed_real,
            'superposed_state_imag': superposed_imag,
            'interference_strength': interference_strength
        }

        if return_all_interests:
            results['individual_interests_real'] = attended_real
            results['individual_interests_imag'] = attended_imag

        return results

    def get_interest_diversity(
        self,
        states_real: torch.Tensor,
        states_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        计算兴趣多样性
        通过测量不同兴趣态之间的量子距离
        
        ⭐ 修复：归一化量子态，确保fidelity在[0,1]范围内
        """
        batch_size = states_real.size(0)
        diversity_scores = []

        for i in range(self.num_interests):
            for j in range(i + 1, self.num_interests):
                # ⭐ 归一化量子态（确保每个态是单位向量）
                state_i_real = states_real[:, i, :]
                state_i_imag = states_imag[:, i, :]
                state_j_real = states_real[:, j, :]
                state_j_imag = states_imag[:, j, :]
                
                # 计算范数
                norm_i = torch.sqrt((state_i_real**2 + state_i_imag**2).sum(dim=-1, keepdim=True) + 1e-8)
                norm_j = torch.sqrt((state_j_real**2 + state_j_imag**2).sum(dim=-1, keepdim=True) + 1e-8)
                
                # 归一化
                state_i_real = state_i_real / norm_i
                state_i_imag = state_i_imag / norm_i
                state_j_real = state_j_real / norm_j
                state_j_imag = state_j_imag / norm_j
                
                # 计算量子态之间的Fidelity: F(ρ, σ) = |⟨ψ_i|ψ_j⟩|^2
                inner_product_real = torch.sum(
                    state_i_real * state_j_real + state_i_imag * state_j_imag,
                    dim=-1
                )
                inner_product_imag = torch.sum(
                    state_i_imag * state_j_real - state_i_real * state_j_imag,
                    dim=-1
                )

                fidelity = inner_product_real**2 + inner_product_imag**2
                fidelity = torch.clamp(fidelity, 0.0, 1.0)  # ⭐ 确保在[0,1]范围
                diversity = 1 - fidelity  # 距离越大，多样性越高
                diversity_scores.append(diversity)

        # 平均多样性
        avg_diversity = torch.stack(diversity_scores).mean(dim=0)
        return avg_diversity


class ComplexMultiheadAttention(nn.Module):
    """
    复数多头自注意力
    在复数域上实现注意力机制
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Q, K, V投影（复数）
        self.q_proj_real = nn.Linear(embed_dim, embed_dim)
        self.q_proj_imag = nn.Linear(embed_dim, embed_dim)

        self.k_proj_real = nn.Linear(embed_dim, embed_dim)
        self.k_proj_imag = nn.Linear(embed_dim, embed_dim)

        self.v_proj_real = nn.Linear(embed_dim, embed_dim)
        self.v_proj_imag = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj_real = nn.Linear(embed_dim, embed_dim)
        self.out_proj_imag = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_real: torch.Tensor,
        query_imag: torch.Tensor,
        key_real: torch.Tensor,
        key_imag: torch.Tensor,
        value_real: torch.Tensor,
        value_imag: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复数自注意力

        Args:
            query_real/imag: (batch, num_interests, embed_dim)
            key_real/imag: (batch, num_interests, embed_dim)
            value_real/imag: (batch, num_interests, embed_dim)

        Returns:
            (output_real, output_imag): (batch, num_interests, embed_dim)
        """
        batch_size, tgt_len, embed_dim = query_real.size()
        src_len = key_real.size(1)

        # 投影Q, K, V
        q_real = self.q_proj_real(query_real)
        q_imag = self.q_proj_imag(query_imag)

        k_real = self.k_proj_real(key_real)
        k_imag = self.k_proj_imag(key_imag)

        v_real = self.v_proj_real(value_real)
        v_imag = self.v_proj_imag(value_imag)

        # Reshape为多头
        q_real = q_real.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_imag = q_imag.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_real = k_real.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_imag = k_imag.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        v_real = v_real.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_imag = v_imag.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数（使用复数内积的模）
        # ⟨q|k⟩ = q_real * k_real + q_imag * k_imag + i(q_imag * k_real - q_real * k_imag)
        # 注意力权重基于模: |⟨q|k⟩|
        attn_real = torch.matmul(q_real, k_real.transpose(-2, -1)) + torch.matmul(q_imag, k_imag.transpose(-2, -1))
        attn_imag = torch.matmul(q_imag, k_real.transpose(-2, -1)) - torch.matmul(q_real, k_imag.transpose(-2, -1))

        attn_magnitude = torch.sqrt(attn_real**2 + attn_imag**2 + 1e-8)
        attn_magnitude = attn_magnitude * self.scaling

        if attn_mask is not None:
            attn_magnitude = attn_magnitude + attn_mask

        attn_weights = F.softmax(attn_magnitude, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到value
        output_real = torch.matmul(attn_weights, v_real)
        output_imag = torch.matmul(attn_weights, v_imag)

        # Reshape回原始形状
        output_real = output_real.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        output_imag = output_imag.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)

        # 输出投影
        output_real = self.out_proj_real(output_real)
        output_imag = self.out_proj_imag(output_imag)

        return output_real, output_imag
