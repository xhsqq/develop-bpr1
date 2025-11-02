"""
Demo: Using the Multimodal Sequential Recommender
演示如何使用多模态序列推荐模型
"""

import torch
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_recommender import MultimodalRecommender


def create_dummy_data(batch_size=2, seq_length=10, num_items=1000):
    """创建模拟数据"""
    # 物品ID序列
    item_ids = torch.randint(1, num_items, (batch_size, seq_length))

    # 多模态特征
    multimodal_features = {
        'text': torch.randn(batch_size, seq_length, 768),    # BERT特征
        'image': torch.randn(batch_size, seq_length, 2048),  # ResNet特征
        'audio': torch.randn(batch_size, seq_length, 128)    # 音频特征
    }

    # 序列长度
    seq_lengths = torch.tensor([seq_length, seq_length - 2])

    # 目标物品
    target_items = torch.randint(1, num_items, (batch_size,))

    return item_ids, multimodal_features, seq_lengths, target_items


def demo_basic_usage():
    """演示基本使用"""
    print("=" * 80)
    print("Demo 1: Basic Usage - 基本使用")
    print("=" * 80)

    # 1. 创建模型
    print("\n1. Creating model...")
    model = MultimodalRecommender(
        modality_dims={'text': 768, 'image': 2048, 'audio': 128},
        disentangled_dim=128,
        num_disentangled_dims=3,  # 功能、美学、情感
        num_interests=4,
        quantum_state_dim=256,
        hidden_dim=512,
        item_embed_dim=256,
        num_items=1000,
        max_seq_length=20
    )

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 2. 创建数据
    print("\n2. Creating dummy data...")
    item_ids, multimodal_features, seq_lengths, target_items = create_dummy_data()

    print(f"   Batch size: {item_ids.size(0)}")
    print(f"   Sequence length: {item_ids.size(1)}")

    # 3. 前向传播
    print("\n3. Forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            item_ids=item_ids,
            multimodal_features=multimodal_features,
            seq_lengths=seq_lengths,
            target_items=None,
            return_loss=False,
            return_explanations=True
        )

    print(f"   Recommendation logits shape: {outputs['recommendation_logits'].shape}")
    print(f"   User representation shape: {outputs['user_representation'].shape}")

    # 4. 获取Top-K推荐
    print("\n4. Getting Top-K recommendations...")
    top_k_items, top_k_scores = model.predict(
        item_ids,
        multimodal_features,
        seq_lengths,
        top_k=10
    )

    print(f"   Top-10 items for user 0: {top_k_items[0].tolist()}")
    print(f"   Top-10 scores for user 0: {top_k_scores[0].tolist()}")

    print("\n✓ Basic usage demo completed!\n")


def demo_disentangled_representation():
    """演示解耦表征学习"""
    print("=" * 80)
    print("Demo 2: Disentangled Representation - 解耦表征学习")
    print("=" * 80)

    from models.disentangled_representation import DisentangledRepresentation

    # 创建解耦模块
    print("\n1. Creating disentangled representation module...")
    disentangled_module = DisentangledRepresentation(
        input_dims={'text': 768, 'image': 2048, 'audio': 128},
        hidden_dim=512,
        shared_dim=256,
        disentangled_dim=128
    )

    # 创建多模态特征
    print("\n2. Creating multimodal features...")
    batch_size = 4
    multimodal_features = {
        'text': torch.randn(batch_size, 768),
        'image': torch.randn(batch_size, 2048),
        'audio': torch.randn(batch_size, 128)
    }

    # 提取解耦特征
    print("\n3. Extracting disentangled features...")
    disentangled_module.eval()
    with torch.no_grad():
        disentangled_features = disentangled_module.get_disentangled_features(
            multimodal_features
        )

    print(f"   Function dimension shape: {disentangled_features['function'].shape}")
    print(f"   Aesthetics dimension shape: {disentangled_features['aesthetics'].shape}")
    print(f"   Emotion dimension shape: {disentangled_features['emotion'].shape}")

    # 分析维度统计
    print("\n4. Analyzing dimension statistics...")
    for dim_name, features in disentangled_features.items():
        if dim_name == 'concat':
            continue
        mean = features.mean().item()
        std = features.std().item()
        print(f"   {dim_name.capitalize()}: mean={mean:.4f}, std={std:.4f}")

    print("\n✓ Disentangled representation demo completed!\n")


def demo_causal_inference():
    """演示因果推断"""
    print("=" * 80)
    print("Demo 3: Causal Inference - 因果推断")
    print("=" * 80)

    from models.causal_inference import CausalInferenceModule

    # 创建因果推断模块
    print("\n1. Creating causal inference module...")
    causal_module = CausalInferenceModule(
        disentangled_dim=128,
        num_dimensions=3,
        hidden_dim=256
    )

    # 创建解耦特征
    print("\n2. Creating disentangled features...")
    batch_size = 4
    disentangled_features = {
        'function': torch.randn(batch_size, 128),
        'aesthetics': torch.randn(batch_size, 128),
        'emotion': torch.randn(batch_size, 128)
    }

    # 进行因果推断
    print("\n3. Performing causal inference...")
    causal_module.eval()
    with torch.no_grad():
        causal_output = causal_module(
            disentangled_features,
            return_uncertainty=True,
            num_mc_samples=10
        )

    # 分析因果效应
    print("\n4. Analyzing causal effects...")
    if 'dimension_importance' in causal_output:
        importance = causal_output['dimension_importance']
        dimensions = ['Function (功能)', 'Aesthetics (美学)', 'Emotion (情感)']

        print("\n   Dimension Importance (维度重要性):")
        for i, (dim, imp) in enumerate(zip(dimensions, importance)):
            print(f"     {dim}: {imp:.4f}")

    # 分析不确定性
    print("\n5. Analyzing uncertainty...")
    uncertainty = causal_output['uncertainty']
    print(f"   Mean confidence: {uncertainty['confidence'].mean():.4f}")
    print(f"   Mean aleatoric uncertainty: {uncertainty['refined_aleatoric'].mean():.4f}")
    print(f"   Mean epistemic uncertainty: {uncertainty['refined_epistemic'].mean():.4f}")

    print("\n✓ Causal inference demo completed!\n")


def demo_quantum_inspired_encoder():
    """演示量子启发编码器"""
    print("=" * 80)
    print("Demo 4: Quantum-Inspired Multi-Interest Encoder - 量子启发多兴趣编码器")
    print("=" * 80)

    from models.quantum_inspired_encoder import QuantumInspiredMultiInterestEncoder

    # 创建量子编码器
    print("\n1. Creating quantum-inspired encoder...")
    quantum_encoder = QuantumInspiredMultiInterestEncoder(
        input_dim=384,  # 3 dimensions * 128
        state_dim=256,
        num_interests=4,
        hidden_dim=512,
        output_dim=256
    )

    # 创建输入特征
    print("\n2. Creating input features...")
    batch_size = 4
    features = torch.randn(batch_size, 384)

    # 编码
    print("\n3. Encoding with quantum-inspired representation...")
    quantum_encoder.eval()
    with torch.no_grad():
        quantum_output = quantum_encoder(
            features,
            return_all_interests=True
        )

    # 分析量子态
    print("\n4. Analyzing quantum states...")
    print(f"   Output shape: {quantum_output['output'].shape}")
    print(f"   Superposed state (real) shape: {quantum_output['superposed_state_real'].shape}")
    print(f"   Superposed state (imag) shape: {quantum_output['superposed_state_imag'].shape}")

    # 计算兴趣多样性
    print("\n5. Computing interest diversity...")
    diversity = quantum_encoder.get_interest_diversity(
        quantum_output['individual_interests_real'],
        quantum_output['individual_interests_imag']
    )
    print(f"   Average interest diversity: {diversity.mean():.4f}")

    # 分析干涉强度
    print("\n6. Analyzing interference patterns...")
    interference = quantum_output['interference_strength']
    print(f"   Interference strength matrix shape: {interference.shape}")
    print(f"   Mean interference strength: {interference.mean():.4f}")
    print(f"   Std interference strength: {interference.std():.4f}")

    print("\n✓ Quantum-inspired encoder demo completed!\n")


def demo_recommendation_explanation():
    """演示推荐解释"""
    print("=" * 80)
    print("Demo 5: Recommendation Explanation - 推荐可解释性")
    print("=" * 80)

    # 创建模型
    print("\n1. Creating model...")
    model = MultimodalRecommender(
        modality_dims={'text': 768, 'image': 2048, 'audio': 128},
        disentangled_dim=128,
        num_disentangled_dims=3,
        num_interests=4,
        quantum_state_dim=256,
        hidden_dim=512,
        item_embed_dim=256,
        num_items=1000
    )

    # 创建数据
    print("\n2. Creating data...")
    item_ids, multimodal_features, seq_lengths, _ = create_dummy_data(batch_size=1)

    # 获取推荐解释
    print("\n3. Generating recommendation explanation...")
    model.eval()
    with torch.no_grad():
        explanation = model.explain_recommendation(
            item_ids,
            multimodal_features,
            seq_lengths
        )

    # 打印解释
    print("\n4. Recommendation Explanation:")
    print("\n   Dimension Importance (维度重要性):")
    for dim, importance in explanation['dimension_importance'].items():
        print(f"     {dim}: {importance:.4f}")

    print("\n   Uncertainty (不确定性):")
    for key, value in explanation['uncertainty'].items():
        print(f"     {key}: {value:.4f}")

    print(f"\n   Interest Diversity (兴趣多样性): {explanation['interest_diversity']}")

    if 'causal_importance' in explanation:
        print("\n   Causal Importance (因果重要性):")
        for dim, importance in explanation['causal_importance'].items():
            print(f"     {dim}: {importance:.4f}")

    print("\n✓ Recommendation explanation demo completed!\n")


def demo_user_interests():
    """演示用户兴趣提取"""
    print("=" * 80)
    print("Demo 6: User Interest Extraction - 用户兴趣提取")
    print("=" * 80)

    # 创建模型
    print("\n1. Creating model...")
    model = MultimodalRecommender(
        modality_dims={'text': 768, 'image': 2048, 'audio': 128},
        disentangled_dim=128,
        num_disentangled_dims=3,
        num_interests=4,
        quantum_state_dim=256,
        hidden_dim=512,
        item_embed_dim=256,
        num_items=1000
    )

    # 创建数据
    print("\n2. Creating data...")
    item_ids, multimodal_features, seq_lengths, _ = create_dummy_data(batch_size=1)

    # 提取用户兴趣
    print("\n3. Extracting user interests...")
    model.eval()
    with torch.no_grad():
        interests = model.get_user_interests(
            item_ids,
            multimodal_features,
            seq_lengths
        )

    # 分析兴趣
    print("\n4. Analyzing user interests:")
    for i in range(4):
        interest_key = f'interest_{i}'
        if interest_key in interests:
            real = interests[interest_key]['real']
            imag = interests[interest_key]['imag']

            # 计算幅度和相位
            amplitude = torch.sqrt(real**2 + imag**2).mean().item()
            phase = torch.atan2(imag, real).mean().item()

            print(f"\n   Interest {i}:")
            print(f"     Amplitude (幅度): {amplitude:.4f}")
            print(f"     Phase (相位): {phase:.4f} rad")
            print(f"     Real mean: {real.mean():.4f}")
            print(f"     Imag mean: {imag.mean():.4f}")

    print("\n✓ User interest extraction demo completed!\n")


def main():
    """运行所有演示"""
    print("\n" + "=" * 80)
    print("Multimodal Sequential Recommender - Demo Suite")
    print("多模态序列推荐系统 - 演示集合")
    print("=" * 80 + "\n")

    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行所有演示
    demo_basic_usage()
    demo_disentangled_representation()
    demo_causal_inference()
    demo_quantum_inspired_encoder()
    demo_recommendation_explanation()
    demo_user_interests()

    print("=" * 80)
    print("All demos completed successfully!")
    print("所有演示完成!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
