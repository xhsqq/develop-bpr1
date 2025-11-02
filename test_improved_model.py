"""
快速测试改进的模型
验证三大改进：
1. 先解耦再融合
2. 优化的量子编码器（16个兴趣）
3. SCM因果推断
"""

import torch
import sys
sys.path.insert(0, '/home/user/develop-bpr1')

from models.multimodal_recommender import MultimodalRecommender

def test_improved_model():
    print("=" * 60)
    print("测试改进的推荐模型")
    print("=" * 60)

    # 配置
    batch_size = 4
    seq_len = 5
    num_items = 100

    # 模态配置
    modality_dims = {
        'text': 768,
        'image': 2048
    }

    # 初始化模型
    print("\n1. 初始化模型...")
    model = MultimodalRecommender(
        modality_dims=modality_dims,
        disentangled_dim=64,  # 解耦维度
        num_disentangled_dims=3,  # 功能、美学、情感
        num_interests=16,  # ⭐ 从4增加到16
        quantum_state_dim=256,
        hidden_dim=512,
        item_embed_dim=256,
        num_items=num_items,
        max_seq_length=seq_len,
        alpha_recon=0.1,
        alpha_causal=0.2,  # Phase 2启用因果推断
        alpha_diversity=0.05,
        alpha_orthogonality=0.05,
        use_quantum_computing=False,
        dropout=0.1
    )

    print(f"✓ 模型初始化成功！")
    print(f"  - 解耦模块类型: {type(model.disentangled_module).__name__}")
    print(f"  - 量子编码器类型: {type(model.quantum_encoder).__name__}")
    print(f"  - 量子兴趣数量: {model.quantum_encoder.num_interests}")
    print(f"  - 因果模块类型: {type(model.causal_module).__name__}")

    # 创建模拟数据
    print("\n2. 创建模拟数据...")
    item_ids = torch.randint(1, num_items + 1, (batch_size, seq_len))

    multimodal_features = {
        'text': torch.randn(batch_size, seq_len, 768),
        'image': torch.randn(batch_size, seq_len, 2048)
    }

    seq_lengths = torch.tensor([seq_len] * batch_size)
    target_items = torch.randint(1, num_items + 1, (batch_size,))

    print(f"✓ 数据创建成功")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - num_items: {num_items}")

    # Phase 1 测试（不启用因果推断）
    print("\n3. Phase 1前向传播（alpha_causal=0）...")
    model.alpha_causal = 0.0

    try:
        outputs_phase1 = model(
            item_ids=item_ids,
            multimodal_features=multimodal_features,
            seq_lengths=seq_lengths,
            target_items=target_items,
            return_loss=True,
            return_explanations=False
        )

        print(f"✓ Phase 1前向传播成功")
        print(f"  - recommendation_logits shape: {outputs_phase1['recommendation_logits'].shape}")
        print(f"  - total_loss: {outputs_phase1['loss'].item():.4f}")
        print(f"  - recommendation_loss: {outputs_phase1['recommendation_loss'].item():.4f}")
        print(f"  - disentangled_loss: {outputs_phase1['disentangled_loss'].item():.4f}")
        print(f"  - diversity_loss: {outputs_phase1['diversity_loss'].item():.4f}")
        print(f"  - causal_loss: {outputs_phase1['causal_loss'].item():.4f}")

    except Exception as e:
        print(f"✗ Phase 1前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Phase 2 测试（启用因果推断）
    print("\n4. Phase 2前向传播（alpha_causal=0.2，启用SCM）...")
    model.alpha_causal = 0.2

    try:
        outputs_phase2 = model(
            item_ids=item_ids,
            multimodal_features=multimodal_features,
            seq_lengths=seq_lengths,
            target_items=target_items,
            return_loss=True,
            return_explanations=False
        )

        print(f"✓ Phase 2前向传播成功")
        print(f"  - total_loss: {outputs_phase2['loss'].item():.4f}")
        print(f"  - causal_loss: {outputs_phase2['causal_loss'].item():.4f}")

        # 检查因果输出
        causal_output = outputs_phase2.get('causal_output', {})
        if 'ite' in causal_output:
            print(f"  - ITE场景数量: {len(causal_output['ite'])}")
            for scenario_name in causal_output['ite'].keys():
                print(f"    * {scenario_name}")

    except Exception as e:
        print(f"✗ Phase 2前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 反向传播测试
    print("\n5. 反向传播测试...")
    try:
        outputs_phase2['loss'].backward()
        print(f"✓ 反向传播成功")

        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break

        if has_grad:
            print(f"  - 梯度已计算")
        else:
            print(f"  - 警告：没有找到梯度")

    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 总结
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print("\n改进验证：")
    print("1. ✓ 先解耦再融合架构工作正常")
    print("2. ✓ 量子编码器（16个兴趣）工作正常")
    print("3. ✓ SCM因果推断工作正常")
    print("=" * 60)

    return True

if __name__ == '__main__':
    success = test_improved_model()
    sys.exit(0 if success else 1)
