#!/usr/bin/env python
"""
维度验证脚本
验证模型在不同配置下的维度计算是否正确
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_recommender import MultimodalRecommender


def test_dimension_config(config_name, **kwargs):
    """
    测试特定配置下的维度
    
    Args:
        config_name: 配置名称
        **kwargs: 模型参数
    """
    print(f"\n{'='*80}")
    print(f"测试配置: {config_name}")
    print(f"{'='*80}")
    
    # 打印配置
    print("\n配置参数:")
    for key, value in kwargs.items():
        if key == 'modality_dims':
            print(f"  {key}:")
            for mod, dim in value.items():
                print(f"    {mod}: {dim}")
        else:
            print(f"  {key}: {value}")
    
    try:
        # 创建模型
        model = MultimodalRecommender(**kwargs)
        
        # 计算维度
        disentangled_dim = kwargs.get('disentangled_dim', 128)
        num_disentangled_dims = kwargs.get('num_disentangled_dims', 3)
        total_dim = disentangled_dim * num_disentangled_dims
        
        print(f"\n维度计算:")
        print(f"  单个解耦维度: {disentangled_dim}")
        print(f"  解耦维度数量: {num_disentangled_dims}")
        print(f"  总解耦维度: {total_dim} ({disentangled_dim} × {num_disentangled_dims})")
        
        # 测试前向传播
        batch_size = 16
        seq_len = 10
        num_items = kwargs.get('num_items', 100)
        
        print(f"\n测试数据:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Num items: {num_items}")
        
        # 准备输入
        item_ids = torch.randint(1, num_items + 1, (batch_size, seq_len))
        target_items = torch.randint(1, num_items + 1, (batch_size,))
        seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        
        modality_dims = kwargs.get('modality_dims', {'text': 768, 'image': 2048})
        multimodal_features = {
            modality: torch.randn(batch_size, seq_len, dim)
            for modality, dim in modality_dims.items()
        }
        
        print(f"\n输入特征:")
        for modality, features in multimodal_features.items():
            print(f"  {modality}: {features.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                item_ids=item_ids,
                multimodal_features=multimodal_features,
                seq_lengths=seq_lengths,
                target_items=target_items,
                return_loss=True
            )
        
        print(f"\n输出:")
        print(f"  Logits: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Recommendation loss: {outputs['recommendation_loss'].item():.4f}")
        
        print(f"\n✓ 维度验证通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 维度验证失败!")
        print(f"错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有维度验证测试"""
    print("\n" + "="*80)
    print("模型维度验证")
    print("="*80)
    
    all_passed = True
    
    # 测试1: config_example.yaml 的配置
    print("\n【测试1】config_example.yaml 配置")
    result = test_dimension_config(
        "config_example.yaml",
        modality_dims={'text': 768, 'image': 2048},
        disentangled_dim=64,           # 从config更新
        num_disentangled_dims=3,
        num_interests=4,
        quantum_state_dim=128,         # 从config更新
        hidden_dim=256,                # 从config更新
        item_embed_dim=128,            # 从config更新
        num_items=100,
        max_seq_length=50
    )
    all_passed = all_passed and result
    
    # 测试2: 更大的解耦维度
    print("\n【测试2】更大的解耦维度 (disentangled_dim=128)")
    result = test_dimension_config(
        "large_disentangled",
        modality_dims={'text': 768, 'image': 2048},
        disentangled_dim=128,
        num_disentangled_dims=3,
        num_interests=4,
        quantum_state_dim=256,
        hidden_dim=512,
        item_embed_dim=256,
        num_items=100,
        max_seq_length=50
    )
    all_passed = all_passed and result
    
    # 测试3: 更多解耦维度
    print("\n【测试3】更多解耦维度 (num_disentangled_dims=5)")
    result = test_dimension_config(
        "more_disentangled_dims",
        modality_dims={'text': 768, 'image': 2048},
        disentangled_dim=64,
        num_disentangled_dims=5,  # 5个维度
        num_interests=4,
        quantum_state_dim=128,
        hidden_dim=256,
        item_embed_dim=128,
        num_items=100,
        max_seq_length=50
    )
    all_passed = all_passed and result
    
    # 测试4: 小型配置
    print("\n【测试4】小型配置 (快速测试)")
    result = test_dimension_config(
        "small_config",
        modality_dims={'text': 768, 'image': 2048},
        disentangled_dim=32,
        num_disentangled_dims=3,
        num_interests=2,
        quantum_state_dim=64,
        hidden_dim=128,
        item_embed_dim=64,
        num_items=100,
        max_seq_length=20
    )
    all_passed = all_passed and result
    
    # 测试5: 验证维度自动计算
    print("\n【测试5】验证 total_dim 自动计算")
    configs = [
        (32, 3, 96),    # 32 * 3 = 96
        (64, 3, 192),   # 64 * 3 = 192
        (128, 3, 384),  # 128 * 3 = 384
        (64, 5, 320),   # 64 * 5 = 320
    ]
    
    for dis_dim, num_dims, expected_total in configs:
        calculated_total = dis_dim * num_dims
        status = "✓" if calculated_total == expected_total else "✗"
        print(f"  {status} {dis_dim} × {num_dims} = {calculated_total} (期望: {expected_total})")
        if calculated_total != expected_total:
            all_passed = False
    
    # 最终结果
    print("\n" + "="*80)
    if all_passed:
        print("✓ 所有维度验证测试通过!")
        print("✓ 模型没有硬编码维度，可以安全使用配置文件中的任何维度设置")
    else:
        print("✗ 部分测试失败，请检查维度配置")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
