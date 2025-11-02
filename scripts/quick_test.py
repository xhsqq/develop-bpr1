"""
Quick test script to verify the data pipeline works
快速测试脚本，验证数据处理流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.dataset import AmazonDataset, collate_fn
from torch.utils.data import DataLoader


def test_dataset_loading(category='beauty'):
    """测试数据集加载"""
    print(f"\n{'='*80}")
    print(f"Testing dataset loading for {category}")
    print('='*80)

    try:
        # 创建数据集
        train_dataset = AmazonDataset(
            category=category,
            split='train',
            data_dir='data/processed',
            max_seq_length=50,
            use_text_features=False
        )

        print(f"✓ Loaded train dataset: {len(train_dataset)} samples")

        # 测试获取一个样本
        sample = train_dataset[0]
        print(f"\nSample structure:")
        print(f"  user_id: {sample['user_id']}")
        print(f"  item_ids shape: {sample['item_ids'].shape}")
        print(f"  target_item: {sample['target_item']}")
        print(f"  seq_length: {sample['seq_length']}")
        print(f"  multimodal_features keys: {list(sample['multimodal_features'].keys())}")
        for key, feat in sample['multimodal_features'].items():
            print(f"    {key} shape: {feat.shape}")

        # 测试DataLoader
        print(f"\nTesting DataLoader...")
        dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        batch = next(iter(dataloader))
        print(f"✓ Batch structure:")
        print(f"  item_ids shape: {batch['item_ids'].shape}")
        print(f"  target_items shape: {batch['target_items'].shape}")
        print(f"  seq_lengths shape: {batch['seq_lengths'].shape}")

        for key, feat in batch['multimodal_features'].items():
            print(f"  multimodal/{key} shape: {feat.shape}")

        print(f"\n✓ Dataset loading test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(category='beauty'):
    """测试模型前向传播"""
    print(f"\n{'='*80}")
    print(f"Testing model forward pass")
    print('='*80)

    try:
        from models.multimodal_recommender import MultimodalRecommender

        # 创建数据集
        dataset = AmazonDataset(
            category=category,
            split='train',
            data_dir='data/processed',
            max_seq_length=50,
            use_text_features=False
        )

        num_items = dataset.num_items

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        batch = next(iter(dataloader))

        # 创建模型
        model = MultimodalRecommender(
            modality_dims={'text': 768, 'image': 2048},  # 真实BERT和ResNet特征维度
            disentangled_dim=32,
            num_disentangled_dims=3,
            num_interests=4,
            quantum_state_dim=64,
            hidden_dim=128,
            item_embed_dim=64,
            num_items=num_items,
            max_seq_length=50,
            use_quantum_computing=False
        )

        print(f"✓ Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(
                item_ids=batch['item_ids'],
                multimodal_features=batch['multimodal_features'],
                seq_lengths=batch['seq_lengths'],
                target_items=batch['target_items'],
                return_loss=True,
                return_explanations=True
            )

        print(f"\n✓ Forward pass successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        print(f"  Recommendation logits shape: {outputs['recommendation_logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")

        # 测试预测
        top_k_items, top_k_scores = model.predict(
            batch['item_ids'],
            batch['multimodal_features'],
            batch['seq_lengths'],
            top_k=10
        )

        print(f"\n✓ Prediction successful!")
        print(f"  Top-10 items shape: {top_k_items.shape}")
        print(f"  Top-10 scores shape: {top_k_scores.shape}")

        print(f"\n✓ Model forward test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Model forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation(category='beauty'):
    """测试评估流程"""
    print(f"\n{'='*80}")
    print(f"Testing evaluation")
    print('='*80)

    try:
        from models.multimodal_recommender import MultimodalRecommender
        from utils.evaluation import FullLibraryEvaluator

        # 创建数据集
        dataset = AmazonDataset(
            category=category,
            split='valid',
            data_dir='data/processed',
            max_seq_length=50,
            use_text_features=False
        )

        num_items = dataset.num_items

        # 创建DataLoader（小批量用于快速测试）
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        # 只取前2个batch测试
        limited_batches = []
        for i, batch in enumerate(dataloader):
            limited_batches.append(batch)
            if i >= 1:
                break

        # 创建模型
        model = MultimodalRecommender(
            modality_dims={'text': 768, 'image': 2048},  # 真实BERT和ResNet特征维度
            disentangled_dim=32,
            num_disentangled_dims=3,
            num_interests=4,
            quantum_state_dim=64,
            hidden_dim=128,
            item_embed_dim=64,
            num_items=num_items,
            max_seq_length=50,
            use_quantum_computing=False
        )

        print(f"✓ Created model and dataset")

        # 创建评估器
        evaluator = FullLibraryEvaluator(
            num_items=num_items,
            k_list=[5, 10, 20]
        )

        # 评估
        class MockDataLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

        mock_loader = MockDataLoader(limited_batches)

        metrics = evaluator.evaluate(
            model,
            mock_loader,
            device='cpu'
        )

        print(f"\n✓ Evaluation successful!")
        print(f"  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")

        print(f"\n✓ Evaluation test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Running Quick Tests")
    print("="*80)

    # 检查数据是否存在
    category = 'beauty'
    data_path = f'data/processed/{category}/train_sequences.pkl'

    if not os.path.exists(data_path):
        print(f"\n✗ Processed data not found at {data_path}")
        print(f"\nPlease run:")
        print(f"  1. python data/download_amazon.py --category {category}")
        print(f"  2. python data/preprocess_amazon.py --category {category}")
        return

    # 运行测试
    results = []

    print("\nTest 1: Dataset Loading")
    results.append(("Dataset Loading", test_dataset_loading(category)))

    print("\nTest 2: Model Forward Pass")
    results.append(("Model Forward", test_model_forward(category)))

    print("\nTest 3: Evaluation")
    results.append(("Evaluation", test_evaluation(category)))

    # 总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed! You can now run training.")
        print(f"\nTo train: python train_amazon.py --category {category}")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
