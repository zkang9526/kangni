#!/usr/bin/env python3
"""
模型评估脚本 - 检验 DMS 和 FCW 训练结果
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

# 项目根目录
ROOT = Path(__file__).parent.parent.resolve()


def load_dms_model(checkpoint_path):
    """加载 DMS 眼睛状态检测模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = mobilenet_v2(num_classes=2)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ DMS模型加载成功: {checkpoint_path}")
    print(f"  训练轮次: {checkpoint['epoch']}")
    print(f"  验证准确率: {checkpoint.get('accuracy', 'N/A')}")

    return model, device


def evaluate_dms(model_path, test_dir):
    """评估 DMS 模型"""
    print("\n" + "="*60)
    print("DMS 眼睛状态检测模型评估")
    print("="*60)

    model, device = load_dms_model(model_path)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 类别映射
    classes = ['closed', 'open']

    # 统计结果
    total = 0
    correct = 0
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}

    # 遍历测试集
    test_path = ROOT / test_dir
    if not test_path.exists():
        print(f"✗ 测试目录不存在: {test_path}")
        return

    print(f"\n正在评估测试集: {test_path}")

    for class_name in classes:
        class_dir = test_path / class_name
        if not class_dir.exists():
            print(f"  跳过不存在的类别: {class_name}")
            continue

        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        print(f"  {class_name}: {len(image_files)} 张图片")

        for img_path in tqdm(image_files, desc=f"  处理 {class_name}"):
            try:
                # 加载并预处理图像
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # 推理
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted = torch.max(outputs, 1)

                # 统计
                true_label = classes.index(class_name)
                pred_label = predicted.item()

                total += 1
                class_total[class_name] += 1

                if pred_label == true_label:
                    correct += 1
                    class_correct[class_name] += 1

            except Exception as e:
                print(f"    处理失败 {img_path.name}: {e}")

    # 打印结果
    print(f"\n{'='*60}")
    print("DMS 评估结果:")
    print(f"{'='*60}")
    print(f"总体准确率: {100 * correct / total:.2f}% ({correct}/{total})")
    print(f"\n各类别表现:")
    for class_name in classes:
        if class_total[class_name] > 0:
            acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"  {class_name:8s}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")

    # 计算召回率和精确率（对于闭眼检测）
    if class_total['closed'] > 0:
        recall = 100 * class_correct['closed'] / class_total['closed']
        print(f"\n闭眼召回率 (Closed Recall): {recall:.2f}%")
        print("  (这是疲劳检测的关键指标 - 不能漏检闭眼)")


def evaluate_fcw(model_path, data_yaml):
    """评估 FCW 模型"""
    print("\n" + "="*60)
    print("FCW 前向碰撞预警模型评估")
    print("="*60)

    # 加载 YOLO 模型
    model = YOLO(model_path)
    print(f"✓ FCW模型加载成功: {model_path}")

    # 在验证集上评估
    print(f"\n正在验证集上评估...")
    results = model.val(data=str(data_yaml), split='val', verbose=True)

    # 打印详细结果
    print(f"\n{'='*60}")
    print("FCW 评估结果:")
    print(f"{'='*60}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"精确率:    {results.box.mp:.4f}")
    print(f"召回率:    {results.box.mr:.4f}")

    # 各类别 mAP
    print(f"\n各类别 mAP50:")
    class_names = ['pedestrian', 'rider', 'car', 'truck', 'bus',
                   'train', 'motorcycle', 'bicycle', 'traffic light']

    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        for i, name in enumerate(class_names):
            if i < len(results.box.ap50):
                print(f"  {name:15s}: {results.box.ap50[i]:.4f}")

    # 保存混淆矩阵和预测结果
    print(f"\n详细结果已保存到: {results.save_dir}")

    return results


def test_inference_speed(dms_model_path, fcw_model_path):
    """测试推理速度"""
    print("\n" + "="*60)
    print("推理速度测试")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # DMS 速度测试
    print("\n--- DMS 模型 ---")
    dms_model, _ = load_dms_model(dms_model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 创建随机输入
    dummy_img = Image.new('RGB', (640, 480))
    img_tensor = transform(dummy_img).unsqueeze(0).to(device)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = dms_model(img_tensor)

    # 测速
    import time
    n_iterations = 100
    start = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = dms_model(img_tensor)
    end = time.time()

    dms_fps = n_iterations / (end - start)
    dms_latency = 1000 * (end - start) / n_iterations
    print(f"推理速度: {dms_fps:.1f} FPS")
    print(f"延迟: {dms_latency:.2f} ms")

    # FCW 速度测试
    print("\n--- FCW 模型 ---")
    fcw_model = YOLO(fcw_model_path)

    # 创建随机输入
    dummy_img_path = ROOT / 'temp_test.jpg'
    dummy_img.save(dummy_img_path)

    # 预热
    for _ in range(10):
        _ = fcw_model(dummy_img_path, verbose=False)

    # 测速
    start = time.time()
    for _ in range(n_iterations):
        _ = fcw_model(dummy_img_path, verbose=False)
    end = time.time()

    fcw_fps = n_iterations / (end - start)
    fcw_latency = 1000 * (end - start) / n_iterations
    print(f"推理速度: {fcw_fps:.1f} FPS")
    print(f"延迟: {fcw_latency:.2f} ms")

    # 清理
    dummy_img_path.unlink()

    print(f"\n实时性评估 (目标: ≥30 FPS):")
    print(f"  DMS: {'✓ 满足' if dms_fps >= 30 else '✗ 不满足'}")
    print(f"  FCW: {'✓ 满足' if fcw_fps >= 30 else '✗ 不满足'}")


def main():
    parser = argparse.ArgumentParser(description='评估 DMS 和 FCW 模型')
    parser.add_argument('--dms-model', type=str,
                       default='runs/dms_eye_mobilenetv2/best.pt',
                       help='DMS 模型路径')
    parser.add_argument('--fcw-model', type=str,
                       default='runs/fcw/fcw_baseline/weights/best.pt',
                       help='FCW 模型路径')
    parser.add_argument('--dms-test-dir', type=str,
                       default='data/dms/eye_state/images/test',
                       help='DMS 测试集目录')
    parser.add_argument('--fcw-config', type=str,
                       default='configs/fcw_unified.yaml',
                       help='FCW 数据配置文件')
    parser.add_argument('--skip-dms', action='store_true',
                       help='跳过 DMS 评估')
    parser.add_argument('--skip-fcw', action='store_true',
                       help='跳过 FCW 评估')
    parser.add_argument('--speed-test', action='store_true',
                       help='测试推理速度')

    args = parser.parse_args()

    # 转换为绝对路径
    dms_model_path = ROOT / args.dms_model
    fcw_model_path = ROOT / args.fcw_model
    dms_test_dir = args.dms_test_dir
    fcw_config = ROOT / args.fcw_config

    # 检查文件
    if not args.skip_dms and not dms_model_path.exists():
        print(f"✗ DMS 模型不存在: {dms_model_path}")
        return 1

    if not args.skip_fcw and not fcw_model_path.exists():
        print(f"✗ FCW 模型不存在: {fcw_model_path}")
        return 1

    # 评估 DMS
    if not args.skip_dms:
        evaluate_dms(dms_model_path, dms_test_dir)

    # 评估 FCW
    if not args.skip_fcw:
        evaluate_fcw(fcw_model_path, fcw_config)

    # 速度测试
    if args.speed_test:
        test_inference_speed(dms_model_path, fcw_model_path)

    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
