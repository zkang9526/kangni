#!/usr/bin/env python3
"""
可视化预测结果 - 直观查看模型表现
"""
import os
import sys
import random
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from ultralytics import YOLO
import argparse


ROOT = Path(__file__).parent.parent.resolve()


def visualize_dms_predictions(model_path, test_dir, output_dir, num_samples=20):
    """可视化 DMS 预测结果"""
    print("\n" + "="*60)
    print("DMS 预测可视化")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = mobilenet_v2(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    classes = ['closed', 'open']
    output_path = ROOT / output_dir / 'dms_predictions'
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集样本
    samples = []
    test_path = ROOT / test_dir
    for class_name in classes:
        class_dir = test_path / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            samples.extend([(img, class_name) for img in images])

    # 随机选择样本
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    print(f"正在处理 {len(samples)} 个样本...")

    correct = 0
    for i, (img_path, true_label) in enumerate(samples):
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)

        pred_label = classes[predicted.item()]
        confidence = probs[predicted.item()].item()

        # 绘制结果
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        # 添加文本
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        color = 'green' if is_correct else 'red'
        text = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"

        # 绘制背景框
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 使用 textbbox 替代 textsize
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle(bbox, fill='black')
        draw.text((10, 10), text, fill=color, font=font)

        # 保存
        save_path = output_path / f"{i:03d}_{true_label}_{pred_label}_{confidence:.2f}.jpg"
        draw_img.save(save_path)

    accuracy = 100 * correct / len(samples)
    print(f"✓ 可视化完成: {output_path}")
    print(f"  准确率: {accuracy:.2f}% ({correct}/{len(samples)})")


def visualize_fcw_predictions(model_path, data_dir, output_dir, num_samples=20):
    """可视化 FCW 预测结果"""
    print("\n" + "="*60)
    print("FCW 预测可视化")
    print("="*60)

    # 加载模型
    model = YOLO(model_path)

    output_path = ROOT / output_dir / 'fcw_predictions'
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集验证集图像
    val_images_dir = ROOT / data_dir / 'images' / 'bdd100k' / 'val'
    if not val_images_dir.exists():
        print(f"✗ 验证集目录不存在: {val_images_dir}")
        return

    images = list(val_images_dir.glob('*.jpg'))
    if len(images) > num_samples:
        images = random.sample(images, num_samples)

    print(f"正在处理 {len(images)} 个样本...")

    for i, img_path in enumerate(images):
        # 推理
        results = model(img_path, verbose=False)

        # 保存可视化结果
        result_img = results[0].plot()
        save_path = output_path / f"{i:03d}_{img_path.name}"

        # PIL保存
        Image.fromarray(result_img).save(save_path)

    print(f"✓ 可视化完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化模型预测结果')
    parser.add_argument('--dms-model', type=str,
                       default='runs/dms_eye_mobilenetv2/best.pt',
                       help='DMS 模型路径')
    parser.add_argument('--fcw-model', type=str,
                       default='runs/fcw/fcw_baseline/weights/best.pt',
                       help='FCW 模型路径')
    parser.add_argument('--dms-test-dir', type=str,
                       default='data/dms/eye_state/images/test',
                       help='DMS 测试集目录')
    parser.add_argument('--fcw-data-dir', type=str,
                       default='data/fcw',
                       help='FCW 数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='evaluation_results',
                       help='输出目录')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='可视化样本数量')
    parser.add_argument('--dms-only', action='store_true',
                       help='只可视化 DMS')
    parser.add_argument('--fcw-only', action='store_true',
                       help='只可视化 FCW')

    args = parser.parse_args()

    # 可视化 DMS
    if not args.fcw_only:
        dms_model_path = ROOT / args.dms_model
        if dms_model_path.exists():
            visualize_dms_predictions(
                dms_model_path,
                args.dms_test_dir,
                args.output_dir,
                args.num_samples
            )
        else:
            print(f"✗ DMS 模型不存在: {dms_model_path}")

    # 可视化 FCW
    if not args.dms_only:
        fcw_model_path = ROOT / args.fcw_model
        if fcw_model_path.exists():
            visualize_fcw_predictions(
                fcw_model_path,
                args.fcw_data_dir,
                args.output_dir,
                args.num_samples
            )
        else:
            print(f"✗ FCW 模型不存在: {fcw_model_path}")

    print(f"\n{'='*60}")
    print("可视化完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    sys.exit(main())
