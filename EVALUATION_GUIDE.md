# 模型评估指南

本指南介绍如何全面评估已训练的 DMS 和 FCW 模型。

## 当前训练结果摘要

### DMS 眼睛状态检测
- **模型**: MobileNetV2
- **最佳准确率**: 98.92% (epoch 4)
- **闭眼召回率**: 99.26%（关键指标）
- **推理速度**: ~0.26ms/样本
- **权重**: `runs/dms_eye_mobilenetv2/best.pt`

### FCW 前向碰撞预警
- **模型**: YOLO11n
- **mAP50**: 35.5%
- **mAP50-95**: 20.9%
- **精确率**: 69%
- **召回率**: 32%
- **权重**: `runs/fcw/fcw_baseline/weights/best.pt`

---

## 评估方式

### 1. 快速评估（推荐先执行）

```bash
# 评估两个模型
python scripts/evaluate_models.py

# 只评估 DMS
python scripts/evaluate_models.py --skip-fcw

# 只评估 FCW
python scripts/evaluate_models.py --skip-dms

# 包含推理速度测试
python scripts/evaluate_models.py --speed-test
```

**输出内容**：
- 测试集准确率、精确率、召回率
- 各类别性能
- mAP 指标（FCW）
- 推理速度（可选）

---

### 2. 可视化预测结果

```bash
# 生成预测可视化
python scripts/visualize_predictions.py

# 自定义样本数量
python scripts/visualize_predictions.py --num-samples 50

# 只可视化 DMS
python scripts/visualize_predictions.py --dms-only

# 只可视化 FCW
python scripts/visualize_predictions.py --fcw-only

# 指定输出目录
python scripts/visualize_predictions.py --output-dir my_results
```

**输出位置**：
- DMS: `evaluation_results/dms_predictions/`
- FCW: `evaluation_results/fcw_predictions/`

图片命名示例：`001_closed_closed_0.98.jpg`（序号_真实标签_预测标签_置信度）

---

### 3. 查看训练历史

#### DMS 训练指标
```bash
# 查看完整训练历史
cat runs/dms_eye_mobilenetv2/metrics.json | python -m json.tool

# 快速查看最佳结果
python -c "
import json
with open('runs/dms_eye_mobilenetv2/metrics.json') as f:
    data = json.load(f)
    best = max(data['history'], key=lambda x: x['accuracy'])
    print(f\"最佳 epoch: {best['epoch']}\")
    print(f\"准确率: {best['accuracy']:.4f}\")
    print(f\"闭眼召回率: {best['closed_recall']:.4f}\")
    print(f\"宏平均 F1: {best['macro_f1']:.4f}\")
"
```

#### FCW 训练曲线
```bash
# 查看训练结果
cat runs/fcw/fcw_baseline/results.csv | head -n 30 | column -t -s,
```

**可视化文件**（已自动生成）：
- `runs/fcw/fcw_baseline/results.png` - 训练曲线
- `runs/fcw/fcw_baseline/confusion_matrix.png` - 混淆矩阵
- `runs/fcw/fcw_baseline/BoxPR_curve.png` - PR 曲线
- `runs/fcw/fcw_baseline/BoxF1_curve.png` - F1 曲线

---

## 关键评估指标

### DMS（驾驶员监测）

| 指标 | 目标值 | 实际值 | 说明 |
|------|--------|--------|------|
| **闭眼召回率** | ≥95% | 99.26% | ✓ 不能漏检闭眼 |
| **总体准确率** | ≥90% | 98.92% | ✓ 整体表现 |
| **推理速度** | ≥30 FPS | ~3800 FPS | ✓ 实时性 |

**关键点**：
- 闭眼召回率是最重要指标，宁可误报（假阳性）也不能漏检（假阴性）
- 当前模型表现优秀，已满足实时疲劳检测需求

### FCW（前向碰撞预警）

| 指标 | 目标值 | 实际值 | 说明 |
|------|--------|--------|------|
| **mAP50** | ≥50% | 35.5% | ⚠ 需改进 |
| **召回率** | ≥50% | 32% | ⚠ 漏检较多 |
| **精确率** | ≥60% | 69% | ✓ 误报可接受 |
| **推理速度** | ≥30 FPS | ~待测 | 待测试 |

**关键点**：
- 召回率偏低，意味着漏检风险较高
- 精确率尚可，但召回率是安全系统的首要指标
- 建议继续训练或调优参数

---

## 改进建议

### DMS 模型
✓ **当前状态良好**，可直接部署
- 考虑添加数据增强以提高鲁棒性
- 测试不同光照条件下的表现
- 可尝试更轻量的模型（MobileNetV3, EfficientNet）

### FCW 模型
⚠ **需要改进**

**短期改进**：
1. 继续训练到收敛（当前50轮可能不够）
2. 调整置信度阈值（牺牲精确率提升召回率）
3. 使用更大的模型（YOLO11s 或 YOLO11m）
4. 增加训练数据

**训练命令示例**：
```bash
# 继续训练 50 轮
python scripts/train_fcw.py --model runs/fcw/fcw_baseline/weights/last.pt --epochs 100 --batch 8

# 使用更大模型
python scripts/train_fcw.py --model yolo11s.pt --epochs 100 --batch 6
```

**推理时调整阈值**：
```python
# 降低置信度阈值以提高召回率
results = model.predict(image, conf=0.25, iou=0.45)
```

---

## 完整评估流程

```bash
# 1. 快速评估性能
python scripts/evaluate_models.py --speed-test

# 2. 生成可视化结果
python scripts/visualize_predictions.py --num-samples 50

# 3. 查看训练曲线
# DMS
cat runs/dms_eye_mobilenetv2/metrics.json | python -m json.tool | less

# FCW
cat runs/fcw/fcw_baseline/results.csv | column -t -s,

# 4. 检查可视化文件
ls -lh runs/fcw/fcw_baseline/*.png
ls -lh evaluation_results/
```

---

## 部署前检查清单

### DMS 模型
- [x] 测试集准确率 ≥90%
- [x] 闭眼召回率 ≥95%
- [x] 推理速度 ≥30 FPS
- [ ] 不同光照条件测试
- [ ] 不同人种/年龄测试
- [ ] 边缘设备测试（如需要）

### FCW 模型
- [ ] mAP50 ≥50%（当前 35.5%）
- [ ] 召回率 ≥50%（当前 32%）
- [x] 精确率 ≥60%（当前 69%）
- [ ] 推理速度 ≥30 FPS
- [ ] 不同天气条件测试
- [ ] 夜间场景测试

---

## 常见问题

**Q: DMS 准确率很高，但实际使用效果不好？**
A: 可能是测试集不够多样化。建议：
- 收集真实场景数据
- 测试不同光照（强光、暗光、侧光）
- 测试不同角度和距离
- 测试戴眼镜、墨镜的情况

**Q: FCW 召回率太低怎么办？**
A: 几个方向：
1. 继续训练（loss 还在下降）
2. 使用更大模型（s/m/l）
3. 调低推理时的 conf 阈值
4. 增加难样本训练数据
5. 使用 nuScenes 的深度信息辅助训练

**Q: 如何在边缘设备上部署？**
A: 需要模型转换：
- DMS: 转 ONNX/TensorRT/OpenVINO
- FCW: YOLO 支持导出多种格式
```bash
# YOLO 导出
yolo export model=runs/fcw/fcw_baseline/weights/best.pt format=onnx
```

---

## 参考指标对比

### 学术基准
- **DMS**: SOTA 眼睛状态检测 >99%（当前 98.92%）
- **FCW**: BDD100K 目标检测 SOTA ~45-55% mAP（当前 35.5%）

### 工业部署
- **响应时间**: <100ms 端到端延迟
- **误报率**: 可容忍（有声音提醒）
- **漏报率**: 尽量避免（安全关键）

当前 DMS 模型已达到部署标准，FCW 需要继续优化。
