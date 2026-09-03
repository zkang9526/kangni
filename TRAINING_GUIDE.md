# 训练操作指南

## 1. 安装环境

```bash
python -m venv .venv
source .venv/bin/activate              # Linux/macOS
# .venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
python scripts/verify_package.py
```

`requirements.txt` 只包含训练脚本需要的 PyTorch、torchvision、Ultralytics、Pillow、NumPy。GPU 机器建议先按照 PyTorch 官网命令安装匹配 CUDA 的版本。

## 2. FCW 训练

```bash
python scripts/train_fcw.py --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

重要参数：

- `--fraction 0.1`：先用 10% 数据做冒烟测试。
- `--device cpu`：无 GPU 时使用；建议 `--batch 2`。
- `--workers 0`：Windows 或内存紧张时使用。
- `--name fcw_baseline`：修改实验名。

输出：`runs/fcw/<name>/weights/best.pt`、训练曲线和验证结果。

类别顺序见 `configs/fcw_classes.txt`，共 9 类：person、rider、car、truck、bus、train、motor、bike、obstacle。

## 3. MRL Eye 闭眼/睁眼训练

```bash
python scripts/train_dms_eye.py --epochs 20 --batch-size 64 --image-size 224 --device auto
```

先用 MRL Eye 训练眼睛状态分类器，再将其用于项目驾驶员 ROI。MRL 是单眼裁剪图，不能直接证明驾驶员双眼闭合超过 2 秒；部署时必须增加连续帧逻辑。

## 4. 训练前检查清单

1. `python scripts/verify_package.py` 返回 `PASS`。
2. FCW train/val 标签与图片同名且数量符合报告。
3. 验证集不参与反复调参后的最终报告。
4. MRL/NTHU 按主体隔离，不要随机打散相邻帧。
5. 项目真车数据另按 `manual_annotation_templates` 的字段制作，不能把合成 SVIRO 当真实精度证据。
