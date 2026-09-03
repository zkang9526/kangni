# 训练操作指南

以下命令均从仓库根目录运行。

## 1. 安装环境

```bash
python -m venv .venv
source .venv/bin/activate              # Linux/macOS
# .venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_datasets.py
```

数据集不会随 Git 仓库下载。首次配置新设备时，先把 `datasets/v3.0/` 从原设备或独立存储完整复制到仓库根目录。

GPU 机器建议先按目标 CUDA 版本安装对应 PyTorch，再安装其余依赖。

## 2. FCW 训练

```bash
python scripts/train_fcw.py --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

常用参数：`--fraction 0.1` 用于小规模冒烟测试；无 GPU 时使用 `--device cpu --batch 2`；Windows 或内存紧张时使用 `--workers 0`。输出位于 `runs/fcw/`。

类别顺序见 `configs/fcw_classes.txt`，共 9 类：person、rider、car、truck、bus、train、motor、bike、obstacle。

## 3. MRL Eye 闭眼/睁眼训练

```bash
python scripts/train_dms_eye.py --epochs 20 --batch-size 64 --image-size 224 --device auto
```

输出位于 `runs/dms_eye_mobilenetv2/`。MRL 是单眼裁剪图，不能直接证明驾驶员双眼闭合超过 2 秒；部署时必须增加连续帧逻辑。

## 4. 训练前检查

1. `python scripts/verify_datasets.py` 返回 `PASS`。
2. FCW 训练集和验证集中的图片、标签对应。
3. 验证集不参与最终报告前的反复调参。
4. MRL/NTHU 按主体隔离，不随机打散相邻帧。
5. 项目真车数据应独立标注，不能把合成 SVIRO 当作真实精度证据。
