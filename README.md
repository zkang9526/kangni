# 康尼智驾三轮车 - 智能感知系统

智能感知项目代码、配置、数据集与项目资料。包含 DMS（驾驶员监测）和 FCW（前向碰撞预警）完整的训练与评估流程。

## 目录结构

```text
.
├─ configs/                    训练配置
│  ├─ fcw_unified.yaml        FCW YOLO 配置
│  └─ fcw_classes.txt         FCW 类别定义
├─ data/                       数据集（不纳入 Git，约 5.93 GiB）
│  ├─ fcw/                    BDD100K + nuScenes Mini
│  └─ dms/                    MRL Eye + NTHU + SVIRO
├─ docs/
│  ├─ reference/              项目需求、方案和硬件手册
│  ├─ datasets.md             数据集说明
│  ├─ dataset-status.md       数据集能力边界
│  └─ training.md             训练指南
├─ manifests/                  数据集清单（相对路径）
├─ scripts/                    训练、评估和验证脚本
│  ├─ train_fcw.py            FCW 训练入口
│  ├─ train_dms_eye.py        DMS 训练入口
│  ├─ evaluate_models.py      模型评估工具
│  ├─ visualize_predictions.py 预测可视化
│  └─ verify_datasets.py      数据集完整性检查
├─ runs/                       训练输出（不纳入 Git）
├─ TRAINING_GUIDE.md           训练详细指南
├─ EVALUATION_GUIDE.md         评估方法指南
├─ EVALUATION_RESULTS.md       最新评估结果
├─ GITHUB_PUSH_GUIDE.md        GitHub 推送指南
└─ requirements.txt            Python 依赖
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/zkang9526/kangni.git
cd kangni
```

### 2. 获取数据集

数据集约 5.93 GiB，不上传到 Git。通过移动硬盘、局域网或云存储获取：

```bash
# 将 data/ 目录复制到项目根目录
# 或者如果在另一台设备上已有完整数据集：
cp -r /path/to/datasets_v3.0/data/ ./data/
```

### 3. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 验证数据集
python scripts/verify_datasets.py
```

### 4. 开始训练

```bash
# 训练 FCW
python scripts/train_fcw.py --model yolo11n.pt --epochs 50 --batch 8 --device 0

# 训练 DMS
python scripts/train_dms_eye.py --epochs 20 --batch-size 64 --device auto
```

详细训练参数见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### 5. 模型评估

```bash
# 完整评估
python scripts/evaluate_models.py

# 包含速度测试
python scripts/evaluate_models.py --speed-test

# 可视化预测
python scripts/visualize_predictions.py --num-samples 50
```

详细评估方法见 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

## 最新评估结果

### DMS 眼睛状态检测 ✓
- **准确率**: 96.88%
- **闭眼召回率**: 93.66%
- **推理速度**: ~3,800 FPS
- **状态**: 可部署

### FCW 前向碰撞预警 ⚠
- **mAP50**: 39.19%
- **召回率**: 35.54%
- **精确率**: 72.69%
- **推理速度**: ~1,666 FPS
- **状态**: 需改进（召回率偏低）

详细结果和改进建议见 [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)

## 数据集说明

### FCW 数据
- **BDD100K**: 69,863 训练图片 + 10,081 验证图片
- **nuScenes Mini**: 404 帧带深度信息
- **类别**: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light

### DMS 数据
- **MRL Eye**: 84,898 眼睛状态图片（闭眼/睁眼）
- **NTHU**: 疲劳驾驶二分类
- **SVIRO**: 后排乘员占用检测

更多信息见 [docs/datasets.md](docs/datasets.md) 和 [docs/dataset-status.md](docs/dataset-status.md)

## 协作约定

- 开始工作前运行 `git pull --rebase`
- 提交前运行 `git status` 检查暂存区
- 不提交 `.venv/`、`runs/`、`data/`、`*.pt` 权重文件
- 数据集同步使用 Git 之外的渠道（移动硬盘、网盘等）
- 保持相同的相对目录结构，脚本可跨设备复用

## 文档索引

- [训练指南](TRAINING_GUIDE.md) - 详细训练命令和参数
- [评估指南](EVALUATION_GUIDE.md) - 模型评估方法
- [评估结果](EVALUATION_RESULTS.md) - 最新测试结果和改进建议
- [GitHub 指南](GITHUB_PUSH_GUIDE.md) - SSH 配置和推送流程
- [数据集文档](docs/datasets.md) - 数据来源和组织
- [参考资料](docs/reference/) - 项目需求和硬件手册

## 系统要求

- Python 3.8+
- CUDA 11.7+ (GPU训练)
- 16GB+ RAM
- 100GB+ 磁盘空间（包含数据集）

推荐配置：
- GPU: NVIDIA RTX 3090 / 4090
- CPU: 8核+
- RAM: 32GB+

## 许可与引用

本项目使用的数据集遵循各自的许可协议：
- BDD100K: [Berkeley DeepDrive License](https://bdd-data.berkeley.edu/)
- nuScenes: [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use)
- MRL Eye Dataset: [Academic Use](http://mrl.cs.vsb.cz/eyedataset)

---

**项目状态**: 活跃开发中 | **最后更新**: 2026-09-03
