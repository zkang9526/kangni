# 感知数据集 V2.0 Portable

这是可直接复制到另一台设备的独立数据包。包内图片和标签均为真实文件，不依赖原设备的绝对路径、Windows 目录联接或硬链接。`PACKAGE_INFO.json` 记录了数量与完整性信息。

## 包含内容

- FCW：BDD100K 训练/验证图片和 YOLO 标签；nuScenes Mini 404 帧、YOLO 标签及米制距离表。
- DMS：MRL Eye 闭眼/睁眼、NTHU 二分类弱标签、SVIRO 后排占用分类。
- 清单、类别文件、训练配置和本说明。

有意排除：原始压缩包、重复 raw 目录、YawDD 未完成逐帧标注的视频、合成几何安全带流程测试集。这些内容不会影响当前基线训练，且避免转移后产生重复占用或误用标签。

由于图片总量约 5.93 GiB，当前目录本身就是最终可搬迁包；直接复制整个 `datasets_v3.0` 文件夹比在 Windows 上重新 gzip 更可靠。

## 解包后目录

```text
datasets_v3.0/
├─ data/fcw/images/{bdd100k,nuscenes_mini}/
├─ data/fcw/labels/{bdd100k,nuscenes_mini}/
├─ data/dms/eye_state/images/{train,val,test}/{closed,open}/
├─ data/dms/drowsiness_binary/images/...
├─ data/dms/rear_occupancy_sviro/images/...
├─ manifests/                         全部为包内相对路径
├─ configs/fcw_unified.yaml           FCW 9 类 YOLO 配置
├─ scripts/verify_package.py          跨设备完整性检查
├─ scripts/train_fcw.py               YOLO FCW 训练入口
├─ scripts/train_dms_eye.py           MRL Eye 分类训练入口
└─ TRAINING_GUIDE.md                  详细训练命令
```

## 另一台设备上的训练

1. 将整个 `datasets_v3.0` 文件夹复制到目标设备。
2. 在该文件夹根目录创建 Python 虚拟环境并安装依赖：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_package.py
```

3. 训练 FCW：

```bash
python scripts/train_fcw.py --model yolo11n.pt --epochs 50 --batch 8 --device 0
```

没有 GPU 时使用 `--device cpu`，并把 batch 调小。训练输出保存在 `runs/fcw/`。

4. 训练闭眼/睁眼基线：

```bash
python scripts/train_dms_eye.py --epochs 20 --batch-size 64 --device auto
```

训练输出保存在 `runs/dms_eye/`。闭眼持续 2 秒、低头和项目安全带/雷达模型不能仅靠本包完成，需导入项目真车标注。

## 跨设备注意事项

- 配置中的路径全部是相对路径，不要把当前盘符写入配置。
- Windows、Linux、macOS 均可使用；推荐 Python 3.10/3.11。
- 若使用 NVIDIA GPU，请按目标设备的 CUDA 版本安装对应 PyTorch，再安装其余依赖。
- BDD100K 和 nuScenes 许可、MRL/SVIRO 条款需按各自官方要求执行。
