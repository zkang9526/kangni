# 感知数据集 V3.0

V3.0 是可跨设备使用的版本化基线数据集。图片和标签均为真实文件，不依赖原设备盘符、目录联接或硬链接；`datasets/v3.0/metadata.json` 记录数量和能力边界。

数据集约 5.93 GiB，不纳入 Git 仓库。更换设备时应单独复制整个 `datasets/v3.0/` 目录；代码、配置和文档仍通过 GitHub 同步。

## 数据内容

- FCW：BDD100K 训练/验证图片与 YOLO 标签；nuScenes Mini 404 帧、YOLO 标签及米制距离清单。
- DMS：MRL Eye 闭眼/睁眼、NTHU 二分类弱标签、SVIRO 后排占用分类。
- `manifests/`：使用数据集根目录相对路径的数据清单。

有意排除原始压缩包、重复 raw 目录、尚未完成逐帧标注的 YawDD 视频，以及合成几何安全带流程测试集。

## 目录

```text
datasets/v3.0/
├─ fcw/
│  ├─ images/{bdd100k,nuscenes_mini}/
│  └─ labels/{bdd100k,nuscenes_mini}/
├─ dms/
│  ├─ eye_state/images/{train,val,test}/{closed,open}/
│  ├─ drowsiness_binary/images/
│  └─ rear_occupancy_sviro/images/
├─ manifests/
└─ metadata.json
```

## 完整性检查

从仓库根目录运行：

```bash
python scripts/verify_datasets.py
```

检查输出为 `PASS` 后再开始训练。BDD100K、nuScenes、MRL 与 SVIRO 数据仍受各自原始许可条款约束。
