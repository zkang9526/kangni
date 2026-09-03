# 康尼智驾三轮车

智能感知项目代码、配置、数据集与项目资料。仓库按“代码与数据分离”组织，`datasets/` 中不存放训练脚本或运行环境文件。

## 目录结构

```text
.
├─ configs/                 训练配置
├─ datasets/
│  └─ v3.0/                本地数据集（不纳入 Git）
├─ docs/
│  ├─ reference/           项目需求、方案和硬件手册
│  ├─ datasets.md          数据集说明
│  ├─ dataset-status.md    数据集能力边界
│  └─ training.md          训练指南
├─ scripts/                 校验与训练入口
└─ requirements.txt         Python 依赖
```

## 在新设备上开始工作

代码与文档通过 GitHub 协作，约 5.93 GiB 的数据集不上传到 Git。先克隆仓库，再通过移动硬盘、局域网或其他大文件存储把当前设备的 `datasets/v3.0/` 完整复制到新设备的同一位置：

```bash
git clone https://github.com/zkang9526/kangni.git
cd kangni
python scripts/verify_datasets.py
```

创建环境和启动训练的命令见 [训练指南](docs/training.md)。数据内容、来源与限制见 [数据集说明](docs/datasets.md)。

## 协作约定

- 开始工作前运行 `git pull --rebase`，提交前运行 `git status`。
- 不提交 `.venv/`、`runs/`、缓存、训练权重和本机绝对路径。
- `datasets/` 整体由 `.gitignore` 排除；数据同步必须使用 Git 之外的渠道。
- 两台设备保持相同的 `datasets/v3.0/` 相对目录，脚本即可复用同一套配置。
