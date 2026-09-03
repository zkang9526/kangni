# 如何推送到 GitHub

## 当前状态

✓ Git 仓库已初始化
✓ 代码已提交到本地仓库
✓ 远程仓库已配置：https://github.com/zkang9526/kangni.git
⚠ 需要身份认证才能推送

## 快速推送（三步）

### 步骤 1：配置 GitHub 认证

**选项 A - Personal Access Token（推荐新手）**

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 填写：
   - Note: `Smart Tricycle Project`
   - Expiration: 90 days（或自定义）
   - 勾选权限：**repo**（完整仓库访问）
4. 点击 "Generate token"
5. **立即复制** token（离开页面后无法再看到）

**选项 B - SSH Key（推荐长期使用）**

```bash
# 1. 生成 SSH key
ssh-keygen -t ed25519 -C "zkang9526@github.com"
# 按回车使用默认路径，可设置密码或留空

# 2. 显示公钥
cat ~/.ssh/id_ed25519.pub

# 3. 复制公钥内容，添加到 GitHub
# 访问 https://github.com/settings/keys
# 点击 "New SSH key"，粘贴公钥内容

# 4. 测试连接
ssh -T git@github.com

# 5. 修改为 SSH 远程地址
git remote set-url origin git@github.com:zkang9526/kangni.git
```

### 步骤 2：推送到 GitHub

**如果使用 Personal Access Token：**
```bash
git push -u origin main

# 会提示输入：
# Username: zkang9526
# Password: [粘贴你的 Personal Access Token，不是 GitHub 密码]
```

**如果使用 SSH：**
```bash
# 确保已修改远程地址为 SSH（见上面步骤 1）
git push -u origin main
```

### 步骤 3：验证

推送成功后，访问：
https://github.com/zkang9526/kangni

你应该能看到所有文件。

## 已推送的内容

- ✓ 训练脚本（FCW + DMS）
- ✓ 评估脚本
- ✓ 可视化脚本
- ✓ 完整文档（README、评估报告、使用指南）
- ✓ 配置文件（YOLO 配置、数据清单）
- ✓ .gitignore（自动排除大文件）

## 未推送的内容（已排除）

- ✗ 训练数据（data/）- 太大，不适合 git
- ✗ 训练权重（runs/weights/）- 太大，建议用 Git LFS 或云存储
- ✗ YOLO 预训练模型（yolo*.pt）- 太大

## 常见问题

**Q: 推送时要求输入密码，但我的 GitHub 密码不对？**
A: GitHub 已禁用密码认证，必须使用 Personal Access Token 或 SSH。

**Q: 提示 "Authentication failed"？**
A:
- 确认使用的是 Personal Access Token，不是 GitHub 密码
- 确认 Token 有 `repo` 权限
- Token 可能已过期，需要重新生成

**Q: 提示 "Permission denied (publickey)"？**
A:
- SSH key 没有正确配置
- 运行 `ssh -T git@github.com` 测试连接
- 确认公钥已添加到 GitHub

**Q: 如何上传训练好的模型？**
A: 模型文件太大（>100MB），不适合直接 push。建议：
1. 使用 Git LFS（Large File Storage）
2. 使用 GitHub Releases 功能上传
3. 使用云存储（阿里云OSS、百度网盘等）并在 README 提供链接

## Git LFS 上传大文件（可选）

如果想要上传模型文件：

```bash
# 1. 安装 Git LFS
# Ubuntu/Debian:
sudo apt-get install git-lfs
# 或从 https://git-lfs.github.com/ 下载

# 2. 初始化 LFS
git lfs install

# 3. 追踪大文件
git lfs track "*.pt"
git lfs track "runs/**/*.pt"

# 4. 提交 .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# 5. 添加并推送模型
git add runs/fcw/fcw_baseline/weights/best.pt
git add runs/dms_eye_mobilenetv2/best.pt
git commit -m "Add trained model weights"
git push
```

## 需要帮助？

如果遇到问题：
1. 查看详细错误信息
2. 检查网络连接
3. 确认 GitHub 账号权限
4. 参考 GitHub 官方文档：https://docs.github.com/cn

---

**提示**：首次推送完成后，后续更新只需：
```bash
git add .
git commit -m "更新说明"
git push
```
