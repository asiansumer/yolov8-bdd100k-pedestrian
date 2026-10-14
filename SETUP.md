# macOS 环境配置指南 (新手版)

适用: Apple 芯片 Mac (M1/M2/M3), 终端默认使用 zsh, 已经装好 Miniconda。
目标: 配好 conda 环境 `bdd` → 确认 MPS (Apple GPU) 可用 → 配好 Git 和 GitHub SSH → 把本仓库推送到 GitHub。

> 下面所有以 `$` 开头的内容不用输入 `$` 本身。

---

## 1. 让 conda 在终端里可用

如果你看到提示 `Run 'conda init' before 'conda activate'`, 说明 conda 还没有"接入"你的 zsh。

```bash
conda init zsh
```

然后**完全关闭终端窗口再重新打开** (或者运行 `source ~/.zshrc`)。命令行最前面出现 `(base)` 就说明成功了。

检查版本 (注意: 没有 `miniconda` 这个命令, 程序名叫 `conda`):

```bash
conda --version
```

如果提示 `conda: command not found`, 用完整路径执行一次初始化 (路径以你的安装位置为准, 默认是 `~/miniconda3`):

```bash
~/miniconda3/bin/conda init zsh
```

确认装的是 Apple 芯片 (arm64) 版本, **否则 MPS 用不了**:

```bash
python -c "import platform; print(platform.machine())"     # 必须输出 arm64; 如果是 x86_64, 需要重装 arm64 版 Miniconda
```

## 2. 创建项目环境

```bash
cd ~/你放代码的目录/yolov8-bdd100k-pedestrian
conda env create -f environment.yml     # 创建名为 bdd 的环境, 第一次会下载 PyTorch, 需要几分钟
conda activate bdd                      # 以后每次打开新终端都要先执行这一句
```

## 3. 验证 MPS (Apple GPU) 可用

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), torch.ones(3, device='mps') * 2)"
```

看到 `True` 和 `tensor([2., 2., 2.], device='mps:0')` 就说明 GPU 可用。再跑一下单元测试:

```bash
pytest -q
```

## 4. 配置 Git

```bash
git --version        # 如果弹窗让你安装 "命令行开发者工具", 点安装, 装完再执行一次
git config --global user.name "Sumer"
git config --global user.email "你注册 GitHub 用的邮箱"
git config --global init.defaultBranch main
```

> 不想公开邮箱的话, 可以在 GitHub → Settings → Emails 里找到 `xxxx+asiansumer@users.noreply.github.com` 形式的地址代替。

## 5. 配置 GitHub SSH 密钥 (一次性)

SSH 密钥让你 push 时不用每次输密码。

```bash
# 1) 生成密钥: 一路回车即可 (passphrase 可以留空, 也可以设一个)
ssh-keygen -t ed25519 -C "你注册 GitHub 用的邮箱"

# 2) 让 macOS 钥匙串记住它
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519

# 3) 复制公钥 (.pub 结尾的才是公钥, 私钥 id_ed25519 永远不要发给任何人)
pbcopy < ~/.ssh/id_ed25519.pub
```

为了重启电脑后不用重新 `ssh-add`, 用文本编辑器创建/打开 `~/.ssh/config` (`open -e ~/.ssh/config`, 文件不存在就先 `touch ~/.ssh/config`), 写入:

```
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

4) 打开 GitHub → 右上角头像 → **Settings → SSH and GPG keys → New SSH key**, Title 写 `MacBook M3 Max`, Key 里粘贴 (Cmd+V), 保存。

5) 测试:

```bash
ssh -T git@github.com
```

第一次会问 `Are you sure you want to continue connecting (yes/no)?`, 输入 `yes`。看到
`Hi asiansumer! You've successfully authenticated, but GitHub does not provide shell access.` 就成功了 (后半句是正常的, 不是报错)。

## 6. 第一次提交并推送到 GitHub

1) 在 GitHub 网页上新建仓库: <https://github.com/new>
   - Repository name: `yolov8-bdd100k-pedestrian`
   - Public
   - **不要**勾选 "Add a README" / ".gitignore" / "license" (本地已经有了, 勾了会导致第一次 push 冲突)

2) 在本地仓库目录里:

```bash
cd ~/你放代码的目录/yolov8-bdd100k-pedestrian
git init
git add .
git status                                   # 检查: 列表里不能有 datasets/、runs/、*.pt、*.jpg
git commit -m "Initial commit: YOLOv8 BDD100K pedestrian detection starter"
git branch -M main
git remote add origin git@github.com:asiansumer/yolov8-bdd100k-pedestrian.git
git push -u origin main
```

以后每次修改后只需要:

```bash
git add .
git commit -m "简短说明这次改了什么"
git push
```

---

## 常见错误与解决

| 报错 / 现象 | 原因 | 解决 |
|---|---|---|
| `zsh: command not found: miniconda` | 没有这个命令 | 用 `conda --version` |
| `Run 'conda init' before 'conda activate'` | conda 没接入 zsh | `conda init zsh`, 然后重开终端 |
| `conda: command not found` | conda 不在 PATH 里 | `~/miniconda3/bin/conda init zsh`, 重开终端 |
| `torch.backends.mps.is_available()` 输出 `False` | 装的是 x86_64 (Intel) 版 Python, 或 macOS 太旧 | 第 1 节检查 `arm64`; 更新 macOS |
| `NotImplementedError: ... not currently implemented for the MPS device` | 个别算子 MPS 还不支持 | 训练前执行 `export PYTORCH_ENABLE_MPS_FALLBACK=1` (该算子会退回 CPU 计算) |
| `MPS backend out of memory` | batch 太大 | `python scripts/train.py --config yolov8l --batch 2`, 并关掉其他占内存的程序 |
| `Dataset ... images not found` | 没转换数据, 或不在仓库根目录运行 | `cd` 到仓库根目录; 先运行第 3 步的转换脚本 |
| `Permission denied (publickey)` | GitHub 上没添加公钥, 或 agent 里没有密钥 | 重做第 5 节; `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` |
| `ERROR: Repository not found` | GitHub 上还没建仓库, 或名字/用户名拼错 | 检查 `git remote -v`; 改地址: `git remote set-url origin git@github.com:asiansumer/yolov8-bdd100k-pedestrian.git` |
| `! [rejected] main -> main (fetch first)` | 网页建仓库时勾了 README | `git pull --rebase origin main` 后再 `git push` |
| `error: src refspec main does not match any` | 还没有 commit, 或分支叫 master | 先 `git commit`; 再 `git branch -M main` |
| `File ... exceeds GitHub's file size limit of 100 MB` | 把权重 / 数据提交进去了 | 见下方"误提交了大文件" |
| `xcrun: error: invalid active developer path` | 没装命令行开发者工具 | `xcode-select --install` |

**误提交了数据或权重 (还没 push 成功时):**

```bash
git rm -r --cached datasets runs        # 从 Git 里移除, 但保留本地文件
git rm --cached *.pt                    # 根目录下自动下载的预训练权重
git commit --amend --no-edit            # 如果是刚才那次提交带进去的, 直接修改那次提交
```

如果大文件在更早的提交里, 需要改写历史, 建议先停下来查资料或找人帮忙, 不要强行 `push --force`。
