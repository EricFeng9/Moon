# Moon：DINOv3 门控融合与亚像素分类修正

月面五类语义分割的 Ascend NPU 实验源码。模型变体为 `dinov3_subpixel_e018`，源码来自正在进行的 200 轮亚像素实验；模型、损失和数据增强保持该实验的原始实现。

## 网络结构

同一张 RGB 图像进入冻结的 DINOv3 ViT-S/16 与可训练的 MiT-B2 前两级。局部分支读取局部对比度视图，两个尺度分别通过逐像素门控与 DINOv3 特征融合。四级解码器使用类别原型上下文、区域特征修正和层级分类。亚像素分支结合原始 RGB 与最终解码特征，生成完整分辨率的分类分数修正。

主要实现：

| 文件 | 内容 |
| --- | --- |
| `dinov3_model.py` | 冻结编码器、逐像素门控、亚像素残差修正 |
| `e018_base.py` | 金字塔解码、类别原型、层级分类和完整损失 |
| `context.py` | 区域类别特征修正 |
| `encoder.py` | MiT 编码器与 SDPA attention |
| `data.py` | 数据读取、标签映射、增强与分布式采样 |
| `train.py` | BF16、梯度累积、Moon5 验证与 checkpoint |
| `predict.py` | Moon5 重放与六视图测试预测 |
| `competition_metrics.py` | 场景加权评分及 pooled 指标 |

## 版本与记录

导出时 Moon5 最优 checkpoint 为第 **171** 轮，mIoU 为 **90.4332%**。完整记录见 `records/best_checkpoint.json`。

| 固定权重 | Moon5 mIoU | 完整测试 S | Pooled test mIoU |
| --- | --- | --- | --- |
| 第 166 轮备份 | 90.2334% | 87.1860% | 86.7718% |
| 第 171 轮 best | 90.4332% | 导出时尚未完成 | 导出时尚未完成 |

第 166 轮完整测试记录位于 `records/snapshot_epoch166_test.json`。最优权重身份、实际训练配置、源码 SHA256 和环境版本分别保存在 `records/best_checkpoint.json`、`records/training_config.json`、`records/source_checksums.json`、`records/environment.json`。

代码仓库保存源码和记录。数据、DINOv3 预训练权重、实验 `best.pt` / `latest.pt`、预测图像和运行缓存在服务器保存。

## 环境与依赖

实际环境为 Python 3.11、CANN 8.5.0、PyTorch 2.9.0、torchvision 0.24.0、torch_npu 2.9.0.post1。具体构建版本见 `records/environment.json`。安装匹配的 PyTorch / torchvision / torch_npu 后，安装其他 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

DINOv3 使用固定 revision 的 Git submodule：

```bash
git clone --recurse-submodules https://github.com/EricFeng9/Moon.git
cd Moon
```

已有普通 clone 可执行 `git submodule update --init --recursive`。DINOv3 代码和权重的来源、版本及 SHA256 记录在 `dependencies.json`，许可证位于 `licenses/DINOv3_LICENSE.md`。

准备权重：

```bash
python scripts/prepare_weights.py
```

脚本将权重保存到 `weights/dinov3_vits16_pretrain_lvd1689m-08c60483.bin`，并检查完整 SHA256。服务器已有权重可直接复制或建立目录链接，路径见 `docs/server_reproduction.md`。

## 训练

训练使用官方训练包中的 Moon1、3、6、7、8，共 6,403 张；Moon5 的 1,599 张用于每轮验证并选择 `best.pt`。输入为 1024，8 张 NPU、全局 batch 32、BF16、AdamW，最多 200 轮，早停关闭，使用普通模型权重。

该版本保留原始训练入口对 NPU 8–15、官方 manifest 来源及初始化身份的检查。训练 manifest 包含 `train`、`val`、`source`、`fingerprint`；每条样本包含 `scene`、`name`、`image`、`mask`。服务器 manifest 和数据位置见 `docs/server_reproduction.md`。

指定新的实验目录和现有 manifest，然后启动训练：

```bash
export MOON_PYTHON=/home/fengjunming/miniconda3/envs/fjm_moon/bin/python
bash scripts/train_npu.sh /path/to/new_run /path/to/official_train/manifest.json
```

训练保留 `selection/best.pt`、`selection/latest.pt`，同时保存配置、训练记录和每个 rank 的日志。中断恢复使用同一实验目录：

```bash
bash scripts/train_npu.sh /path/to/new_run /path/to/official_train/manifest.json --resume
```

原始 `run.py`、`run_npu.sh`、`launch_200.sh` 及续训验证脚本保留服务器调度时的路径。以上 `scripts/train_npu.sh` 从当前仓库调用同一份 `train.py`。

## 测试与指标

```bash
bash scripts/predict_npu.sh /path/to/new_run
```

测试前重放 Moon5，检查结果与 best checkpoint 记录一致；随后对 Moon2、4、9 共 5,004 张图像执行 768 / 1024 / 1280 三种尺寸及水平翻转的六视图预测。

类别顺序为 `regolith, rock, crater, mountain, sky`。评分先在每个场景累积像素混淆矩阵，再计算该场景 mIoU，场景权重为 Moon2:Moon4:Moon9 = 1:1:2。Pooled mIoU 单独由全部测试像素计算。

本仓库记录的公开标签测试属于独立研究评估。标签文件与图像分别管理；训练与 checkpoint 选择使用训练场景和 Moon5。

## 源码核验

```bash
python scripts/check_source.py
```

该命令检查原始运行源码是否与导出记录逐文件一致。模型验证入口 `verify_model.py` 使用真实训练图像，检查 1024 输出、冻结权重、门控梯度、亚像素梯度及零初始化输出一致性。
