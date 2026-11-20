# 服务器复现位置

## 原始实验

- 源码：`/home/fengjunming/project/Moon/Moon_1115/official_dinov3_subpixel`
- 解释器：`/home/fengjunming/miniconda3/envs/fjm_moon/bin/python`
- CANN：`/usr/local/Ascend/ascend-toolkit/set_env.sh`
- 训练设备：物理 NPU 8–15。
- 实验目录：`/home/fengjunming/data/Moon_1115_artifacts/runs/official_dinov3_subpixel/261119_moon_dinov3_subpixel`
- 训练 manifest：`/home/fengjunming/data/Moon_1115_artifacts/runs/official_detail/2611182052_e018_detail/official_train/manifest.json`
- 官方训练包：`/home/fengjunming/data/LuSNAR_train.rar`
- 官方测试图像：`/home/fengjunming/data/LuSNAR_test`
- 公开测试标签索引：`/home/fengjunming/project/Moon/Moon_1115/data/sealed_test_labels.json`

原始训练入口要求 manifest 的来源为上述官方训练包路径，样本文件位于 manifest 同级的 `Moon_*/color` 和 `Moon_*/label`。原始测试入口读取上述官方测试图像目录。

## 权重

DINOv3：

`/home/fengjunming/data/Moon_1115_artifacts/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.bin`

最优 checkpoint：

`/home/fengjunming/data/Moon_1115_artifacts/runs/official_dinov3_subpixel/261119_moon_dinov3_subpixel/selection/best.pt`

最近 checkpoint 位于同一目录的 `latest.pt`。导出时 best 为第 171 轮，SHA256 记录在 `records/best_checkpoint.json`。

在新 clone 中复用已有预训练文件：

```bash
mkdir -p weights
cp /home/fengjunming/data/Moon_1115_artifacts/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.bin weights/
python scripts/prepare_weights.py
```

## 真实图像模型验证

`verify_model.py` 接收一个包含 `official_train/manifest.json` 的实验目录。可使用独立验证目录与已有数据目录的链接：

```bash
mkdir -p outputs/model_verification
ln -s /home/fengjunming/data/Moon_1115_artifacts/runs/official_detail/2611182052_e018_detail/official_train outputs/model_verification/official_train
python verify_model.py outputs/model_verification
```

该入口在 CPU 上检查真实图像输出，并执行真实梯度更新。验证结果写入独立目录。

## 源码身份

导出的模型、数据和训练代码保持原始文件内容。`scripts/` 提供仓库内运行入口、权重准备与源码校验。它们调用原始模型和训练实现。

`records/training_config.json` 保存原始配置及源码身份；`records/source_checksums.json` 保存此次导出文件的完整 SHA256。
