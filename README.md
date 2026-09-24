# Compact MNIST Model

这是一个参数量低于 4K 的 MNIST CNN baseline。当前阶段的目标是保持模型结构不变，建立可靠的 FP32 训练入口，并保存可恢复的 checkpoint；后续再在此基础上增加 INT8 推理导出。

## 模型约束

- 输入：`1 x 28 x 28` grayscale MNIST image
- 输出：10 个数字类别
- 可训练参数：`3,632`
- FP32 参数存储：约 `14.2 KiB`
- 网络结构：五个 `3x3 Conv/BatchNorm/ReLU`、三次池化，最后 `144 -> 10` 全连接
- 本项目的训练工程化不会改变 `Net` 的层数、通道数、kernel、pool 或全连接维度

## 安装

建议使用 Python 3.10+ 的虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch 的 CUDA 安装方式可能因平台不同而不同；如果需要 GPU，请按照对应 CUDA 版本的 PyTorch 安装命令替换 `torch` 和 `torchvision` 两项。

## 一键训练入口

依赖安装完成后，在项目根目录执行：

```bash
python train.py
```

默认行为：

- 自动选择 MPS、CUDA 或 CPU；
- 自动下载 MNIST 到 `data/`；
- 训练 20 个 epoch；
- 每个 epoch 结束后测试一次；
- 将训练指标写入 `outputs/metrics.jsonl`；
- 每个 epoch 保存最近状态到 `outputs/checkpoints/last.pt`；
- 当测试准确率刷新时保存 `outputs/checkpoints/best.pt`。

默认参数沿用原始 baseline：batch size `512`、最大学习率 `0.4`、momentum `0.95`、weight decay `0.0005`、OneCycleLR。模型输出 raw logits，训练使用 CrossEntropyLoss。

可以先检查配置和路径而不启动训练：

```bash
python train.py --dry-run
```

常用覆盖参数：

```bash
python train.py --epochs 20 --batch-size 512 --device cpu
python train.py --data-dir /path/to/mnist-data --output-dir outputs/run-01
```

## 断点恢复

`last.pt` 保存模型、optimizer、scheduler、epoch、最佳准确率和训练配置。恢复训练：

```bash
python train.py --resume outputs/checkpoints/last.pt
```

恢复时默认继续跑到 `--epochs` 指定的总 epoch 数。若修改了总 epoch、学习率或模型结构，应新建一个 output 目录，避免混用实验结果。

## 测试

运行模型结构测试和训练入口测试：

```bash
pytest -q
```

当前测试覆盖：

- 参数量必须为 `3,632`；
- Same padding、BatchNorm、无 bias 的卷积和全连接层尺寸正确；
- 前向输出形状为 `(batch, 10)`；
- `train.py` 的帮助参数和默认路径；
- checkpoint 保存与恢复训练状态。

## 输出文件

```text
data/                         # MNIST 数据，git ignored
outputs/
├── metrics.jsonl             # 每个 epoch 一行 JSON 指标
└── checkpoints/
    ├── best.pt               # 测试准确率最佳 checkpoint
    └── last.pt               # 最近一次 checkpoint
```

`outputs/`、数据集和模型二进制文件不会提交到 Git。后续 INT8 阶段会从 `best.pt` 生成独立的量化模型文件，不覆盖 FP32 checkpoint。

## INT8 精度测试

训练完成后，可以对 FP32 最佳 checkpoint 做静态 PTQ：

```bash
python quantize.py \
  --checkpoint outputs/checkpoints/best.pt \
  --data-dir data \
  --calibration-batches 20
```

脚本会融合 Conv+BN+ReLU，使用训练集样本做 calibration，然后在 CPU 上运行 INT8 Conv/Linear，并输出：

```text
fp32_accuracy=...
int8_accuracy=...
accuracy_drop=...
```

量化结果保存到：

```text
outputs/quantized/model_int8.pt
```

当前脚本用于验证 PyTorch 静态 INT8 的精度变化；后续硬件导出还需要把量化权重、INT32 bias、scale 和 zero-point 导出成硬件格式。
