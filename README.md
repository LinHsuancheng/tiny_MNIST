# Tiny MNIST CNN

面向 8 KB SRAM 硬件部署的 MNIST CNN。训练阶段使用 FP32；部署阶段将 BN fold 到前置卷积，并使用 INT8 权重、激活和 INT32 bias。

## 输入输出

- 输入：`28 × 28 × 1`，单通道灰度图
- 输出：`10` 个类别 logits
- 卷积：全部为 `3 × 3，padding=1`；Conv1、Conv3、Conv4 使用 `stride=2`，其余使用 `stride=1`
- 激活：ReLU
- 下采样：stride=2 卷积计算原卷积输出中每个 `2 × 2` 区域的左上角位置，不再执行 MaxPool
- Conv4 的 `7×7 → 4×4` 输出裁掉最后一行和最后一列，保持原来的 `3×3` 特征尺寸

## 网络结构

```text
28 × 28 × 1
    │
    └─ Conv 3×3  1 → 2, stride=2 + BN + ReLU
         ↓
      14 × 14 × 2

    ├─ Conv 3×3  2 → 4
    ├─ BN + ReLU
         ↓
      14 × 14 × 4

    └─ Conv 3×3  4 → 8, stride=2 + BN + ReLU
         ↓
       7 × 7 × 8

    └─ Conv 3×3  8 → 8, stride=2 + BN + ReLU + 裁剪 4×4→3×3
         ↓
       3 × 3 × 8

    ├─ Conv 3×3  8 → 16
    └─ BN + ReLU
         ↓
       3 × 3 × 16

    └─ Flatten
         ↓
          144

    └─ FC 144 → 10
         ↓
       10 logits
```

## 逐层结构

| 序号 | 输入尺寸 | 运算 | 输出尺寸 | 参数量 |
|---:|---|---|---|---:|
| 0 | `28×28×1` | Input | `28×28×1` | 0 |
| 1 | `28×28×1` | Conv `3×3`, `1→2`, `p=1, s=2` + BN + ReLU | `14×14×2` | 22 |
| 2 | `14×14×2` | Conv `3×3`, `2→4`, `p=1, s=1` + BN + ReLU | `14×14×4` | 80 |
| 3 | `14×14×4` | Conv `3×3`, `4→8`, `p=1, s=2` + BN + ReLU | `7×7×8` | 304 |
| 4 | `7×7×8` | Conv `3×3`, `8→8`, `p=1, s=2` + BN + ReLU | `4×4×8` | 592 |
| 5 | `4×4×8` | 裁掉最后一行、最后一列 | `3×3×8` | 0 |
| 6 | `3×3×8` | Conv `3×3`, `8→16`, `p=1, s=1` + BN + ReLU | `3×3×16` | 1,184 |
| 7 | `3×3×16` | Flatten | `144` | 0 |
| 8 | `144` | Linear `144→10` | `10` | 1,450 |

卷积层均为 `bias=False`，因此卷积参数量只包含权重；表中的 BN 参数包含可训练的 scale 和 bias。完整参数计算为：

```text
Conv + BN = 22 + 80 + 304 + 592 + 1,184 = 2,182
FC        = 1,450
Total     = 3,632
```

按每次乘加计 1 MAC，五层卷积从原来的 `123,264 MAC/图` 降至 `51,336 MAC/图`（约减少 `58.4%`）。这个数字包含 Conv4 实际计算的完整 `4×4` 输出，未把裁掉的位置算成节省的计算量；实际运行时间仍需在目标硬件上测量。

对应的 PyTorch 模块顺序为：

```text
conv1 = Conv(1, 2, stride=2)  → BN(2)  → ReLU
conv2 = Conv(2, 4)  → BN(4)  → ReLU
conv3 = Conv(4, 8, stride=2)  → BN(8)  → ReLU
conv4 = Conv(8, 8, stride=2)  → BN(8)  → ReLU → 裁剪至 3×3
conv5 = Conv(8, 16) → BN(16) → ReLU
fc1   = Linear(144, 10)
```

## 参数量

| 模块 | 参数量 |
|---|---:|
| Conv 权重 | 2,106 |
| BatchNorm 可训练参数 | 76 |
| FC 权重和 bias | 1,450 |
| 总参数量 | **3,632** |

FP32 参数存储约为 `14,528 B`，即 `14.2 KiB`。

## INT8 部署

部署时执行 BN folding：

```text
Conv + BN + ReLU  →  INT8 Conv + INT8 ReLU
```

BN 不作为独立硬件模块存在，也不额外占用 activation SRAM。折叠后：

- Conv 和 FC 权重使用 INT8
- bias 使用 INT32
- 中间激活使用 INT8
- 累加器使用 INT32
- 输出 scale 和 zero-point 用于层间重新量化

折叠后 INT8 权重为 `3,546 B`，48 个 bias 使用 INT32 为 `192 B`；逐输出通道权重 scale 若使用 FP32，另需 `192 B`。这些参数和下述激活缓冲区合计 `5,106 B`，还需为层描述、输出 scale/zero-point、对齐及计算暂存预留空间。PyTorch 的 `.pt` 检查点包含序列化开销，不能直接作为 SRAM 镜像。

主要 INT8 activation buffer 尺寸如下：

| 位置 | INT8 元素数 | 存储量 |
|---|---:|---:|
| 输入 | 784 | 784 B |
| Conv1 输出 | 392 | 392 B |
| Conv2 输出 | 784 | 784 B |
| Conv3 输出 | 392 | 392 B |
| Conv4 卷积输出 | 128 | 128 B |
| Conv4 裁剪后 | 72 | 72 B |
| Conv5 输出 | 144 | 144 B |

硬件采用输入/输出 buffer 复用，不保留全部中间激活；最大同时存活的输入和卷积输出约为 `1,176 B`（输入 784 B + Conv1 输出 392 B）。如果硬件直接跳过 Conv4 最后一行、列的无用位置，还可避免这些位置的计算和存储。

## 结构约束

- 总可训练参数保持为 `3,632`
- 保持 5 个卷积层，通道变化为：`1→2→4→8→8→16`
- Conv1、Conv3、Conv4 使用 stride=2，并保持左上角采样位置
- 保持全连接层尺寸 `144→10`
- 不使用额外 padding 层或额外分支

原始 MaxPool 是取每个 `2×2` 区域的最大值；当前实现取原卷积输出的左上角值，因此两种模型的数值结果和准确率不保证相同，需重新训练并评估。旧 checkpoint 的权重形状仍可加载，但不能直接当作新模型的精度验证。

## 训练验证

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
NO_ALBUMENTATIONS_UPDATE=1 TQDM_DISABLE=1 .venv/bin/python train.py --device cuda --num-workers 2 --epochs 120 --max-lr 0.003 --warmup-pct 0.3 --output-dir outputs/narrow_e120_lr003
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python - <<'PY'
import torch
import quantize

torch.manual_seed(14596)
quantize.main([
    '--checkpoint', 'outputs/narrow_e120_lr003/checkpoints/best.pt',
    '--data-dir', 'data',
    '--output', 'outputs/pool-stride2-fp32-int8/model_int8.pt',
    '--calibration-batches', '20',
    '--num-workers', '4',
])
PY
```

2026-09-26 使用原始 `2→4→8→8→16` 通道数，在 A100 上训练 120 个 epoch：最佳 FP32 测试准确率为 `99.07%`；相同 checkpoint 在 CPU 上量化后为 `99.01%`。15 项测试通过，其中包含量化前后无 MaxPool 的结构检查。量化流程本身未修改；这些准确率不代表旧 MaxPool 模型的准确率，卷积 MAC 减少也不是实测运行时间。

已保存的 INT8 模型可直接运行：`uv run example.py` 推理测试集第 0 张图，添加 `--index 42` 可选择其他图片，添加 `--accuracy` 可评估全部测试集。无需重新训练或量化。
