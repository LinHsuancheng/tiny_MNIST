# Tiny MNIST CNN

面向 8 KB SRAM 硬件部署的 MNIST CNN。训练阶段使用 FP32；部署阶段将 BN fold 到前置卷积，并使用 INT8 权重、激活和 INT32 bias。

## 输入输出

- 输入：`28 × 28 × 1`，单通道灰度图
- 输出：`10` 个类别 logits
- 卷积：全部为 `3 × 3，padding=1，stride=1`
- 激活：ReLU
- 池化：`2 × 2 MaxPool，stride=2`

## 网络结构

```text
28 × 28 × 1
    │
    ├─ Conv 3×3  1 → 2
    ├─ BN + ReLU
    └─ MaxPool 2×2
         ↓
      14 × 14 × 2

    ├─ Conv 3×3  2 → 4
    ├─ BN + ReLU
         ↓
      14 × 14 × 4

    ├─ Conv 3×3  4 → 8
    ├─ BN + ReLU
    └─ MaxPool 2×2
         ↓
       7 × 7 × 8

    ├─ Conv 3×3  8 → 8
    ├─ BN + ReLU
    └─ MaxPool 2×2
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
| 1 | `28×28×1` | Conv `3×3`, `1→2`, `p=1` + BN + ReLU | `28×28×2` | 22 |
| 2 | `28×28×2` | MaxPool `2×2`, `s=2` | `14×14×2` | 0 |
| 3 | `14×14×2` | Conv `3×3`, `2→4`, `p=1` + BN + ReLU | `14×14×4` | 80 |
| 4 | `14×14×4` | Conv `3×3`, `4→8`, `p=1` + BN + ReLU | `14×14×8` | 304 |
| 5 | `14×14×8` | MaxPool `2×2`, `s=2` | `7×7×8` | 0 |
| 6 | `7×7×8` | Conv `3×3`, `8→8`, `p=1` + BN + ReLU | `7×7×8` | 592 |
| 7 | `7×7×8` | MaxPool `2×2`, `s=2` | `3×3×8` | 0 |
| 8 | `3×3×8` | Conv `3×3`, `8→16`, `p=1` + BN + ReLU | `3×3×16` | 1,184 |
| 9 | `3×3×16` | Flatten | `144` | 0 |
| 10 | `144` | Linear `144→10` | `10` | 1,450 |

卷积层均为 `bias=False`，因此卷积参数量只包含权重；表中的 BN 参数包含可训练的 scale 和 bias。完整参数计算为：

```text
Conv + BN = 22 + 80 + 304 + 592 + 1,184 = 2,182
FC        = 1,450
Total     = 3,632
```

对应的 PyTorch 模块顺序为：

```text
conv1 = Conv(1, 2)  → BN(2)  → ReLU → MaxPool
conv2 = Conv(2, 4)  → BN(4)  → ReLU
conv3 = Conv(4, 8)  → BN(8)  → ReLU → MaxPool
conv4 = Conv(8, 8)  → BN(8)  → ReLU → MaxPool
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

折叠后的核心权重和 bias 约占 `3.7 KB`，配合复用 activation buffer，可满足 `8 KB SRAM` 的部署目标。

主要 INT8 activation buffer 尺寸如下：

| 位置 | INT8 元素数 | 存储量 |
|---|---:|---:|
| 输入 | 784 | 784 B |
| Conv1 输出 | 1,568 | 1,568 B |
| Conv2 输出 | 784 | 784 B |
| Conv3 输出 | 392 | 392 B |
| Conv4 输出 | 392 | 392 B |
| Conv5 输出 | 144 | 144 B |

硬件采用输入/输出 buffer 复用，不保留全部中间激活；最大同时存活的输入和卷积输出约为 `2,352 B`。

## 结构约束

- 总可训练参数保持为 `3,632`
- 保持 5 个卷积层，通道变化为：`1→2→4→8→8→16`
- 保持 3 个 MaxPool，位置为 Conv1、Conv3、Conv4 之后
- 保持全连接层尺寸 `144→10`
- 不使用额外 padding 层或额外分支
