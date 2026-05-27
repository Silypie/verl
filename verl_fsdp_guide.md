# VERL FSDP 训练参数详解与配置指南

> 本文档介绍 VERL 框架中用于控制 FSDP（Fully Sharded Data Parallel）分布式训练的参数及其原理，帮助初学者理解并正确配置大规模模型训练。

---

## 目录

1. [引言：为什么需要 FSDP？](#1-引言为什么需要-fsdp)
2. [FSDP 核心原理](#2-fsdp-核心原理)
3. [VERL FSDP 参数详解](#3-verl-fsdp-参数详解)
4. [参数配置速查表](#4-参数配置速查表)
5. [常见问题与排错](#5-常见问题与排错)

---

## 1. 引言：为什么需要 FSDP？

### 1.1 单卡训练的内存瓶颈

假设你正在训练一个 70 亿参数（7B）的大语言模型，使用 Adam 优化器。我们来算一笔显存账：

| 模型状态 | 计算公式 | 显存占用 |
|---------|---------|---------|
| 参数（FP32）| 4 bytes × 7B | 28 GB |
| 梯度（FP32）| 4 bytes × 7B | 28 GB |
| 优化器状态（momentum + variance）| 8 bytes × 7B | 56 GB |
| **模型状态合计** | | **112 GB** |

这还没算上激活值（activations）、临时缓冲区和其他开销。单张 A100 GPU 只有 80GB 显存，根本放不下一个 7B 模型的完整训练状态。

这就是**内存墙（Memory Wall）**问题——模型越来越大，单卡显存却有限。

### 1.2 DDP：数据并行，但模型冗余

传统的 **Distributed Data Parallel (DDP)** 是这样工作的：

```
GPU 0: [完整模型] + [数据分片 0] → 计算梯度 0
GPU 1: [完整模型] + [数据分片 1] → 计算梯度 1
GPU 2: [完整模型] + [数据分片 2] → 计算梯度 2
GPU 3: [完整模型] + [数据分片 3] → 计算梯度 3
              ↓
        All-Reduce 同步梯度
              ↓
        每张卡独立更新模型
```

DDP 的核心思想是**数据分片、模型复制**。每张 GPU 都保存一份完整的模型副本，只有输入数据不同。计算完梯度后，通过 All-Reduce 操作让所有卡的梯度保持一致。

**DDP 的问题**：模型状态（参数 + 梯度 + 优化器状态）在每张卡上都是完整的。对于 7B 模型，4 张卡一共冗余存储了 448GB 的模型状态，但每张卡仍然需要 112GB，瓶颈没有解决。

### 1.3 FSDP：把模型也分片

**Fully Sharded Data Parallel (FSDP)** 的核心思想是：**不仅数据要分片，模型状态也要分片**。

```
GPU 0: [参数分片 0] + [梯度分片 0] + [优化器状态分片 0] + [数据分片 0]
GPU 1: [参数分片 1] + [梯度分片 1] + [优化器状态分片 1] + [数据分片 1]
GPU 2: [参数分片 2] + [梯度分片 2] + [优化器状态分片 2] + [数据分片 2]
GPU 3: [参数分片 3] + [梯度分片 3] + [优化器状态分片 3] + [数据分片 3]
```

在 FSDP 中，每张 GPU 只存储模型状态的 1/N（N 为 GPU 数量）。需要计算时，通过 All-Gather 通信把参数从其他卡收集过来；计算完成后，再把参数丢弃，只保留自己的分片。

**效果**：4 张卡训练 7B 模型，每张卡只需约 28GB 模型状态（112GB ÷ 4），加上激活值后也能轻松放入 A100。

### 1.4 VERL 中的 FSDP

VERL 是一个用于 RLHF（基于人类反馈的强化学习）和 PPO 训练的大模型训练框架。与普通的监督学习不同，RLHF 训练同时需要维护多个模型：

- **Actor（策略模型）**：生成回答
- **Critic（价值模型）**：评估回答质量
- **Reference（参考模型）**：提供 KL 散度约束
- **Rollout（推理模型）**：生成训练数据

四个模型同时驻留显存，内存压力是普通训练的数倍。因此，**正确配置 FSDP 参数对 VERL 至关重要**——它直接决定了你能训练多大的模型、用多长的序列、以及多大的批次。

接下来的章节，我们将深入讲解 VERL 中每个 FSDP 参数的含义、原理和配置建议。

---

## 2. FSDP 核心原理

### 2.1 分片：像拼图一样分配模型

想象你有一幅巨大的拼图（模型），但你和朋友们（GPU）每个人的桌子都很小，放不下完整的图。FSDP 的做法是：**把拼图拆成多块，每个人只保管其中一块**。

当需要看图的全貌时（前向/反向计算），大家把各自手里的拼图块拿出来拼在一起；看完之后又各自拆回去，只保留自己的那块。

在 FSDP 中，"拼图块"就是**模型状态的分片（shard）**。具体来说：

- **参数分片**：每个参数张量被切成 N 份，GPU i 只保存第 i 份
- **梯度分片**：同样切分，计算后各卡只保留自己那份梯度
- **优化器状态分片**：Adam 的 momentum 和 variance 也按同样方式切分

### 2.2 参数的生命周期

FSDP 中参数不是一直完整的，而是在"完整"和"分片"两种状态之间动态切换：

```
┌─────────────┐     All-Gather      ┌─────────────┐
│  分片状态    │  ───────────────→  │  完整状态    │
│ (Shard)     │   收集所有分片      │ (Unshard)   │
│  每张卡存    │                    │  每张卡有    │
│  1/N 参数   │                    │  完整参数    │
└─────────────┘                    └─────────────┘
       ↑                                    │
       │     Reduce-Scatter                 │
       │  ←──────────────────              │ Forward / Backward
       │   汇总并重新分片梯度                │ 前向/反向计算
       │                                    ↓
└─────────────┐                    ┌─────────────┐
│  更新后的    │                    │  计算梯度    │
│  分片参数    │                    │  (Gradient)  │
└─────────────┘                    └─────────────┘
```

**详细流程：**

1. **初始状态**：所有参数处于**分片状态（Sharded）**，每张卡只存 1/N
2. **All-Gather 阶段**：前向传播开始前，各卡通过 All-Gather 通信把参数分片凑成完整参数
3. **Forward 阶段**：用完整参数计算前向传播，得到激活值和输出
4. **Backward 阶段**：用完整参数计算反向传播，得到完整梯度
5. **Reduce-Scatter 阶段**：把完整梯度汇总并重新分片，每张卡只保留自己那部分梯度的累加结果
6. **Optimizer Step**：各卡用自己的参数分片 + 梯度分片 + 优化器状态分片，独立更新参数

### 2.3 两种关键通信操作

FSDP 依赖两种集体通信（Collective Communication）操作：

#### All-Gather：把碎片拼成完整图

```
GPU 0: [分片 A0] ───┐
GPU 1: [分片 A1] ───┼──→ All-Gather ──→ GPU 0~3: [完整 A]
GPU 2: [分片 A2] ───┤
GPU 3: [分片 A3] ───┘
```

每张卡贡献自己的分片，最终所有卡都得到完整的数据。**通信量**：每个参数需要从其他卡接收 (N-1)/N 的数据，总通信量约为 `参数大小 × (N-1)/N`。

#### Reduce-Scatter：汇总后重新分片

```
GPU 0~3: [完整梯度 G] ──→ Reduce-Scatter ──→ GPU 0: [分片 G0]
                                               GPU 1: [分片 G1]
                                               GPU 2: [分片 G2]
                                               GPU 3: [分片 G3]
```

所有卡先把自己的梯度贡献出来做求和（Reduce），然后把结果分片发回（Scatter）。**通信量**：与 All-Gather 相当。

### 2.4 前向与反向的参数状态差异

在前向传播和反向传播过程中，参数的状态管理有所不同：

| 阶段 | 参数状态 | 说明 |
|-----|---------|------|
| 前向开始前 | All-Gather → 完整 | 收集所有分片 |
| 前向计算中 | 完整 | 用完整参数计算 |
| 前向结束后 | Reshard → 分片（默认）| 丢弃完整参数，释放显存 |
| 反向开始前 | All-Gather → 完整 | 再次收集参数（用于梯度计算）|
| 反向计算中 | 完整 | 计算梯度 |
| 反向结束后 | Reduce-Scatter → 分片 | 梯度也分片 |

**关键洞察**：默认情况下（`reshard_after_forward=True`），前向结束后参数会重新分片。这意味着在两次计算之间，模型占用的显存是最小的。但代价是反向传播时需要再次 All-Gather，多了一次通信。

### 2.5 FSDP1 vs FSDP2

VERL 同时支持 FSDP1 和 FSDP2，理解它们的区别有助于选择正确的参数：

| 特性 | FSDP1 | FSDP2 |
|-----|-------|-------|
| PyTorch API | `FullyShardedDataParallel` wrapper | `fully_shard` composable API |
| 模型包装方式 | 把整个模型包在一个 FSDP 对象里 | 逐层（layer-wise）应用 FSDP |
| 参数存储 | FlatParameter（展平参数）| DTensor（分布式张量）|
| CPU Offload | `CPUOffload` 参数 | `CPUOffloadPolicy` |
| 适用 PyTorch 版本 | 广泛支持 | PyTorch ≥ 2.4 |
| 与 `torch.compile` 兼容性 | 一般 | 更好 |

**在 VERL 中的选择**：
- `strategy: fsdp` → 使用 FSDP1
- `strategy: fsdp2` → 使用 FSDP2

**参数兼容性注意**：
- `forward_prefetch` 和 `use_orig_params` **仅适用于 FSDP1**
- `offload_policy` **仅适用于 FSDP2**
- `param_offload` 和 `optimizer_offload` 是 VERL 自己实现的卸载逻辑，与 FSDP 版本无关

---

## 3. VERL FSDP 参数详解

### 3.1 内存优化参数组

#### 3.1.1 param_offload

**参数路径**：`actor_rollout_ref.actor.fsdp_config.param_offload`

**默认值**：`False`

**作用**：是否将模型参数从 GPU 显存**卸载（offload）**到 CPU 内存。

**原理详解**：

在标准 FSDP 中，参数虽然被分片，但仍然驻留在 GPU 显存中。当 `param_offload=True` 时，VERL 会在不计算时把参数分片从 GPU 搬到 CPU，需要计算时再搬回来。

```
GPU 显存: [参数分片] ──offload──→ CPU 内存: [参数分片]
                ↑______________|
                  load back when needed
```

**代码实现**：VERL 在 `fsdp_utils.py` 中实现了 `offload_fsdp_model_to_cpu()` 和 `load_fsdp_model_to_gpu()` 函数，通过操作 FSDP 的 `flat_param` 来完成卸载和加载。

**适用场景**：
- 显存极度紧张，即使 FSDP 分片后仍不够用
- 使用大模型（如 70B+）进行推理或 Reference 模型计算

**Trade-off**：
- ✅ 大幅降低 GPU 显存占用
- ❌ 增加 CPU ↔ GPU 数据传输开销，速度明显变慢
- ❌ VERL 注释明确指出：Actor 训练时开启 CPU offload 会导致梯度累积结果不正确

**配置建议**：
- **Actor 训练**：保持 `False`（VERL 强制关闭，因为会导致梯度累积错误）
- **Reference/Rollout 模型**：可以设为 `True`，因为它们只做前向传播
- **Critic 模型**：视显存情况而定

---

#### 3.1.2 optimizer_offload

**参数路径**：`actor_rollout_ref.actor.fsdp_config.optimizer_offload`

**默认值**：`False`

**作用**：是否将**优化器状态**（Adam 的 momentum 和 variance）从 GPU 卸载到 CPU。

**原理详解**：

优化器状态通常占模型状态的 2 倍大小（Adam 需要保存 momentum 和 variance）。对于 7B 模型，优化器状态约 56GB。当 `optimizer_offload=True` 时，这些状态存储在 CPU 内存中，只在执行 `optimizer.step()` 时临时搬到 GPU。

```
GPU 显存: [参数分片] + [梯度分片]        CPU 内存: [优化器状态分片]
                ↓                                    ↓
        optimizer.step() 时临时加载 ───────────────→
```

**代码实现**：VERL 在 `fsdp_utils.py` 中实现了 `offload_fsdp_optimizer()` 和 `load_fsdp_optimizer()` 函数，遍历 optimizer state 中的所有张量进行设备迁移。

**适用场景**：
- 显存紧张，但参数本身可以放在 GPU 上
- 训练大模型时，优化器状态是显存的大头

**Trade-off**：
- ✅ 显著降低 GPU 显存占用（可降低约 2/3 的模型状态显存）
- ❌ 每次 `optimizer.step()` 需要 CPU ↔ GPU 传输，训练速度下降
- ❌ 需要足够的 CPU 内存来存储优化器状态

**配置建议**：
- **显存充裕**：保持 `False`，追求训练速度
- **显存紧张**：设为 `True`，用 CPU 内存换 GPU 显存
- **注意**：可以与 `param_offload` 独立配置，常见组合是 `param_offload=False, optimizer_offload=True`

---

#### 3.1.3 offload_policy

**参数路径**：`actor_rollout_ref.actor.fsdp_config.offload_policy`

**默认值**：`False`

**作用**：**仅用于 FSDP2**，控制是否使用 PyTorch 原生的 `CPUOffloadPolicy` 进行参数/梯度/优化器卸载。

**原理详解**：

FSDP2 引入了原生的 `CPUOffloadPolicy`，与 VERL 自己实现的 `param_offload`/`optimizer_offload` 不同。当 `offload_policy=True` 时，FSDP2 会在训练过程中自动将参数、梯度和优化器状态卸载到 CPU，并在需要时自动加载。

```python
# FSDP2 中的实现（verl/workers/engine/fsdp/transformer_impl.py）
if self.engine_config.offload_policy or self.engine_config.forward_only:
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    self._uses_fsdp2_cpu_offload_policy = True
```

注意代码中的 `pin_memory=True`：这会使用固定内存（pinned memory），加速 CPU ↔ GPU 的数据传输。

**适用场景**：
- 使用 FSDP2 策略时
- 希望利用 PyTorch 原生卸载机制而不是 VERL 自定义实现

**Trade-off**：
- ✅ FSDP2 原生支持，与 `torch.compile` 兼容性更好
- ✅ `pin_memory=True` 加速数据传输
- ❌ 仅 FSDP2 可用，FSDP1 不支持
- ❌ 同样会引入 CPU ↔ GPU 传输开销

**配置建议**：
- **FSDP1 用户**：此参数无效，使用 `param_offload` 和 `optimizer_offload`
- **FSDP2 用户**：如果需要卸载，优先使用 `offload_policy=True`（比 VERL 自定义实现更稳定）
- **forward_only 模式**：VERL 会自动启用 `offload_policy`（见 3.5.1 节）

---

#### 3.1.4 reshard_after_forward

**参数路径**：`actor_rollout_ref.actor.fsdp_config.reshard_after_forward`

**默认值**：`True`

**作用**：控制**前向传播结束后是否重新分片参数**，是 FSDP 最核心的内存-速度权衡参数。

**原理详解**：

在 FSDP 中，前向传播开始前需要 All-Gather 收集完整参数。前向传播结束后，有两种选择：

**当 `reshard_after_forward=True`（默认）**：
```
前向开始前: All-Gather → [完整参数] → Forward → Reshard → [分片参数]
反向开始前: All-Gather → [完整参数] → Backward → Reduce-Scatter → [分片梯度]
```
- 前向结束后立即丢弃完整参数，只保留分片
- **显存优势**：两次前向之间显存占用最小
- **通信代价**：反向传播前需要再次 All-Gather

**当 `reshard_after_forward=False`**：
```
前向开始前: All-Gather → [完整参数] → Forward → [保持完整参数] → Backward → Reduce-Scatter
```
- 前向结束后保持参数完整，反向传播直接使用
- **显存代价**：参数在前后向之间保持完整，占用更多显存
- **通信优势**：省去反向前的 All-Gather，速度更快

**在 VERL 中的实现**：

```python
# verl/workers/engine/fsdp/utils.py
if zero3_enable:
    fsdp_strategy = ShardingStrategy.FULL_SHARD      # reshard_after_forward=True
    hsdp_strategy = ShardingStrategy.HYBRID_SHARD
else:
    fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP   # reshard_after_forward=False
    hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
```

- `reshard_after_forward=True` → 使用 `FULL_SHARD`（ZeRO-3）
- `reshard_after_forward=False` → 使用 `SHARD_GRAD_OP`（ZeRO-2）

**适用场景**：
- **显存紧张**：保持 `True`（默认），用通信换显存
- **追求速度**：设为 `False`，用显存换速度
- **推理/前向 only**：设为 `False`，因为不需要反向传播

**Trade-off**：

| 设置 | 显存占用 | 通信量 | 适用场景 |
|-----|---------|-------|---------|
| `True` | 低 | 高（两次 All-Gather）| 训练大模型，显存紧张 |
| `False` | 高 | 低（一次 All-Gather）| 小模型，追求速度，或纯推理 |

**配置建议**：
- **默认保持 `True`**，这是大多数训练场景的最佳选择
- **如果显存充裕且追求速度**：设为 `False`
- **如果看到 OOM（显存溢出）**：确保是 `True`
- **FSDP2 中**：同样支持此参数，通过 `reshard_after_forward` 直接传给 `fully_shard()`

### 3.2 性能优化参数组

#### 3.2.1 forward_prefetch

**参数路径**：`actor_rollout_ref.actor.fsdp_config.forward_prefetch`

**默认值**：`False`

**作用**：**仅用于 FSDP1**，在前向传播过程中**预取（prefetch）**下一层所需的参数，以重叠通信和计算。

**原理详解**：

在 FSDP1 中，模型被包装成一个大的 FSDP 对象，内部包含多个 FSDP 单元（通常是 Transformer 层）。默认情况下，每一层的前向传播流程是：

```
Layer 0: All-Gather 参数 → Forward 计算 → Reshard 参数
Layer 1: All-Gather 参数 → Forward 计算 → Reshard 参数
Layer 2: All-Gather 参数 → Forward 计算 → Reshard 参数
...
```

每一层都要等待 All-Gather 完成才能开始计算，通信和计算是串行的。

当 `forward_prefetch=True` 时，FSDP1 会在**当前层还在计算时**，就提前发起下一层的 All-Gather 请求：

```
Layer 0: All-Gather ──→ Forward 计算 ──→ Reshard
                              ↓（ overlap ）
Layer 1:              All-Gather ──→ Forward 计算 ──→ Reshard
                                         ↓（ overlap ）
Layer 2:                           All-Gather ──→ Forward 计算
```

这样，下一层的参数收集与当前层的计算**并行进行**，隐藏了通信延迟。

**在 VERL 中的实现**：

```python
# verl/utils/fsdp_utils.py
if config.get("forward_prefetch", False):
    fsdp_modules = [m for m in modules if isinstance(m, FSDPModule)]
    for i, m in enumerate(fsdp_modules):
        next_targets = fsdp_modules[i + 1 : i + 2]  # depth=1
        if next_targets and hasattr(m, "set_modules_to_forward_prefetch"):
            m.set_modules_to_forward_prefetch(next_targets)
```

注意：这是 FSDP2 中的兼容实现，通过 `set_modules_to_forward_prefetch` 模拟 FSDP1 的 prefetch 行为。在纯 FSDP1 中，直接通过 `forward_prefetch=True` 参数启用。

**适用场景**：
- 使用 FSDP1 策略
- 网络通信是瓶颈（如使用 InfiniBand 但带宽仍不足）
- 层数较多的模型（如 32 层、64 层 Transformer）

**Trade-off**：
- ✅ 重叠通信和计算，提升吞吐量
- ❌ 增加显存占用（需要同时保存当前层和预取层的完整参数）
- ❌ 仅 FSDP1 有效，FSDP2 有自己的预取机制

**配置建议**：
- **FSDP1 + 通信瓶颈**：设为 `True`
- **FSDP2**：此参数无效，FSDP2 内部自动处理预取
- **显存紧张**：保持 `False`，避免额外的显存开销

---

#### 3.2.2 use_orig_params

**参数路径**：`actor_rollout_ref.actor.fsdp_config.use_orig_params`

**默认值**：`False`

**作用**：**仅用于 FSDP1**，控制 FSDP 是否使用模块的**原始参数**来初始化，而不是创建新的 FlatParameter。

**原理详解**：

FSDP1 的默认行为是将多个参数展平（flatten）成一个大的 `FlatParameter`。这带来两个问题：

1. **参数访问**：原始参数被替换为 `FlatParameter` 的视图，直接访问 `module.weight` 得到的是视图而不是真实参数
2. **多参数模块**：某些模块（如包含多个权重的自定义层）可能无法正确包装

当 `use_orig_params=True` 时，FSDP1 会保留原始参数，不进行展平：

```
use_orig_params=False（默认）:
  param1, param2, param3 → [FlatParameter] → 内部管理

use_orig_params=True:
  param1, param2, param3 → 保持独立 → FSDP 分别管理
```

**关键影响**：

| 特性 | `use_orig_params=False` | `use_orig_params=True` |
|-----|------------------------|------------------------|
| 参数存储 | FlatParameter（展平） | 原始参数（独立） |
| 与 LoRA 兼容性 | 有限 | 更好 |
| 冻结参数支持 | 同一 FSDP 单元内必须全冻结或全可训练 | 支持混合冻结 |
| 显存开销 | 略低 | 略高 |

**适用场景**：
- 使用 LoRA 微调（PEFT）
- 需要冻结部分参数、训练其他参数
- 使用非标准参数结构的自定义模型

**Trade-off**：
- ✅ 更好的 LoRA 和 PEFT 兼容性
- ✅ 支持混合冻结/训练参数
- ❌ 略高的显存开销（因为不展平，无法极致优化内存）

**配置建议**：
- **使用 LoRA**：设为 `True`
- **全参数训练**：保持 `False`（默认），显存更优
- **FSDP2**：此参数不存在，FSDP2 天然使用原始参数（DTensor）

---

#### 3.2.3 fsdp_size

**参数路径**：`actor_rollout_ref.actor.fsdp_config.fsdp_size`

**默认值**：`-1`（自动，使用所有 GPU）

**作用**：控制每个 **FSDP 分片组（shard group）** 中的 GPU 数量。

**原理详解**：

默认情况下（`fsdp_size=-1`），FSDP 会把所有 GPU 放入一个分片组，每张卡保存 1/N 的参数（N = 总 GPU 数）。

当设置 `fsdp_size=k`（k < N）时，GPU 被分成多个独立的 FSDP 组：

```
总 GPU 数 = 8, fsdp_size = 4

FSDP Group 0: GPU 0, 1, 2, 3  → 每张卡存 1/4 参数
FSDP Group 1: GPU 4, 5, 6, 7  → 每张卡存 1/4 参数

数据并行：Group 0 和 Group 1 之间是 DDP 关系（数据分片，模型复制）
```

这就是 **Hybrid Sharding（混合分片）**：组内 FSDP（模型分片），组间 DDP（数据并行）。

**在 VERL 中的实现**：

```python
# verl/workers/engine/fsdp/utils.py
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh
```

当 `fsdp_size < world_size` 时，创建 2D device mesh：
- `mesh_dim_names=["ddp", "fsdp"]`
- 第一维是 DDP 维度（数据并行）
- 第二维是 FSDP 维度（模型分片）

**适用场景**：
- **节点内通信快、节点间通信慢**：如单机 8 卡，节点间用 Ethernet
  - 设 `fsdp_size=8`（单机内 FSDP，机间 DDP）
- **模型较小，不需要全部分片**：如 7B 模型用 64 卡，分片太细反而增加通信
  - 设 `fsdp_size=8`，8 组各 8 卡，每组存 1/8 模型
- **与流水线并行/张量并行组合**：需要精细控制并行维度

**Trade-off**：

| fsdp_size | 分片粒度 | 通信模式 | 适用场景 |
|-----------|---------|---------|---------|
| `-1`（默认）| 最细（1/N）| 全 FSDP | 大模型，节点内高速互联 |
| `8`（如单机）| 中等（1/8）| 组内 FSDP + 组间 DDP | 多机训练，机间带宽低 |
| `1` | 无分片 | 纯 DDP | 小模型，追求速度 |

**配置建议**：
- **默认 `-1`**：大多数情况下无需修改
- **多机训练（节点间带宽低）**：设为单机的 GPU 数（如 8）
- **注意**：`fsdp_size` 必须能整除总 GPU 数

### 3.3 序列并行参数

#### 3.3.1 ulysses_sequence_parallel_size

**参数路径**：`actor_rollout_ref.actor.ulysses_sequence_parallel_size`

**默认值**：`1`（不使用序列并行）

**作用**：启用 **Ulysses 序列并行（Sequence Parallelism）**，将长序列沿序列维度分片到多个 GPU 上处理。

**原理详解**：

在 Transformer 训练中，激活值的显存占用与序列长度成正比。对于长序列（如 32K、64K tokens），即使使用了 FSDP 分片参数，激活值仍可能占满显存。

Ulysses 序列并行的核心思想是：**把输入序列切成多段，每段交给不同的 GPU 处理**。

```
输入序列: [t1, t2, t3, t4, t5, t6, t7, t8]  (长度 8)

ulysses_sequence_parallel_size = 2:
  GPU 0 (SP rank 0): [t1, t2, t3, t4]
  GPU 1 (SP rank 1): [t5, t6, t7, t8]
  
ulysses_sequence_parallel_size = 4:
  GPU 0: [t1, t2]
  GPU 1: [t3, t4]
  GPU 2: [t5, t6]
  GPU 3: [t7, t8]
```

**关键特点**：

1. **与 FSDP 正交**：FSDP 沿参数维度分片，Ulysses 沿序列维度分片，两者可以叠加
2. **All-Gather 通信**：前向传播后需要通过 All-Gather 收集各段的输出，拼成完整序列
3. **与 FSDP 共享进程组**：在 VERL 中，Ulysses 序列并行组是在 FSDP 进程组之上创建的

**在 VERL 中的实现**：

```python
# verl/workers/engine/fsdp/transformer_impl.py
if self.engine_config.ulysses_sequence_parallel_size > 1 and not self.use_remove_padding:
    raise ValueError(
        "When using sequence parallelism (ulysses_sequence_parallel_size > 1), "
        "you must enable `use_remove_padding`."
    )

# 创建设备 mesh
dp_size = self.get_data_parallel_size()
if self.ulysses_sequence_parallel_size > 1:
    self.ulysses_device_mesh = init_device_mesh(
        device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
    )
    self.ulysses_parallel_group = self.ulysses_device_mesh["sp"].get_group()
```

**重要约束**：使用 Ulysses 序列并行时，**必须启用 `use_remove_padding`**。因为序列并行需要对齐的序列长度，而 padding 会破坏这种对齐。

**适用场景**：
- 训练长上下文模型（如 32K、64K、128K tokens）
- 显存瓶颈主要来自激活值而非参数
- 批次大小较小，但序列很长

**Trade-off**：
- ✅ 支持超长序列训练
- ✅ 降低单卡激活值显存占用
- ❌ 增加序列并行通信开销（All-Gather 输出）
- ❌ 需要 `use_remove_padding=True`，增加了数据预处理复杂度

**配置建议**：
- **短序列（≤8K）**：保持 `1`，无需序列并行
- **长序列（≥32K）**：根据序列长度和 GPU 数设置，常见值为 2、4、8
- **取值参考**：`ulysses_sequence_parallel_size` 应该使得 `seq_len / sp_size` 后的单卡序列长度在 4K~8K 范围内
  - 例如 64K 序列，设 `sp_size=8`，则每卡处理 8K tokens
  - 例如 32K 序列，设 `sp_size=4`，则每卡处理 8K tokens
- **必须同时设置**：`use_remove_padding: true`

---

### 3.4 熵计算优化参数组

#### 3.4.1 entropy_from_logits_with_chunking

**参数路径**：`actor_rollout_ref.actor.entropy_from_logits_with_chunking`

**默认值**：`False`

**作用**：在计算策略熵（entropy）时，将 logits **分块（chunk）处理**，降低显存峰值。

**原理详解**：

在 PPO/RLHF 训练中，需要计算策略分布的熵：

```
entropy = -sum(p(x) * log(p(x)))
```

其中 `p(x) = softmax(logits)`。对于大词汇表（如 100K tokens）和长序列，一次性计算所有位置的 softmax 和 log_prob 会产生巨大的中间张量。

```python
# 不分块（默认）
logits: [batch_size, seq_len, vocab_size]  # 如 [4, 4096, 100000]
# 中间张量: [4, 4096, 100000] 的 softmax 输出 → 约 6.4GB

# 分块处理
chunk_size = 1024
for chunk in logits.split(chunk_size, dim=1):
    entropy_chunk = compute_entropy(chunk)  # 每次只处理 [4, 1024, 100000]
# 峰值显存降低为 1/4
```

**在 VERL 中的实现**：

```python
# verl/workers/engine/fsdp/transformer_impl.py
if self.engine_config.entropy_from_logits_with_chunking:
    entropy_from_logits = verl_F.entropy_from_logits_with_chunking
else:
    entropy_from_logits = verl_F.entropy_from_logits

self.compute_entropy_from_logits = (
    torch.compile(entropy_from_logits, dynamic=True)
    if self.engine_config.use_torch_compile
    else entropy_from_logits
)
```

注意：分块计算后还会通过 `torch.compile` 进一步加速（如果 `use_torch_compile=True`）。

**适用场景**：
- 词汇表很大（>50K）
- 序列较长（>2048）
- 训练过程中出现 OOM，且峰值显存出现在熵计算阶段

**Trade-off**：
- ✅ 降低显存峰值，避免 OOM
- ✅ 对最终数值结果无影响（只是改变计算顺序）
- ❌ 略微增加计算时间（分块循环的开销）

**配置建议**：
- **OOM 且怀疑是熵计算导致**：设为 `True`
- **显存充裕**：保持 `False`（默认），避免不必要的分块开销
- **与 `entropy_checkpointing` 的区别**：`chunking` 是分块计算，`checkpointing` 是重计算，两者可以独立使用

---

#### 3.4.2 entropy_checkpointing

**参数路径**：`actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`

**默认值**：`False`

**作用**：对熵计算启用**梯度检查点（Gradient Checkpointing）**，用计算换显存。

**原理详解**：

梯度检查点（也叫激活值重计算）是一种经典的显存优化技术。标准反向传播需要保存前向传播的中间激活值用于梯度计算；而梯度检查点只保存部分检查点，反向传播时从最近的检查点重新计算所需的激活值。

```
标准反向传播:
Forward:  x → [a1] → [a2] → [a3] → output
                              ↓保存所有激活值
Backward: output ← [a3] ← [a2] ← [a1] ← grad

梯度检查点:
Forward:  x → [a1]* → [a2] → [a3]* → output
          * 只保存检查点
Backward: 从 a1* 重新计算 a2, a3，再算梯度
```

当 `entropy_checkpointing=True` 时，VERL 对熵计算函数应用 `torch.utils.checkpoint.checkpoint`：

```python
# verl/workers/engine/fsdp/transformer_impl.py
if not self.engine_config.entropy_checkpointing:
    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
else:
    entropy_rmpad = torch.utils.checkpoint.checkpoint(
        self.compute_entropy_from_logits, logits_rmpad
    )
```

**适用场景**：
- 熵计算的激活值占用大量显存
- 可以接受额外的计算开销来换取显存

**Trade-off**：
- ✅ 显著降低熵计算相关的激活值显存
- ✅ 对最终梯度结果无影响
- ❌ 增加约 20-30% 的计算时间（需要重计算）

**配置建议**：
- **显存紧张**：设为 `True`
- **追求速度**：保持 `False`
- **可以与 `entropy_from_logits_with_chunking` 同时使用**：两者从不同角度优化显存，互不冲突

### 3.5 调试与确定性参数组

#### 3.5.1 forward_only

**参数路径**：`actor_rollout_ref.actor.fsdp_config.forward_only`

**默认值**：`False`

**作用**：标记当前模块**只进行前向传播**，不构建计算图、不计算梯度、不创建优化器。

**原理详解**：

在 RLHF 训练中，不同角色有不同的计算需求：

| 角色 | 需要反向传播 | 需要优化器 | 典型配置 |
|-----|-----------|-----------|---------|
| Actor | ✅ | ✅ | `forward_only: false` |
| Critic | ✅ | ✅ | `forward_only: false` |
| Reference | ❌ | ❌ | `forward_only: true` |
| Rollout | ❌ | ❌ | `forward_only: true` |

Reference 模型和 Rollout 模型在 RLHF 中只用于生成文本或计算参考 log_prob，不需要更新参数。因此可以标记为 `forward_only=True`，跳过所有与训练相关的设置。

**在 VERL 中的实现**：

```python
# verl/workers/engine/fsdp/transformer_impl.py

# 1. 模型数据类型
if torch_dtype is None:
    torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

# 2. FSDP 包装
if self.engine_config.strategy == "fsdp":
    cpu_offload = None
    if self.engine_config.forward_only:
        cpu_offload = CPUOffload(offload_params=True)
        self._is_offload_param = False
        self._is_offload_optimizer = False

# 3. FSDP2 包装
elif self.engine_config.strategy == "fsdp2":
    if self.engine_config.offload_policy or self.engine_config.forward_only:
        offload_policy = CPUOffloadPolicy(pin_memory=True)
        self._uses_fsdp2_cpu_offload_policy = True

# 4. QAT 跳过
if self._qat_enabled and not self.engine_config.forward_only:
    module = self._apply_qat(module)

# 5. 优化器跳过
if not self.engine_config.forward_only:
    optimizer = self._build_optimizer(module)
    lr_scheduler = self._build_lr_scheduler(optimizer)

# 6. to() 方法跳过
if self.engine_config.forward_only:
    return  # 跳过 offload 逻辑
```

**关键行为变化**：

1. **数据类型**：`forward_only=True` 时默认使用 `bfloat16`（不需要 fp32 精度做梯度计算）
2. **CPU Offload**：自动启用参数卸载（节省显存，因为不需要保留梯度）
3. **QAT 跳过**：不对推理模型做量化感知训练
4. **优化器跳过**：不创建优化器和学习率调度器
5. **to() 跳过**：跳过自定义的 offload 逻辑（因为 FSDP 已经处理了）

**适用场景**：
- **Reference 模型**：提供参考策略的 log_prob，用于 KL 散度约束
- **Rollout 模型**：生成训练数据（回答文本）
- **纯推理任务**：只需要模型输出，不需要训练

**Trade-off**：
- ✅ 显著降低显存（不保存梯度、不创建优化器状态）
- ✅ 自动启用 CPU offload，进一步节省显存
- ✅ 可以使用更低的精度（bfloat16）
- ❌ 如果误设为 `True`，模型不会更新（训练无效果）

**配置建议**：
- **Actor/Critic**：必须 `False`，否则不训练
- **Reference/Rollout**：设为 `True`，节省显存
- **注意**：VERL 的 `forward_only` 会自动强制关闭 `param_offload` 和 `optimizer_offload`（因为 FSDP 的 CPUOffload/CPUOffloadPolicy 已经处理了卸载）

---

#### 3.5.2 full_determinism

**参数路径**：`actor_rollout_ref.actor.fsdp_config.full_determinism`

**默认值**：`False`

**作用**：启用**完全确定性（Full Determinism）**，确保分布式训练的每次运行产生完全相同的数值结果。

**原理详解**：

在分布式训练中，由于浮点数加法的非结合性（`(a + b) + c ≠ a + (b + c)`），不同顺序的梯度聚合可能产生略微不同的结果。此外，CUDA 中的某些操作（如卷积、矩阵乘法）使用了非确定性的算法来优化速度。

当 `full_determinism=True` 时，VERL 调用 `enable_full_determinism()` 函数：

```python
# verl/workers/engine/fsdp/transformer_impl.py
if self.engine_config.full_determinism:
    enable_full_determinism(seed=self.engine_config.seed)
```

这个函数会设置一系列 PyTorch 确定性标志：

```python
# verl/workers/engine/utils.py
def enable_full_determinism(seed=42):
    """Enable full determinism for reproducible distributed training."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 启用 CUDNN 确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 启用 CUDA 确定性算法
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
```

**关键设置**：

1. **`torch.manual_seed()`**：固定 PyTorch 的随机种子
2. **`cudnn.deterministic=True`**：使用确定性的卷积算法
3. **`cudnn.benchmark=False`**：禁用自动寻找最快算法的 benchmark
4. **`CUBLAS_WORKSPACE_CONFIG`**：为确定性矩阵乘法分配固定大小的 workspace
5. **`torch.use_deterministic_algorithms(True)`**：强制所有操作使用确定性算法

**适用场景**：
- **调试**：排查数值问题，需要每次运行结果一致
- **论文复现**：确保实验结果可重复
- **A/B 测试**：比较不同配置时，需要排除随机性干扰

**Trade-off**：
- ✅ 完全可复现的训练结果
- ✅ 便于调试数值问题
- ❌ **显著降低训练速度**（可能慢 10-30%）
- ❌ 某些操作可能没有确定性实现，会报错

**配置建议**：
- **正常训练**：保持 `False`（默认）
- **调试阶段**：设为 `True`，定位问题后再关闭
- **论文投稿前**：运行一次 `True` 验证结果可复现
- **注意**：仅用于调试，**不要在生产训练中使用**

---

## 4. 参数配置速查表

### 4.1 按场景推荐配置

#### 场景 A：显存紧张，训练大模型（如 70B）

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      param_offload: false      # Actor 训练不能开，会出错
      optimizer_offload: true   # 优化器状态放 CPU，省显存
      offload_policy: false     # FSDP2 原生卸载（可选）
      reshard_after_forward: true  # 默认，省显存
      fsdp_size: -1             # 全部分片
      forward_prefetch: false   # 显存紧张不开
      use_orig_params: false    # 全参数训练
      entropy_from_logits_with_chunking: true   # 省显存
      entropy_checkpointing: true               # 省显存
      forward_only: false       # Actor 要训练
      full_determinism: false   # 正常训练不开
    ulysses_sequence_parallel_size: 1
```

#### 场景 B：追求速度，显存充裕（如 7B 模型，8x A100）

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      param_offload: false
      optimizer_offload: false  # 优化器放 GPU，速度快
      reshard_after_forward: false  # 用显存换速度
      fsdp_size: -1
      forward_prefetch: false   # FSDP2 无效
      use_orig_params: false
      entropy_from_logits_with_chunking: false
      entropy_checkpointing: false
      forward_only: false
      full_determinism: false
    ulysses_sequence_parallel_size: 1
```

#### 场景 C：Reference/Rollout 模型（纯推理）

```yaml
actor_rollout_ref:
  ref:
    strategy: fsdp2
    fsdp_config:
      param_offload: false      # forward_only 会自动处理卸载
      optimizer_offload: false
      reshard_after_forward: false  # 推理不需要反向，保持完整参数更快
      fsdp_size: -1
      forward_only: true        # 关键！标记纯推理
      full_determinism: false
    ulysses_sequence_parallel_size: 1
```

#### 场景 D：长上下文训练（32K+ tokens）

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      param_offload: false
      optimizer_offload: true
      reshard_after_forward: true
      fsdp_size: -1
      entropy_from_logits_with_chunking: true
      entropy_checkpointing: true
      forward_only: false
      full_determinism: false
    ulysses_sequence_parallel_size: 4   # 序列并行
    use_remove_padding: true            # 必须同时开启！
```

#### 场景 E：LoRA 微调

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp  # 或 fsdp2
    fsdp_config:
      param_offload: false
      optimizer_offload: false   # LoRA 参数少，优化器状态也小
      reshard_after_forward: true
      fsdp_size: -1
      forward_prefetch: false
      use_orig_params: true      # LoRA 必须开启！
      forward_only: false
      full_determinism: false
    ulysses_sequence_parallel_size: 1
```

### 4.2 参数快速对照表

| 参数 | 默认值 | FSDP1 | FSDP2 | 主要作用 |
|-----|-------|-------|-------|---------|
| `param_offload` | `False` | ✅ | ✅ | 参数卸载到 CPU |
| `optimizer_offload` | `False` | ✅ | ✅ | 优化器状态卸载到 CPU |
| `offload_policy` | `False` | ❌ | ✅ | FSDP2 原生卸载 |
| `reshard_after_forward` | `True` | ✅ | ✅ | 前向后重新分片 |
| `fsdp_size` | `-1` | ✅ | ✅ | 分片组大小 |
| `forward_prefetch` | `False` | ✅ | ❌ | 前向预取 |
| `use_orig_params` | `False` | ✅ | ❌ | 使用原始参数 |
| `ulysses_sequence_parallel_size` | `1` | ✅ | ✅ | 序列并行大小 |
| `entropy_from_logits_with_chunking` | `False` | ✅ | ✅ | 分块计算熵 |
| `entropy_checkpointing` | `False` | ✅ | ✅ | 熵计算重计算 |
| `forward_only` | `False` | ✅ | ✅ | 仅前向传播 |
| `full_determinism` | `False` | ✅ | ✅ | 完全确定性 |

---

## 5. 常见问题与排错

### 5.1 FSDP1 vs FSDP2 如何选择？

| 维度 | 推荐选择 |
|-----|---------|
| PyTorch < 2.4 | 只能用 FSDP1 (`strategy: fsdp`) |
| PyTorch ≥ 2.4 | 推荐 FSDP2 (`strategy: fsdp2`) |
| 使用 `torch.compile` | FSDP2 兼容性更好 |
| 需要 `forward_prefetch` | 只能用 FSDP1 |
| 需要 `offload_policy` | 只能用 FSDP2 |
| 使用 LoRA | 两者都可以，FSDP2 更稳定 |

### 5.2 OOM（显存溢出）排查清单

遇到 OOM 时，按以下顺序检查和调整：

1. **确认 `reshard_after_forward=True`**
   - 默认就是 `True`，但如果你手动改过，请优先改回 `True`
   - 这是 FSDP 最基础的显存优化，前向结束后立即释放完整参数

2. **启用 `optimizer_offload=True`**
   - 优化器状态占模型状态的约 2/3（Adam 的 momentum + variance）
   - 开启后，这部分从 GPU 搬到 CPU，**模型状态显存降低约 2/3**
   - 注意：参数和梯度仍在 GPU，总显存降低幅度取决于激活值占比

3. **启用 `entropy_from_logits_with_chunking=True`**
   - 降低熵计算阶段的显存峰值，对大词汇表和长序列效果显著

4. **启用 `entropy_checkpointing=True`**
   - 用额外的计算开销换取熵计算相关的激活值显存

5. **减小批次大小或序列长度**
   - 最直接有效，但会影响训练效果和收敛速度

6. **启用 `ulysses_sequence_parallel_size`**（序列长度 ≥ 8K 时推荐）
   - 将序列沿长度维度分片到多卡，降低单卡激活值显存
   - **必须同时设置 `use_remove_padding: true`**

7. **考虑 `param_offload=True`**
   - ⚠️ **仅用于 Reference/Rollout 模型，Actor 绝对不能开**（会导致梯度累积错误）

### 5.3 参数冲突与注意事项

#### ⚠️ Actor 模型不要开 `param_offload`

VERL 代码注释明确说明：
> "We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation"

Actor 训练时开启参数卸载会导致梯度累积结果错误。

#### ⚠️ `ulysses_sequence_parallel_size > 1` 时必须开 `use_remove_padding`

否则会报错：
```
ValueError: When using sequence parallelism (ulysses_sequence_parallel_size > 1), you must enable `use_remove_padding`.
```

#### ⚠️ `forward_only=True` 会自动覆盖 offload 设置

当 `forward_only=True` 时，VERL 会自动：
- 设置 `cpu_offload = CPUOffload(offload_params=True)`（FSDP1）
- 或设置 `offload_policy = CPUOffloadPolicy(pin_memory=True)`（FSDP2）
- 强制 `param_offload = False` 和 `optimizer_offload = False`

不要手动为 Reference/Rollout 设置 `param_offload=True`，让 `forward_only=True` 自动处理。

#### ⚠️ `fsdp_size` 必须整除总 GPU 数

例如 8 张 GPU，`fsdp_size` 可以是 1、2、4、8，但不能是 3、5、6、7。

### 5.4 如何验证配置生效？

1. **查看日志**：VERL 启动时会打印 FSDP 配置信息
2. **监控显存**：使用 `nvidia-smi` 或 `torch.cuda.memory_summary()`
3. **检查 FSDP 包装**：打印模型结构，确认层被正确包装

```python
# 检查 FSDP 包装
for name, module in model.named_modules():
    if isinstance(module, FSDP):
        print(f"FSDP wrapped: {name}")
```

### 5.5 性能调优建议

1. **先确保能跑起来**：从保守配置开始（默认参数）
2. **逐步关闭优化**：显存充裕时，逐步关闭 `optimizer_offload`、`entropy_checkpointing` 等
3. **监控通信开销**：如果 GPU 利用率低但通信高，考虑调整 `fsdp_size`
4. **使用混合精度**：确保 `dtype: bfloat16` 已开启（默认）
5. **启用 `torch.compile`**：VERL 默认开启，可显著加速训练

---

*文档版本：1.0*
