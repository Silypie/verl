# Rollout Correction 晨会分享稿

> 分享时长：15 分钟（13 min 内容 + 2 min Q&A）
> 听众：组内 RL 算法工程师
> 资料来源：<https://verl.readthedocs.io/en/latest/algo/rollout_corr.html>

---

## 1. Why: LLM-RL 的 off-policy 困境  *(3 min)*

LLM-RL 训练里有一个常见但隐蔽的假设错误——**naive PPO 把 rollout policy 当成 old policy**。

### 1.1 教科书里的 PPO 假设

PPO 的目标函数里隐含一条假设：**`π_old` = data collection policy**。只要满足这条，importance ratio `π_θ/π_old` 才有意义，clipping 也才稳定。

### 1.2 LLM-RL 里这条假设几乎不成立

rollout 阶段和 training 阶段用的是**不同实现**：

| drift 来源 | 例子 |
|---|---|
| 数值精度 | rollout 用 FP8/BF16，training 用 FP32 |
| 推理后端 | rollout 用 vLLM / SGLang，training 用 FSDP / Megatron |
| 异步 staleness | rollout worker 拿到的是 N 步前的 checkpoint |
| 训练-推理的算子差异 | 同样的权重、不同的实现，log_prob 也可能不一致 |

只要 rollout 和 training 哪怕只差一个算子或一个 dtype，`π_rollout ≠ π_old`，那 `π_θ/π_old` 这个 ratio **不再是无偏估计**。

### 1.3 后果

- 训练初期 KL 看起来正常，后期突然发散
- reward 提升到某个值后 crash 回退（policy collapse）
- 调 PPO clip / KL coeff 等超参都救不回来
- "训练-推理一致性"成了新工程的玄学问题

verl 团队的 *When Speed Kills Stability* 博客把这个问题叫 **RL collapse from training-inference mismatch**，并由此催生了 Rollout Correction 框架。

### 1.4 解决思路

需要在 framework 层面**显式建模** `π_rollout`、`π_old`、`π_θ` 三个 policy（或把 `π_rollout = π_old` 当 2 policy），并对数据分布做 importance correction。

→ 下一节：verl 的 Rollout Correction 是什么。

---

## 2. What: Rollout Correction 是什么  *(2 min)*

verl 给出的方案是 **Rollout Correction (RC)**：一个统一框架，显式处理 rollout ↔ training 之间的任何分布漂移。**任何 off-policy 场景都适用**，不限于上面那 4 类。

### 2.1 两个正交的机制

| 机制 | 输出 | 作用 | 名字 |
|---|---|---|---|
| **IS Weights** | `rollout_is_weights`（连续浮点） | 梯度再加权 | variance reduction |
| **Rejection Sampling** | `modified_response_mask`（0/1） | 硬过滤极端样本 | trust region |

两者**互相独立**，可以单独开 / 同时开 / 都不开。

### 2.2 核心公式

Decoupled 模式下，每个 token 的 importance weight：

```
ρ_t = π_old(t) / π_rollout(t)
```

> **Bypass 模式 ≠ 把 π_rollout 直接当 π_old 用。** Bypass 是**显式声明** `π_old := π_rollout`，因此公式变为：
>
> ```
> ρ_t = π_θ(t) / π_rollout(t)
> ```
>
> 这个 ratio 反映的是 **θ 与当前 rollout 版本之间的偏离**。当 rollout ↔ training mismatch 很小，它退化成标准 PPO；mismatch 大时它会**偏保守**（clipping 更激进），这本身有 trust-region 效果——是一个**有意的设计权衡**，不是"含义一样"。

### 2.3 安全性设计（防数值爆炸）

- 每个 ratio clamp 到 `[exp(-20), exp(20)] ≈ [2e-9, 5e8]`
- 上界再做 `.clamp(max=rollout_is_threshold)`（TIS 截断）
- padding 位置乘 `response_mask` 清零
- 内存开销 ~1%，计算开销 1–3%

> **TIS 是经典 Truncated Importance Sampling 的实现形式**，`rollout_is_threshold` 是个 **bias-variance 旋钮**：
> - 调紧（→1.0~1.5）：保留的样本少、bias 大、variance 小
> - 调松（→3.0~5.0）：保留的样本多、bias 小、variance 大
> - 起点 `2.0` 是 token 级社区经验值，sequence 级可放宽到 10.0。`2.0` 不是 magic number，是从经验分布里取的合理中位数。

> 设计哲学一句话：**IS 做"软修正"压方差，RS 做"硬过滤"砍 outlier。**

→ 下一节：怎么把这些机制拼起来用。

---

## 3. How: 核心概念  *(3 min)*

三个维度可以独立组合，构成了 RC 的全部自由度。

### 3.1 维度一：Operating Mode（怎么算 `π_old`）

| Mode | Policy 数 | 含义 | 代价 |
|---|---|---|---|
| **Decoupled** | 3 个：π_rollout, π_old, π_θ | 单独再 forward 一遍算 `old_log_prob` | 多一次前向 |
| **Bypass** | 2 个：π_rollout = π_old, π_θ | 直接拿 rollout 的 log_prob 当 old | 省一次前向 |

> Bypass 不是"偷懒"，是显式声明 `π_rollout` 就是我的 proximal policy。

### 3.2 维度二：Loss Type（bypass 模式下选哪个目标）

| Loss | 含义 | 何时用 |
|---|---|---|
| `ppo_clip`（默认） | PPO clipped objective，IS 由 ratio 隐式处理 | 大多数场景；想"开箱即用、不踩坑"时 |
| `reinforce` | 纯策略梯度，IS 权重显式乘到 gradient 上 | 想**显式控制 IS 强度**（比如想观察 IS 分布对训练的影响、做 ablation）；off-policy 漂移很大、需要 IS 强信号时 |

### 3.3 维度三：聚合粒度（IS/RS 怎么算）

| 粒度 | 含义 | 特点 |
|---|---|---|
| `token` | 每个 token 单独算 ratio | 低方差，但对 outlier 不敏感 |
| `sequence` | 一条 sequence 内连乘 | 敏感于 outlier，常需更高 threshold |
| `geometric` | sequence level 的几何平均（log-mean） | 在 K1/K2/K3 估计量下更稳 |

粒度**与 mode 正交**：token/sequence 都能配 decoupled 或 bypass。

### 3.4 一张决策速查图

```
                        ┌─ 想快？ ─→ Bypass + ppo_clip
              ┌─ Mode ──┤
              │         └─ 想严格？ ─→ Decoupled
Operating ───┤
              │              ┌─ 大多数 ─→ ppo_clip
              └─ Loss type ─┤
                             └─ 想显式 IS ─→ reinforce
```

→ 下一节：把这些维度压成一个 preset。

---

## 4. Presets 与推荐工作流  *(3 min)*

verl 把上面的维度组合打成了一组**验证过的 preset**，开箱即用。

### 4.1 Python API 入口

```python
from verl.trainer.config.algorithm import RolloutCorrectionConfig

# Step 1 — metrics-only：先观测 off-policy 程度
config = RolloutCorrectionConfig.disabled()

# Step 2 — 加 RS：硬过滤极端样本
config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()

# Step 3 — 完整 IS：剩余样本做 importance correction
config = RolloutCorrectionConfig.bypass_pg_geo_rs_token_tis()
```

### 4.2 最小 YAML 配置示例

```yaml
algorithm:
  rollout_correction:
    rollout_is: token              # "token" | "sequence" | null
    rollout_is_threshold: 2.0      # TIS 上界
    rollout_is_batch_normalize: false
    rollout_rs: null               # 如 "seq_mean_k3" / "token_k1"
    rollout_rs_threshold: null
    bypass_mode: true              # 推荐 true
    loss_type: ppo_clip            # 或 "reinforce"

actor_rollout_ref:
  rollout:
    calculate_log_probs: true      # 必开：让 rollout 端额外算 log-prob
  actor:
    use_rollout_log_probs: true    # 必开：actor 端使用上面这个 log-prob
    policy_loss:
      loss_mode: bypass_mode       # 必开：actor loss 切换到 bypass 公式
      rollout_correction:           # 这里是 override，字段同 algorithm.rollout_correction
        rollout_is: token
        rollout_is_threshold: 2.0
        bypass_mode: true
        loss_type: ppo_clip
```

> **两个 RC 配置块的关系**：`algorithm.rollout_correction` 是 default；`actor_rollout_ref.actor.policy_loss.rollout_correction` 是 override。两者字段相同，**任一处生效**，但**3 个开关必须同时设置正确**：`calculate_log_probs`、`use_rollout_log_probs`、`loss_mode: bypass_mode`，缺一会**静默退化为无效配置**（不会有告警），训练表面上跑通但 RC 没生效。

### 4.3 Preset 速查表

| Preset | Mode | Loss | IS | RS | 何时用 |
|---|---|---|---|---|---|
| `bypass_ppo_clip` | Bypass | ppo_clip | — | — | 最快，没 RS 兜底 |
| `bypass_ppo_clip_geo_rs` | Bypass | ppo_clip | — | seq_mean_k1¹ | **推荐起点** |
| `bypass_pg_geo_rs_token_tis` | Bypass | reinforce | token | seq_mean_k1¹ | 想显式控制 IS |
| `decoupled_seq_is` | Decoupled | ppo | seq | — | 严格对齐老 PPO 行为 |
| `decoupled_geo_rs_token_tis` | Decoupled | ppo | token | seq_mean_k1¹ | 严格 + token 精度 |
| `disabled` | — | — | — | — | 只看 metric，不做修正 |

> ¹ RS 列的 `geo` 是速记写法，全名是 `seq_mean_k1`（geometric 粒度 = sequence mean + K1 estimator）。完整的 RS 模式命名规则见 §3.3 / 官方文档 §rollout_rs。

### 4.4 三步走工作流

```
Step 1: metrics-only
   rollout_is=null, rollout_rs=null
   → 看 rollout_corr/kl, chi2_token, is_mean
   → 判断 off-policy 到底有多严重

Step 2: 加 RS
   rollout_rs=seq_mean_k1, threshold="0.5_2.0"  # ratio 上下界
   → 把最离谱的样本砍掉
   → 看 rollout_rs_masked_fraction 是否合理（< 20%）
   → RS 阈值用 K1 estimator 风格 (lower_upper)："0.5_2.0" 意为 ratio 落在 [0.5, 2.0] 外的样本被砍

Step 3: 完整 IS
   rollout_is=token, threshold=2.0
   loss_type=reinforce（可选）
   → 剩余样本做 importance correction
```

> 原则：**永远不要跳过 Step 1**，先看再修。盲目开全量 IS 反而会引入额外方差。

→ 下一节：怎么读 metric 知道训练健不健康。

---

## 5. Diagnostics: 关键 metric 与健康阈值  *(2 min)*

所有 metric 都带 `rollout_corr/` 前缀，log 到 wandb / tensorboard。

### 5.1 5 个核心指标

| Metric | 健康值 | 含义 |
|---|---|---|
| `rollout_is_mean` | ≈ 1.0 | 平均 IS 权重（按 token level 聚合的 ratio 均值）。偏离 1 越远，off-policy 越严重 |
| `rollout_is_eff_sample_size` | > 0.3 | `1 / mean(w²)`，有效样本占比。越低说明权重越集中在少数样本 |
| `rollout_is_std` | < 1.0 | IS 权重标准差 |
| `kl` | \|kl\| < 0.1 | KL(π_rollout ‖ π_train)，可正可负 |
| `chi2_token` | < 1.0 | token-level χ² 散度（E[ρ²] - 1），> 1 表示严重漂移 |

### 5.2 3 个告警规则

```python
if rollout_is_mean < 0.5 or rollout_is_mean > 2.0:
    warn("off-policy gap 太大，检查 calculate_log_probs 和数据流")

if rollout_is_eff_sample_size < 0.3:
    warn("有效样本不足，权重太集中，考虑收紧 threshold 或切到 geometric")

if chi2_token > 1.0:
    warn("token 级别分布漂移严重，先只开 RS 别开 IS")
```

### 5.3 常见症状速查

| 症状 | 根因 | 建议 |
|---|---|---|
| `is_mean` 偏离 1.0 | 见下方「is_mean 偏离的 5 个根因」 | 按表排查 |
| `is_std` 大、`ess` 小 | sequence-level outlier 多 | 切 geometric，或收紧 threshold |
| `kl` 飘 | rollout 真的和 training 差太远 | 先 RS 砍极端 sample，必要时开 IS |
| `rs_masked_fraction` > 30% | 阈值太严了 | 放宽 `rollout_rs_threshold` |

#### `is_mean` 偏离 1.0 的 5 个常见根因

1. **`calculate_log_probs: true` 没开** → 检查 `actor_rollout_ref.rollout` 配置
2. **rollout dtype ≠ training dtype**（vLLM BF16 vs FSDP FP32）→ 量化对 log-prob 的影响
3. **异步 staleness** → 检查 rollout worker 拿到的 checkpoint 步数
4. **prompt 分布漂移** → 当前 batch 难度和历史 batch 差距大
5. **训练初期 on-policy warm-up** → 头 100 步正常会偏离，跑 500 步再看

> **诊断口诀：先 mean 定位"有没有偏"，再 std/ess 定位"偏得均不均匀"，最后 kl/chi2 定位"偏得有多大"。**

→ 最后一节：3 句话带走。

---

## 6. Takeaways + Q&A 引导  *(2 min)*

### 6.1 三句话带走

1. **LLM-RL 里的 "naive PPO" 几乎一定是错的**——只要 rollout 和 training 不是同一份实现，就该显式建模 `π_rollout`。
2. **RC = IS（软修正） + RS（硬过滤）**，两者正交，按需打开。新任务默认从 `bypass_ppo_clip_geo_rs` 起步。
3. **先观测再修**：永远先跑 `disabled()` 看 `kl` / `is_mean` / `chi2_token`，再决定上 RS 还是 IS。

### 6.2 我们要不要用？

判断标准（按顺序回答 3 个问题）：

- [ ] 我们 rollout 用了和 training 不同的后端 / 精度吗？ → **是** → 继续
- [ ] 训练出现过不明原因的 KL 飘、collapse 吗？ → **是** → RC 大概率能帮上
- [ ] 愿意付 1–3% 的计算 + 多 1 个 yaml 块的工程成本吗？ → **是** → 上

### 6.3 预演 Q&A

> **Q1：开了 RC 是不是就不用调 PPO clip / KL coeff 了？**
> 不是。RC 处理的是"rollout ≠ old"这个 mismatch；PPO clip / KL coeff 解决的是 trust region 强度。两个维度正交，都要看。

> **Q2：Bypass 会不会比 Decoupled 差？**
> 表达力有差别。Bypass 把 `π_rollout` 当 proximal policy，省一次前向；Decoupled 严格区分 3 个 policy，能拿到"batch size invariance"等性质。生产里大多数场景 Bypass 够用，但需要严格对齐老 PPO 行为时用 Decoupled。

> **Q3：GRPO / DAPO 也要开吗？**
> 都要开。RC 修的是"rollout 分布"问题，与 base 算法无关。
>
> **和 DAPO `dynamic sampling` 的关系**：DAPO 的 `filter_groups` 在 **group 层 filter**（reward 全 0/全 1 的 prompt group 整组丢弃，目的是给 GRPO-style advantage 留出方差）；RC 的 RS 在 **token/sequence 层 filter**（按 ratio 砍 IS 权重过大的样本）。两者**作用对象和阶段都不同**，可以叠加——参考 `recipe/dapo/run_dapo_qwen2.5_32b_rollout_corr.sh`。
>
> **坑**：当 RS 砍掉 30%+ 样本时，DAPO 的 `max_num_gen_batches` 可能反复触发重采样、消耗额外 rollout。两种缓解：调高 `max_num_gen_batches`，或在 IS 漂移特别大的训练初期临时关掉 `filter_groups`。

> **Q4：threshold 怎么选？**
> 起点 `rollout_is_threshold=2.0`、`rollout_rs_threshold="0.5_2.0"`（即 K1 比值上下界）。看 `rs_masked_fraction` 和 `is_eff_sample_size` 再调。

---

## 附录：参考资源

- **官方文档（本稿主线）**：<https://verl.readthedocs.io/en/latest/algo/rollout_corr.html>
- **数学公式与推导**：<https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html>
- **主博客 *When Speed Kills Stability***：<https://richardli.xyz/rl-collapse>
- **博客 Part 1（TV 距离 / χ² 分析框架）**
- **博客 Part 2（token vs sequence 的 bias-variance 权衡）**
- **博客 Part 3（toxic tail / length trap 为何选 RS）**
- **最新论文**：arXiv:2512.23075 — Trust Region Masking for Long-Horizon LLM RL
- **示例代码**：
  - `examples/rollout_correction/run_qwen2_5_7b_fsdp.sh`
  - `examples/rollout_correction/run_qwen2_5_7b_fsdp_multi_rs.sh`
  - `recipe/dapo/run_dapo_qwen2.5_32b_rollout_corr.sh`（DAPO + RC）
- **核心实现**：
  - `verl/trainer/ppo/rollout_corr_helper.py` — IS/RS 计算主入口
  - `verl/trainer/ppo/core_algos.py` — bypass / reinforce 损失
  - `verl/trainer/config/algorithm.py` — `RolloutCorrectionConfig`
