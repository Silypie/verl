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

[待写：统一框架总览、IS 与 RS 两大机制、核心公式 ρ = π_old/π_rollout]

---

## 3. How: 核心概念  *(3 min)*

[待写：Decoupled (3-policy) vs Bypass (2-policy)、聚合粒度 token/sequence/geometric、loss_type ppo_clip/reinforce]

---

## 4. Presets 与推荐工作流  *(3 min)*

[待写：RolloutCorrectionConfig API、推荐 preset、最小配置示例、三步走工作流]

---

## 5. Diagnostics: 关键 metric 与健康阈值  *(2 min)*

[待写：rollout_is_mean、kl、chi2_token、推荐阈值与告警规则]

---

## 6. Takeaways + Q&A 引导  *(2 min)*

[待写：3 个核心 takeaway、参考资源、可能的 Q&A 预演]

---

## 附录：参考资源

[待写：官方文档、博客系列、数学公式文档、相关 PR/issue]
