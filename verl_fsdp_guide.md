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

[内容待填充]

---

## 2. FSDP 核心原理

[内容待填充]

---

## 3. VERL FSDP 参数详解

### 3.1 内存优化参数组

#### 3.1.1 param_offload

[内容待填充]

#### 3.1.2 optimizer_offload

[内容待填充]

#### 3.1.3 offload_policy

[内容待填充]

#### 3.1.4 reshard_after_forward

[内容待填充]

### 3.2 性能优化参数组

#### 3.2.1 forward_prefetch

[内容待填充]

#### 3.2.2 use_orig_params

[内容待填充]

#### 3.2.3 fsdp_size

[内容待填充]

### 3.3 序列并行参数

#### 3.3.1 ulysses_sequence_parallel_size

[内容待填充]

### 3.4 熵计算优化参数组

#### 3.4.1 entropy_from_logits_with_chunking

[内容待填充]

#### 3.4.2 entropy_checkpointing

[内容待填充]

### 3.5 调试与确定性参数组

#### 3.5.1 forward_only

[内容待填充]

#### 3.5.2 full_determinism

[内容待填充]

---

## 4. 参数配置速查表

[内容待填充]

---

## 5. 常见问题与排错

[内容待填充]

---

*文档版本：1.0*
