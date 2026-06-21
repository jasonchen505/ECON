# LLM & Agent 算法实习面试准备指南

> 基于 ECON (Efficient Coordination via Nash Equilibrium) 框架的深度技术准备
> 适用岗位：LLM算法实习、Agent应用、后训练(Post-training)相关

---

## 一、项目概述与核心思想

### 1.1 项目一句话介绍（面试自我介绍用）

> "我参与了 ECON 框架的开发，这是一个基于贝叶斯纳什均衡的多智能体 LLM 推理框架。核心思想是用隐式的信念驱动协调替代传统多智能体辩论中的显式消息传递，通过将 LLM 的采样参数（temperature, repetition_penalty）建模为连续动作空间，利用 MARL 中的 QMIX 架构和 BNE 理论实现多智能体的高效协调，最终在数学推理任务上实现了更好的准确率和更低的 token 消耗。"

### 1.2 解决的核心问题

| 传统多智能体辩论(MAD)问题 | ECON 的解决方案 |
|--------------------------|-----------------|
| 显式消息传递导致 token 消耗巨大 | 用 128 维信念向量替代文本通信 |
| 没有收敛保证 | 提供贝叶斯纳什均衡的理论收敛保证 |
| 信息交换容易超出 LLM 上下文限制 | 层次化架构，本地到全局，尊重上下文限制 |

### 1.3 核心创新点（面试高频考点）

1. **Prompt 参数作为连续动作空间**：不直接选择答案，而是控制 temperature 和 repetition_penalty
2. **QMIX 风格的混合网络**：保证单调性约束，实现去中心化执行
3. **Commitment 作为协调信号**：Coordinator 生成的承诺替代显式消息传递
4. **跨 episode 的记忆机制**：GRU 隐藏状态在 episode 间传递，实现时间信用分配

---

## 二、技术架构深度解析

### 2.1 整体架构（必须能画出来）

```
输入: 数学问题
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Coordinator LLM (Llama-3.1-70B)                        │
│  ├── 策略生成: 问题 → step-by-step 策略 (≤80 tokens)     │
│  └── 承诺聚合: N个回复 → 结构化 JSON commitment           │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: 个体信念形成                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │ Agent 1  │ │ Agent 2  │ │ Agent 3  │                │
│  │ BPN      │ │ BPN      │ │ BPN      │  (独立参数)     │
│  └──────────┘ └──────────┘ └──────────┘                │
│       │            │            │                        │
│       ▼            ▼            ▼                        │
│   (b₁,T₁,p₁)  (b₂,T₂,p₂)  (b₃,T₃,p₃)                 │
│       │            │            │                        │
│       └────────────┼────────────┘                        │
│                    ▼                                     │
│            BeliefEncoder (Attention)                     │
│                    │                                     │
│                    ▼                                     │
│              group_repr                                  │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: BNE 协调 (K 轮迭代)                            │
│  for k = 1 to K:                                        │
│    1. 用 (T_i, p_i) 调用 Executor LLM 生成回复           │
│    2. Coordinator 聚合为 commitment                      │
│    3. RefineModule 计算 delta_e                          │
│    4. e_new = e_old + η * delta_e                        │
│    5. 检查收敛 (3 种机制)                                 │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  LLMQMixer: 注意力混合网络                               │
│  q_local [B,T,N] → q_tot [B,T]                         │
│  + consistency_loss + alignment_loss                     │
└─────────────────────────────────────────────────────────┘
      │
      ▼
输出: 最终答案 + 奖励计算
```

### 2.2 核心组件详解

#### 2.2.1 BeliefPolicyNetwork (BPN) - 信念策略网络

**位置**: `src/modules/agents/belief_policy_network.py`

**架构**:
```
token_ids → Embedding → PositionalEncoding → TransformerBlocks (2层, 4头)
    → Pool(last_valid) → GRU (带跨 episode 记忆)
    → 三个头:
        ├── Belief Head → belief_state [128]
        ├── Prompt Head → (T, p) 通过 sigmoid 缩放到 [T_min, T_max], [p_min, p_max]
        └── Q Head → q_value (标量)
```

**关键设计决策**:
- 每个 agent 有**独立参数**（BNE 模式下不共享权重）
- GRU 接收上一个 episode 的记忆向量，实现**跨 episode 的时间信用分配**
- Prompt Head 的输出是连续值，直接控制 LLM 的采样参数

**面试深挖点**:
- Q: 为什么用 GRU 而不是简单的 MLP？
- A: GRU 可以维护跨 episode 的隐藏状态，使得 agent 能够记住之前 episode 的经验，实现更好的时间信用分配。这类似于 Meta-Learning 中的思路。

- Q: 为什么每个 agent 不共享参数？
- A: 在 BNE 框架中，每个 agent 需要有独立的策略来形成纳什均衡。如果共享参数，所有 agent 的策略会趋同，失去多样性，无法达到真正的均衡状态。

#### 2.2.2 BeliefEncoder - 信念编码器

**位置**: `src/modules/belief_encoder.py`

**架构**:
```
belief_states [B, N, d]
    → MultiheadAttention (self-attention)
    → LayerNorm (residual)
    → FeedForward
    → LayerNorm (residual)
    → mean_pool(dim=1)
    → Linear
    → group_repr [B, d]
```

**作用**: 聚合所有 agent 的个体信念，形成群体表示。这个群体表示包含了 agent 间的关系信息，供 Mixer 和 RefineModule 使用。

**面试深挖点**:
- Q: 为什么用 self-attention 而不是简单的 average pooling？
- A: Self-attention 可以学习 agent 之间的依赖关系，自动发现哪些 agent 的信念更相关。例如，如果两个 agent 的信念相似，attention 会给它们更高的权重。

#### 2.2.3 LLMQMixer - 混合网络

**位置**: `src/modules/mixer/mix_llm.py`

**核心公式**:
```
q_tot = MLP(mix_ctx) + Σ(w_i * q_local_i)
```

其中 `w_i = softmax(agent_weight([agent_ctx_i, group_exp]))`

**三个损失项**:
1. **consistency_loss**: `(q_tot - q_local.mean())^2` — QMIX 单调性约束
2. **alignment_loss**: `1 - cosine_sim(mix_ctx, commitment_emb)` — 承诺对齐
3. **主 TD 损失**: `(q_tot_refined - (r + γ * q_tot_target))^2`

**面试深挖点**:
- Q: QMIX 的单调性约束是什么意思？为什么重要？
- A: 单调性约束要求 `∂q_tot/∂q_i ≥ 0`，即个体 Q 值的增加不会导致全局 Q 值下降。这使得去中心化执行成为可能：每个 agent 只需要最大化自己的 Q 值，就能最大化全局 Q 值。

- Q: alignment_loss 的作用是什么？
- A: 它强制 Mixer 的混合上下文与 Coordinator 的承诺对齐。这确保了混合网络的决策与协调信号一致，避免 agent 偏离协调目标。

#### 2.2.4 RefineModule - BNE 精炼模块

**位置**: `src/modules/agents/refine_module.py`

**核心逻辑**:
```python
输入: [belief_i, output_i_emb, commitment_emb, group_repr, e_prev]
    → concat → Linear(256) → ReLU → LayerNorm → Dropout
    → Linear(128) → ReLU → Linear(2) → Tanh
    → delta_e = output * max_delta (0.3)
输出: delta_e (ΔT, Δp) in [-0.3, +0.3]
```

**面试深挖点**:
- Q: 为什么用 Tanh 激活函数？
- A: Tanh 将输出限制在 [-1, 1]，再乘以 max_delta 得到 [-0.3, 0.3] 的范围。这确保了每次调整不会太大，避免震荡。

- Q: 为什么 max_delta 设为 0.3？
- A: 这是一个经验值。太大会导致调整过猛，容易震荡；太小会导致收敛太慢。0.3 在温度范围 [0.2, 1.5] 的约 20%，是一个合理的步长。

### 2.3 BNE 框架的收敛机制

**三种收敛检测**:
1. **承诺收敛**: 文本相等 OR 余弦相似度 ≥ 0.995
2. **参数收敛**: `||delta_e||_2 < 0.08`
3. **震荡检测**: 检测到 A-B-A-B 模式（最近 4 个承诺）

**面试深挖点**:
- Q: 为什么需要三种收敛检测？
- A: 单一检测可能有盲区。例如，承诺可能因为随机性而略有不同（即使参数已收敛），所以需要余弦相似度。参数收敛可能很慢，但承诺可能已经稳定。震荡检测防止在两个等价解之间振荡。

---

## 三、奖励系统与训练算法

### 3.1 三组件奖励系统

**位置**: `src/envs/huggingface_dataset_env.py`

| 组件 | 公式 | 含义 |
|------|------|------|
| TS (Task-Specific) | `1.0 if |pred - gt| < 1e-5 else 0.0` | 答案正确性 |
| AL (Action Likelihood) | `mean(cosine_sim(embed(r_i), embed(commitment)))` | 回复与承诺的一致性 |
| CC (Collaborative Contribution) | `0.7 * fidelity + 0.3 * novelty` | 协作贡献（一致性 + 多样性） |

**最终奖励**: `r = α₁ * AL + α₂ * TS + α₃ * CC`

**动态权重学习**:
```python
alpha = softmax(alpha_logits)
L_alpha = E[(r_combined - q_expectation)^2]
```

**面试深挖点**:
- Q: CC 中为什么要同时考虑 fidelity 和 novelty？
- A: Fidelity 确保 agent 的回复与承诺一致（有用的），novelty 鼓励 agent 提供多样化的视角（不是简单复制）。两者结合避免了"群体思维"（所有人都说一样的话）。

- Q: 为什么要动态学习 alpha 权重？
- A: 不同任务可能需要不同的奖励平衡。例如，数学任务可能更重视 TS（正确性），而创意任务可能更重视 CC（多样性）。动态学习可以自动适应。

### 3.2 训练损失函数

**BNE 模式** (`_train_bne`):

| 损失 | 公式 | 权重 | 作用 |
|------|------|------|------|
| L_TD | `E[(q_tot_refined - (r + γ * q_tot_target))^2]` | 1.0 | 主 TD 学习信号 |
| L_consistency | `E[(q_tot_refined - sum(q_local))^2]` | 0.1 | QMIX 单调性 |
| L_equilibrium | `E[\|\|e_refined - e_init\|\|^2]` | 0.05 | 鼓励稳定均衡 |
| L_commit_improve | `-E[q_tot_refined - q_tot_init]` | 0.1 | 鼓励精炼改进 Q |

**面试深挖点**:
- Q: L_equilibrium 和 L_commit_improve 看起来矛盾，怎么理解？
- A: L_equilibrium 鼓励精炼后的参数不要偏离初始值太远（稳定性），L_commit_improve 鼓励精炼能带来 Q 值提升（改进）。两者平衡了稳定性和改进性。实际中，好的均衡应该是在小调整下获得大改进。

### 3.3 目标网络更新

**软更新** (每 8 步):
```
φ' ← τ * φ + (1-τ) * φ'    其中 τ = 0.01
```

**面试深挖点**:
- Q: 为什么用软更新而不是硬更新？
- A: 软更新使目标网络参数缓慢变化，避免 Q 值估计的剧烈波动，提高训练稳定性。τ=0.01 意味着目标网络每次只更新 1% 的新参数。

---

## 四、面试高频考察点与深度问题

### 4.1 LLM 相关考察点

#### Q1: 如何将 LLM 的采样参数建模为连续动作空间？

**标准答案**:
传统 RL 的动作空间是离散的（如选择 A/B/C/D），但 LLM 的输出受采样参数影响。ECON 将 temperature 和 repetition_penalty 建模为连续动作：
- temperature 控制输出的随机性：低温度 → 更确定性，高温度 → 更多样
- repetition_penalty 控制重复程度：低惩罚 → 允许重复，高惩罚 → 鼓励多样

agent 的策略网络输出连续值 (T, p)，直接控制 LLM 的生成行为。这比离散动作空间更精细，可以实现更微妙的控制。

**追问**: temperature 和 repetition_penalty 具体怎么影响 LLM 的输出？
- Temperature: 在 softmax 之前除以 T，T < 1 使分布更尖锐（更确定），T > 1 使分布更平坦（更随机）
- Repetition Penalty: 对已出现的 token 的 logits 除以惩罚因子，降低其再次被选中的概率

#### Q2: Coordinator LLM 和 Executor LLM 的设计有什么考虑？

**标准答案**:
- **Coordinator (Llama-3.1-70B)**: 用更大的模型，因为需要更强的推理能力来生成策略和聚合承诺。低 temperature (0.3) 和 top_p (0.4) 确保输出稳定、确定性高。
- **Executor (Llama-3.1-8B)**: 用较小的模型，因为需要快速生成大量回复。temperature 和 repetition_penalty 由 agent 的策略网络动态控制。

这种设计体现了"强领导 + 多执行者"的层次化架构。

#### Q3: Commitment 的结构化 JSON 格式有什么好处？

**标准答案**:
```json
{
  "final_value": "42",
  "reasoning": "Step-by-step reasoning...",
  "confidence": 0.85,
  "checklist": ["Check 1", "Check 2"]
}
```

好处：
1. **易于解析**: 结构化格式可以用正则表达式可靠提取答案
2. **信息丰富**: 不只是答案，还有推理过程和置信度
3. **可验证**: checklist 可用于验证推理的完整性
4. **鲁棒性**: 多层解析策略（JSON → boxed → regex → 数字回退）

### 4.2 多智能体 RL 相关考察点

#### Q4: ECON 与 QMIX 的关系是什么？

**标准答案**:
ECON 借鉴了 QMIX 的核心思想：
1. **单调性约束**: Mixer 保证 `∂q_tot/∂q_i ≥ 0`，使去中心化执行成为可能
2. **值分解**: `q_tot = f(q_local, global_state)`，其中 f 是 Mixer
3. **CTDE 范式**: 训练时 Mixer 看到全局信息，执行时每个 agent 独立行动

但 ECON 有重要改进：
- 用注意力机制替代简单的 MLP 混合
- 引入 commitment 对齐损失
- 处理连续动作空间（QMIX 原本处理离散动作）

#### Q5: 什么是 CTDE（Centralized Training, Decentralized Execution）？

**标准答案**:
- **训练时**: 使用全局信息（所有 agent 的信念、承诺等）训练 Mixer，得到全局 Q 值
- **执行时**: 每个 agent 只用自己的局部信息（自己的信念）选择动作

好处：
- 训练时有更多信息，可以学到更好的策略
- 执行时不需要通信，更高效、更鲁棒

#### Q6: 为什么用 Q-learning 而不是 Policy Gradient？

**标准答案**:
1. **样本效率**: Q-learning 可以从历史数据中学习（off-policy），Policy Gradient 需要 on-policy 数据
2. **稳定性**: Q-learning 的目标更稳定（TD 目标），Policy Gradient 的方差更大
3. **连续动作**: 虽然 Q-learning 通常用于离散动作，但 ECON 通过将 prompt 参数作为连续动作，结合 QMIX 的值分解，巧妙地处理了连续动作空间

### 4.3 系统设计相关考察点

#### Q7: 如何保证多智能体系统的可扩展性？

**标准答案**:
ECON 的可扩展性设计：
1. **独立参数**: 每个 agent 有独立的 BPN，不需要共享参数，可以并行计算
2. **层次化架构**: Coordinator 只调用一次（每轮 BNE），不是每个 agent 调用一次
3. **向量通信**: 用 128 维信念向量替代文本通信，大幅降低通信成本
4. **本地计算**: 每个 agent 本地计算自己的 (T, p)，不需要全局协调

实际测试：3 个 agent 的 BNE 只需要 3 轮 LLM 调用（每轮 BNE），而 MAD 需要 O(K*N) 次。

#### Q8: 如何处理 LLM 的上下文长度限制？

**标准答案**:
1. **策略压缩**: Coordinator 生成的策略 ≤80 tokens，远小于完整推理过程
2. **承诺聚合**: 将 N 个回复聚合成一个承诺，而不是传递所有原始回复
3. **向量表示**: 用 128 维信念向量表示 agent 的状态，而不是完整的文本历史
4. **层次化**: 本地 agent 只处理自己的上下文，不需要看到其他 agent 的完整历史

#### Q9: 系统的故障恢复机制是什么？

**标准答案**:
1. **多层解析**: 承诺解析有 4 层回退策略（JSON → boxed → regex → 数字）
2. **收敛检测**: 3 种收敛机制防止无限循环
3. **参数裁剪**: prompt 参数被裁剪到有效范围 [T_min, T_max], [p_min, p_max]
4. **目标网络**: 软更新机制防止 Q 值估计的剧烈波动
5. **检查点**: 定期保存模型，支持从任意点恢复

### 4.4 训练与优化相关考察点

#### Q10: 如何处理奖励稀疏性问题？

**标准答案**:
数学推理任务的奖励非常稀疏（只有最终答案正确才有高奖励）。ECON 的解决方案：
1. **多组件奖励**: 除了 TS（正确性），还有 AL（一致性）和 CC（协作贡献），提供更密集的信号
2. **动态权重**: 自动学习奖励组件的权重，适应不同任务
3. **跨 episode 记忆**: GRU 隐藏状态传递经验，帮助 agent 学习长期策略
4. **承诺对齐**: alignment_loss 提供额外的监督信号

#### Q11: 如何避免多智能体训练中的"懒惰 agent"问题？

**标准答案**:
"懒惰 agent"是指某些 agent 不学习，依赖其他 agent。ECON 的解决方案：
1. **独立参数**: 每个 agent 有独立的 BPN，无法"搭便车"
2. **CC 奖励**: 协作贡献奖励鼓励 agent 提供多样化的贡献
3. **novelty 项**: CC 中的 novelty 鼓励 agent 提供不同于其他 agent 的回复
4. **本地 Q 值**: 每个 agent 有自己的 Q 值，直接反映其贡献

#### Q12: 学习率、折扣因子等超参数如何选择？

**标准答案**:
- **学习率 (0.0004)**: 较小的学习率确保稳定训练，避免 Q 值估计的剧烈波动
- **折扣因子 (0.99)**: 接近 1 的折扣因子重视长期奖励，适合需要多步推理的任务
- **目标网络 τ (0.01)**: 小的 τ 使目标网络缓慢变化，提高稳定性
- **BNE 轮数 (训练 1 轮，推理 3 轮)**: 训练时用少轮数提高效率，推理时用多轮数提高准确率

---

## 五、代码实现细节（需要熟悉的文件）

### 5.1 必须熟悉的文件

| 文件 | 核心内容 | 面试考察频率 |
|------|----------|--------------|
| `src/controllers/basic_mac.py` | Coordinator + BNE 协调逻辑 | ⭐⭐⭐⭐⭐ |
| `src/modules/agents/belief_policy_network.py` | Agent 的策略网络 | ⭐⭐⭐⭐⭐ |
| `src/modules/mixer/mix_llm.py` | QMIX 混合网络 | ⭐⭐⭐⭐ |
| `src/modules/agents/refine_module.py` | BNE 精炼模块 | ⭐⭐⭐⭐ |
| `src/envs/huggingface_dataset_env.py` | 奖励计算 | ⭐⭐⭐⭐ |
| `src/learners/q_learner.py` | 损失函数与训练 | ⭐⭐⭐⭐ |
| `src/modules/belief_encoder.py` | 信念编码器 | ⭐⭐⭐ |

### 5.2 关键代码片段

#### Coordinator 的策略生成 (basic_mac.py:~150)
```python
def _get_strategy_and_format(self, question, ...):
    prompt = f"""Given this math problem: {question}
    Provide a step-by-step strategy (under 80 tokens) without revealing the answer."""
    response = self.llm.generate(prompt, temperature=0.3, top_p=0.4)
    return response
```

#### Agent 的前向传播 (belief_policy_network.py:~80)
```python
def forward(self, obs, memory):
    # Embedding + Transformer
    x = self.embedding(obs)
    x = self.transformer_blocks(x)
    x = self.pool(x)
    # GRU with memory
    x, new_memory = self.gru(x, memory)
    # Three heads
    belief = self.belief_head(x)
    prompt = self.prompt_head(x)  # (T, p)
    q_value = self.q_head(x)
    return belief, prompt, q_value, new_memory
```

#### Mixer 的混合逻辑 (mix_llm.py:~120)
```python
def forward(self, q_local, belief_states, prompt_embeddings, group_repr, commitment_emb):
    # Agent mixing weights
    weights = softmax(self.agent_weight(cat(agent_ctx, group_exp)))
    # Weighted Q aggregation
    q_bar = sum(weights * q_local)
    # Mix context
    mix_ctx = sum(weights * agent_ctx) + group_repr
    # Final Q
    q_tot = self.mix_mlp(mix_ctx) + q_bar
    # Auxiliary losses
    consistency_loss = (q_tot - q_local.mean())^2
    align_loss = 1 - cosine_sim(mix_ctx, proj_commitment)
    return q_tot, consistency_loss, align_loss
```

---

## 六、面试者自我介绍模板

### 6.1 技术深度版（3分钟）

> "面试官您好，我是[姓名]，[学校]的在读硕士，研究方向是[方向]。
>
> 我最近参与了 ECON 框架的开发，这是一个基于贝叶斯纳什均衡的多智能体 LLM 推理框架。项目的核心创新是将 LLM 的采样参数（temperature, repetition_penalty）建模为连续动作空间，利用 MARL 中的 QMIX 架构实现多智能体的高效协调。
>
> 技术上，我主要负责/参与了以下几个部分：
> 1. **信念策略网络的设计**：每个 agent 有独立的 Transformer + GRU 网络，输出信念状态和 prompt 参数，GRU 维护跨 episode 的记忆
> 2. **QMIX 混合网络的实现**：用注意力机制替代简单的 MLP，引入 commitment 对齐损失
> 3. **BNE 精炼模块**：基于 RefineModule 的迭代精炼，包含三种收敛检测机制
> 4. **多组件奖励系统**：设计了 TS、AL、CC 三个奖励组件，并实现了动态权重学习
>
> 这个项目让我深入理解了多智能体 RL、LLM 的采样控制、以及如何将博弈论引入 LLM 系统。我对 LLM 的训练和推理优化有浓厚的兴趣，特别是在 Agent 和后训练方面。"

### 6.2 简洁版（1分钟）

> "我是[姓名]，[学校]硕士。我参与了 ECON 框架的开发，这是一个基于贝叶斯纳什均衡的多智能体 LLM 推理框架。核心创新是用隐式信念协调替代显式消息传递，将 LLM 的采样参数建模为连续动作空间，通过 QMIX 实现多智能体协调。我负责了信念策略网络、混合网络和奖励系统的设计与实现。"

---

## 七、常见面试场景与应对策略

### 7.1 场景一：请介绍一下你做过的项目

**策略**: 用"问题-方案-效果"的结构
1. 问题：多智能体辩论的 token 消耗大、没有收敛保证
2. 方案：ECON 用 BNE 替代显式消息传递
3. 效果：在 GSM8K 等数据集上实现了更好的准确率和更低的 token 消耗

### 7.2 场景二：请解释一下某个技术细节

**策略**: 先说直觉，再说公式，最后说实现
- 例如解释 QMIX：先说"让每个 agent 的改进能带来全局改进"，再说公式 `∂q_tot/∂q_i ≥ 0`，最后说代码实现

### 7.3 场景三：请分析某个设计决策的优缺点

**策略**: 用"对比分析"的方法
- 例如分析"为什么用 GRU 而不是 Transformer 处理序列"
- 优点：GRU 更轻量、更适合处理短序列（episode 间的记忆）
- 缺点：无法处理长距离依赖（但在 ECON 中不是问题）

### 7.4 场景四：请提出改进建议

**策略**: 从"效率、效果、可扩展性"三个角度
- 效率：可以用更小的模型做 Coordinator，或者用模型蒸馏
- 效果：可以引入更多的奖励组件，或者用更大的 LLM
- 可扩展性：可以支持更多的 agent 类型，或者支持异构 agent

---

## 八、扩展知识准备

### 8.1 需要了解的相关技术

1. **LLM 采样策略**: temperature, top_p, top_k, repetition_penalty 的区别和联系
2. **RL 基础**: Q-learning, Policy Gradient, Actor-Critic, PPO
3. **多智能体 RL**: QMIX, MAPPO, MADDPG, CTDE 范式
4. **Transformer 架构**: Self-attention, Multi-head attention, Positional encoding
5. **博弈论基础**: 纳什均衡, 贝叶斯博弈, 最优响应

### 8.2 推荐阅读

1. **QMIX 论文**: "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
2. **MADDPG 论文**: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
3. **ECON 论文**: "From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium"

---

## 九、面试模拟题（自测用）

### 基础题
1. 请解释 ECON 的核心思想（30 秒）
2. 什么是贝叶斯纳什均衡？（1 分钟）
3. QMIX 的单调性约束是什么？（1 分钟）

### 进阶题
4. ECON 如何处理连续动作空间？（2 分钟）
5. 为什么用 Coordinator + Executor 的架构？（2 分钟）
6. 奖励系统中的 CC 组件如何平衡 fidelity 和 novelty？（2 分钟）

### 开放题
7. 如果要将 ECON 应用到代码生成任务，需要做哪些修改？
8. 如何评估多智能体系统的公平性（某个 agent 是否贡献过少）？
9. 如果 agent 数量增加到 100 个，系统需要如何改进？

---

## 十、总结

### 面试核心要点

1. **理解项目本质**: ECON 是用博弈论解决多智能体 LLM 协调问题
2. **掌握技术细节**: 信念网络、QMIX、BNE 精炼、奖励系统
3. **能够清晰表达**: 用"问题-方案-效果"的结构
4. **展现思考深度**: 能够分析优缺点、提出改进建议

### 准备建议

1. 熟悉代码实现，特别是核心组件的前向传播
2. 理解每个设计决策的原因，而不仅仅是知道是什么
3. 准备 2-3 个可以深入讨论的技术点
4. 练习用简洁的语言解释复杂的技术概念

---

*最后更新: 2026年6月*
*基于 ECON 框架 v1.0*
