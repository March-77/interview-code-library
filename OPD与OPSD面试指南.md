# OPD 与 OPSD 面试指南

本文默认读者已了解 Policy Gradient、PPO、蒸馏、KL、teacher forcing 和 importance sampling，只保留 OPD/OPSD 的训练流程、核心公式与高频追问。

## 一张表先说清楚

| 问题 | OPD | OPSD |
|---|---|---|
| rollout 谁生成 | 当前/近期 student | 当前/近期 student |
| teacher 是谁 | 独立、通常更强且冻结的模型 | 同一底座模型在 privileged context 下形成的条件 teacher |
| teacher 多看到什么 | 通常没有额外上下文，能力来自参数 | reference solution $y^\star$ 等 privileged information |
| teacher 做什么 | 在 student prefix 上输出 next-token 分布 | 在 `x + y^\star + student prefix` 上输出 next-token 分布 |
| 梯度更新谁 | student | 只更新 student 分支；teacher logits stop-gradient |
| 是否需要标准答案 | 不一定，只用 prompts 也可以 | 原论文需要 $(x,y^\star)$ |
| 本质差异 | 外部参数更强的 teacher | 同源模型 + 上下文信息不对称 |

两者都属于 on-policy token-level distillation。区别不在“是否 on-policy”，也不在“是否 token-level”，而在 **teacher 的来源及其额外信息**。

## 统一记号

对问题 $x$，rollout policy $\mu$ 生成：

$$
\hat y=(a_1,\ldots,a_T)\sim\mu(\cdot\mid x),
\qquad s_t=(x,a_{<t}).
$$

在 student 访问到的前缀上：

$$
p_t(v)=\pi_\theta(v\mid s_t),
\qquad q_t(v)=\pi_T(v\mid s_t).
$$

$v$ 是任意词表 token，$a_t$ 是 rollout 实际采到的 token。全词表蒸馏使用所有 $p_t(v),q_t(v)$；sampled-token 实现只使用 $p_t(a_t),q_t(a_t)$。

## OPD 的真实训练流程

1. 从数据集采样 prompt $x$。
2. rollout policy $\mu$ 生成 student response $\hat y$。同步训练时 $\mu$ 是当前 student；异步或多轮更新时通常是 student 的旧快照 $\pi_{\theta_{old}}$。
3. 将同一条 `[x, \hat y]` teacher-force 给 student 和外部 teacher。两者在相同的 student prefix $a_{<t}$ 上输出 $p_t$、$q_t$。
4. 在每个 response 位置计算全词表 divergence，或构造 sampled-token teacher signal。
5. teacher 冻结且不接收梯度，只更新 student。
6. student 更新后重新 rollout；旧 response 复用过久就不再是 on-policy 数据。

Teacher 通常不生成训练 response，只评价 student 已访问的状态。不要把 teacher 与 rollout policy 混为一谈。

## OPSD 的真实训练流程

给定 reasoning 数据集：

$$
\mathcal S=\{(x_i,y_i^\star)\}_{i=1}^N.
$$

同一个底座模型通过不同上下文形成两个策略：

$$
p_S(\cdot\mid x),
\qquad
p_T(\cdot\mid x,y^\star).
$$

一批训练过程是：

1. student 只看 $x$，生成 $\hat y\sim p_S(\cdot\mid x)$。
2. student forward：`[x, \hat y]`。
3. teacher forward：`[x, y^\star, instruction, \hat y]`。
4. 在每个 student prefix 上比较：

   $$
   p_S(\cdot\mid x,\hat y_{<t})
   \quad\text{与}\quad
   p_T(\cdot\mid x,y^\star,\hat y_{<t}).
   $$

5. 只在 response mask 上计算 loss；$y^\star$ 不直接作为 student label。
6. teacher forward 使用 `no_grad`，只对 student logits 反传。

$y^\star$ 的作用是改变 teacher 的条件分布，而不是直接给 rollout token 打对错标签。Teacher 也不必先生成完整 reasoning；对拼接序列做一次 forward 即可并行评价所有 student prefixes。

### OPSD 的 `self` 到底是什么

`self` 指 teacher 与 student 来自同一个底座模型，但条件信息不同：teacher 看得到 $y^\star$，student 看不到。它不代表：

- 无监督；原论文依赖 reference solution；
- teacher 和 student 两边一起反传；只有 student 有梯度；
- response 由 teacher 生成；rollout 仍来自 student；
- 模型能凭空获得其在 privileged context 下也无法理解的能力。

### Teacher 权重是否随 student 更新

需要区分框架定义与主实验：

- **框架定义**：同源模型在不同上下文下形成 teacher/student。
- **论文主实验**：teacher 固定为 step 0 初始模型；LoRA 实现中 teacher forward 禁用 adapter，只更新 student adapter。
- **常见变体**：dynamic teacher 使用当前 student 权重但 stop-gradient；EMA teacher 使用 student 的滑动平均。前者是移动 target，通常更不稳定。

面试中不要只说“两者参数完全相同”。更准确的是：**二者同源；单步 teacher 无梯度；主实验跨训练步固定 teacher。**

## `on-policy` 的准确含义

On-policy 指训练 prefixes 来自当前或近期 student 的生成分布：

$$
\hat y\sim\pi_{\theta_{old}}(\cdot\mid x),
\qquad s_t=(x,\hat y_{<t}).
$$

它描述的是状态/前缀的采样分布，与以下问题无关：

- teacher 是否冻结；
- loss 使用 forward KL、reverse KL 还是 JSD；
- prompt 是否重复使用；
- 是否复用 RL 训练代码。

若生成后 student 已更新，使用 importance ratio：

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}.
$$

它校正“数据由 old policy 生成、目标却属于 new policy”的分布差异。固定一批 response 训练很多 epoch 后，即使它最初由 student 生成，也已经相对新 student 变成 off-policy。

## 三类核心 loss

### 1. 全词表 Forward KL

$$
\mathcal L_t^{FKL}
=D_{KL}(q_t\|p_t)
=\sum_{v\in\mathcal V}q_t(v)
\log\frac{q_t(v)}{p_t(v)}.
$$

Teacher 冻结时等价于 soft-target cross-entropy，其 student logit 梯度为：

$$
\frac{\partial\mathcal L_t^{FKL}}
{\partial z_t(v)}=p_t(v)-q_t(v).
$$

它直接提高 $q_t(v)>p_t(v)$ 的 token logit，使用 teacher 的完整词表分布。

### 2. 全词表 Reverse KL

$$
\mathcal L_t^{RKL}
=D_{KL}(p_t\|q_t)
=\sum_{v\in\mathcal V}p_t(v)
\log\frac{p_t(v)}{q_t(v)}.
$$

其 logit 梯度为：

$$
\frac{\partial\mathcal L_t^{RKL}}
{\partial z_t(v)}
=p_t(v)\left[
\log\frac{p_t(v)}{q_t(v)}
-D_{KL}(p_t\|q_t)
\right].
$$

Student 当前几乎不覆盖的 teacher mode，其梯度可能也很弱；student 已覆盖但 teacher 不认可的区域会受到强惩罚。

### Forward 与 reverse 不只是交换位置

$$
D_{KL}(q\|p)
=\mathbb E_{v\sim q}
\left[\log\frac{q(v)}{p(v)}\right],
$$

$$
D_{KL}(p\|q)
=\mathbb E_{v\sim p}
\left[\log\frac{p(v)}{q(v)}\right].
$$

交换位置同时改变了 ratio 方向和加权分布：

- Forward KL 由 teacher 加权，重点惩罚 student 漏掉 teacher mode，倾向 mode-covering。
- Reverse KL 由 student 加权，重点惩罚 student 生成 teacher 不认可的 mode，倾向 mode-seeking。

若 student 表达能力足够且能找到全局最优，两者最优点相同：

$$
p=q,
\qquad
D_{KL}(q\|p)=D_{KL}(p\|q)=0.
$$

所以差异不在理想最优解，而在 **student 无法完美复制 teacher 时如何取舍，以及有限采样下能否到达该最优点**：

> Forward KL 倾向于“尽量不要漏掉 teacher 的模式”；reverse KL 倾向于“尽量不要生成 teacher 不认可的模式”。

OPSD 原论文主实验中，全词表 forward KL 效果最好；一些 sampled-token OPD 使用 reverse KL，是因为它能自然写成 student-sampled Policy Gradient estimator。不能脱离 student 初始化、容量、teacher 多模态程度和实现方式断言哪种 KL 更好。

### 3. Sampled-token reverse-KL Policy Gradient

只查询 rollout token 的 log-prob：

$$
A_t
=\log q_t(a_t)-\log p_{old,t}(a_t).
$$

$A_t$ detach 后作为 token advantage：

$$
\mathcal L_{sampled}
=-\mathbb E_t
\left[
\rho_t(\theta)\operatorname{sg}(A_t)
\right].
$$

若 rollout 与更新使用同一策略，也常写成同一参数点的一阶梯度估计：

$$
\mathcal L_{sampled}
=-\mathbb E_t
\left[
\operatorname{sg}(A_t)
\log p_t(a_t)
\right].
$$

对 $a\sim p_t$ 取期望：

$$
-\mathbb E_{a\sim p_t}
\left[
(\log q_t(a)-\log p_t(a))
\nabla_\theta\log p_t(a)
\right]
=\nabla_\theta D_{KL}(p_t\|q_t).
$$

必须注意：

- 单个 $\log p_t(a_t)-\log q_t(a_t)$ 不是完整 KL，只是 sampled-token Monte Carlo 项；
- advantage 必须 detach；
- sampled-token 与全词表 reverse KL 目标相关，但单批梯度方差、显存和实际 loss 不同；
- 常见即时 advantage 不会把未来位置的 divergence 回传给更早 token。

## Sequence-level 与 token-level

OPD/OPSD 常对一条 response 的 token divergence 求平均：

$$
\ell(x,\hat y)
=\frac1T\sum_{t=1}^T D(q_t\|p_t).
$$

它最终是一个 sequence 标量，但每个位置有独立的 teacher/student 分布，不像 outcome GRPO 通常把同一个 sequence advantage 复制给所有 token。

Token-level dense supervision 也不等于完美长期 credit assignment。Teacher 回答的是“给定当前 prefix，下一 token 应如何分布”，不会自动判断更早哪个 token 对最终错误负有因果责任。错误答案末尾的 token 甚至可能几乎不受罚，因为在错误长前缀下，它们对 teacher 也已经可预测。

若 teacher 与 student 定义在相同输入上，sequence reverse KL 满足 chain rule：

$$
D_{KL}(P_S(\hat y\mid x)\|P_T(\hat y\mid x))
=\mathbb E_{\hat y\sim P_S}
\left[
\sum_tD_{KL}(p_t\|q_t)
\right].
$$

但实践中常对 rollout sampling stop-gradient，sampled-token 配方还可能只用即时 reward，因此实际使用的是局部/semi-gradient。Student-prefix 上的 forward KL 也不等于 sequence forward KL：后者的 prefix expectation 应来自 teacher。

## 与 SFT、KD、PPO、GRPO 的区别

| 方法 | prefix/response 来源 | 信号 | 更新对象 |
|---|---|---|---|
| SFT | 固定 expert trajectory | one-hot next-token CE | policy |
| 普通 soft KD | 固定 expert/teacher trajectory | teacher 全词表分布 | student |
| OPD | student rollout | 外部 teacher 在 student prefix 上的分布 | student |
| OPSD | student rollout | 同源 privileged teacher 在 student prefix 上的分布 | student 分支 |
| PPO | old-policy rollout | reward、Critic、GAE、ratio、clip | actor 和 critic |
| Outcome GRPO | policy 对同题多次 rollout | 组内相对 sequence reward | policy |

关键结论：

- SFT 可视为经验数据分布到模型的 forward KL，但只使用 one-hot target，且 prefixes 固定。
- 普通 soft KD 与 forward-KL OPD 的内层 loss 可以相同；区别是前者在固定 expert prefixes 上训练，后者在 student prefixes 上训练。
- Sampled-token OPD 可以复用 importance-sampling RL loss，但没有 Critic、GAE 与 PPO clip 就不是 PPO。
- Outcome GRPO 一组全对/全错时可能无信号；OPD/OPSD 一条 rollout 即可在各位置产生 teacher discrepancy。
- RL 直接优化 verifier/RM reward；OPD 优化 teacher matching。Teacher 错误或目标不一致时，dense 信号同样可能有害。

## 稳定性、成本与适用场景

### OPD

适合：外部 teacher 明显更强、与 student 的 tokenizer/推理模式兼容，并能返回 logits。

成本：student rollout + student forward/backward + 大 teacher no-grad forward。全词表 logits 还有 $O(BT|\mathcal V|)$ 的存储/通信压力。

典型失败：

- sampled-token reverse KL 下 student 采不到有效 teacher mode；
- teacher/student 思维或格式不兼容，KL 被风格 token 主导；
- teacher 并未真正优于 student；
- 异步 rollout 过旧，ratio 与 policy lag 过大；
- 长序列后段只有低价值、由错误前缀决定的信号。

能力/support 差距大时，先用 SFT/off-policy distillation cold start，再做 OPD 通常更合理。

### OPSD

适合：有可靠 $y^\star$，模型看 solution 后能够理解并评价推理，但不希望维护外部大 teacher。

成本：student-context 与 privileged-teacher-context 两次 forward；全词表 objective 仍有大词表显存开销。`self` 省掉外部大模型，不会省掉 teacher forward。

典型失败：

- 问题超过模型理解阈值，看 $y^\star$ 也无法形成可靠 teacher；
- reference solution 错误或风格偏置；
- style-token divergence 主导梯度；
- dynamic teacher 与 student 同步漂移；
- 训练很快达到峰值后退化，需要 early stopping。

OPSD 论文使用逐词表项 divergence clipping 抑制少数 style entries。它不是 PPO ratio clipping，也不只是截断一个位置的总 KL。

论文中 OPSD 每题使用 1 条、最多 1024-token rollout；对比 GRPO 使用每题 8 条、每条最多 16k token，因此显示出明显 token efficiency。这是实验配置，不是固定算法倍数。

## “OPD 与 OPSD 最本质区别”的面试版回答

> OPD 和 OPSD 都由当前或近期 student 生成 rollout，再让 teacher 在 student 真正访问的每个 prefix 上给 next-token 分布，最后只更新 student。因此两者都是 on-policy、token-level dense supervision。最本质的区别是 teacher 的来源和额外信息：OPD 使用独立、通常更强且冻结的外部模型；OPSD 使用同一底座模型在 privileged context 下形成 teacher，例如 teacher 能看到 reference solution，而 student 只看问题。OPSD 的 `self` 是模型同源但条件信息不对称，不代表无监督，也不代表 teacher 与 student 两边一起反传。原论文主实验固定 step-0 teacher，只更新 student LoRA；rollout 始终由 student 产生，teacher 负责评价而不是生成训练答案。

## 高频误区

- OPD response 由 teacher 生成——错，由 student/rollout policy 生成。
- 使用 KL 就叫 on-policy——错，on-policy 由 prefix 采样分布决定。
- Teacher 等于 rollout policy——错，rollout policy 是 student 快照。
- OPSD 无需监督——错，原论文需要 reference solution。
- OPSD teacher 必须跟随 student 更新——错，主实验固定初始 teacher。
- Teacher 接收梯度——错，teacher logits stop-gradient。
- OPSD 默认使用 reverse KL——错，主实验使用全词表 forward KL。
- Forward-KL OPD 就是 SFT——错，OPD prefixes 来自 student，SFT prefixes 固定。
- 一个 sampled log-ratio 就是 KL——错，它只是 Monte Carlo 项。
- Dense token signal 已解决长期 credit assignment——错，它只指导给定 prefix 下的下一 token。
- Sequence 平均意味着所有 token 信号相同——错，每个位置的分布不同。
- OPSD 没有 teacher 成本——错，仍需 privileged teacher forward。
- OPSD 的 token-efficiency 倍数是算法保证——错，取决于 rollout 配置。
- OPSD divergence clipping 就是 PPO clipping——错，前者截 divergence contribution，后者截 new/old ratio surrogate。

## 参考论文与实现

Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024.

Zhao et al., [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734), 2026.

[OPSD 官方实现](https://github.com/siyan-zhao/OPSD)：用于区分 fixed、dynamic、EMA teacher 及 full-vocabulary、sampled-token 变体。

Lu & Thinking Machines Lab, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)：sampled-token reverse-KL Policy Gradient 配方。
