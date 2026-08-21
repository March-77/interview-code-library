# DPO、PPO、GRPO、DAPO、GSPO 面试细节

本文默认你已经知道 policy gradient、actor-critic、importance sampling、KL divergence 等基本概念，只讨论它们落到 LLM 后最容易被追问的细节。

## PPO

### 先把 LLM 生成写成强化学习

对 prompt $x$ 生成 response $y=(a_1,\ldots,a_T)$ 时：

$$
s_t=(x,a_1,\ldots,a_{t-1}),\qquad a_t=\text{第 }t\text{ 个 token}
$$

Actor 给出下一个 token 的分布 $\pi(a_t|s_t)$；Critic 给出 $V(s_t)$，表示从当前前缀继续生成时，预期还能获得多少折扣累计奖励：

$$
V^\pi(s_t)=\mathbb E_\pi[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots\mid s_t]
$$

$\mathbb E$ 是对策略采样和环境/奖励随机性取期望。它是 value 的理论定义，训练时不需要人工算出这个期望，而是用 rollout 样本去估计。

### 真实 rollout 先向前生成，再统一算训练信号

1. Old policy 自回归生成完整 response，保存每个已采样 token 的 $\log\pi_{old}(a_t|s_t)$。
2. Reward model 或 verifier 看完整回答，给出最终分数 $R$。
3. 把 `[prompt, response]` 一次输入 Critic，得到 response 各位置对应的 $V_{old}(s_1),\ldots,V_{old}(s_T)$。Causal mask 保证每个位置不会偷看后续 token。
4. 有了 token reward 和 values 后，计算 TD residual，再算 GAE。
5. 固定这批 rollout，用 minibatch 更新 Actor 和 Critic；然后丢弃这批数据，用新 Policy 重新 rollout。

因此，“从后往前”指的是 GAE 的计算顺序，不是 Critic 倒着读文本。Critic 的一次 Transformer forward 可以并行产生所有前缀的 value。

### 最终 reward 如何变成 token reward

若暂时忽略 KL，Reward Model 只对完整 response 评分，则：

$$
r_1=\cdots=r_{T-1}=0,\qquad r_T=R
$$

「PPO 只能有最终 reward」是错的；这只是 LLM outcome-reward 的常见设定。实际 RLHF 还常加逐 token KL 惩罚：

$$
r_t^{KL}=-\beta\left(\log\pi_{old}(a_t|s_t)-\log\pi_{ref}(a_t|s_t)\right)
$$

中间位置使用 $r_t=r_t^{KL}$，最后位置使用 $r_T=r_T^{KL}+R$。因此 Critic 预测的是后续 shaped return，不一定只是最终 RM 分数。

### TD residual：一步结果比预期好了多少

$$
\delta_t=r_t+\gamma V_{old}(s_{t+1})-V_{old}(s_t)
$$

$r_t+\gamma V(s_{t+1})$ 是走完这一步后的新估值，$V(s_t)$ 是走之前的旧估值。所以：

- $\delta_t>0$：这次一步结果比 Critic 原先的平均预期好。
- $\delta_t<0$：这次结果比预期差。

“更好”不一定是 $r_t$ 更大，也可能是该 token 进入了一个更有前途的 $s_{t+1}$。若 Critic 完全准确，Bellman 方程保证的是：

$$
\mathbb E[\delta_t\mid s_t]=0
$$

单次 rollout 的 $\delta_t$ 仍然可正可负，因为动作、后续生成和 reward 都有采样噪声；而且 Critic 本身也可能不准。给定实际采样的动作时，它在期望上对应一步 advantage：

$$
\mathbb E[\delta_t\mid s_t,a_t]=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t)
$$

终止状态取 $V(s_{T+1})=0$，因此 $\delta_T=r_T-V(s_T)$。

### GAE：把后续多步信息汇总给当前 token

单个 $\delta_t$ 只比较一步前后的估值。GAE（Generalized Advantage Estimation，广义优势估计）把当前及后续 TD residual 折扣汇总：

$$
\hat A_t=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

实现上从 response 末尾向前递推：

$$
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}
$$

例如 token 3 的 advantage 不是 token 数字 `3+4+5+6`，而是：

$$
\hat A_3=\delta_3+\gamma\lambda\delta_4+(\gamma\lambda)^2\delta_5+(\gamma\lambda)^3\delta_6
$$

准确区分：$\delta_3$ 表示生成 token 3 后的「一步超预期程度」；$\hat A_3$ 表示结合整个后续结果后，token 3 相对当时平均预期好或差多少。

$\lambda$ 控制偏差—方差折中：

- $\lambda=0$：$\hat A_t=\delta_t$，最依赖 Critic，方差低但 Critic 不准时偏差大。
- $\lambda=1$：终止轨迹中等价于 $\hat A_t=G_t-V(s_t)$，更接近完整 Monte Carlo return，但方差大。
- PPO 常用 $\gamma=0.99,\lambda=0.95$；有限且确定终止的 LLM rollout 也可用 $\gamma=1$。

### 有了 Advantage，Actor 怎么更新

对 rollout 中实际生成的 token $a_t$，重新用正在训练的 Policy 计算概率，并与 rollout 时的概率相比：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
=\exp\left(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)\right)
$$

$\rho_t$ 不是超参数；它是新旧 Policy 对同一 token 的概率比。$\rho_t=1.2$ 表示新 Policy 给该 token 的概率是旧 Policy 的 1.2 倍。

PPO 希望 $\hat A_t>0$ 时提高该 token 概率，$\hat A_t<0$ 时降低概率，但又不允许一次改得过猛：

$$
L_t^{clip}(\theta)=\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
$$

这是要最大化的 objective。代码中通常用梯度下降，所以 Actor loss 加负号，并只在 response 的有效 token 上平均：

$$
\mathcal L_{actor}=-\mathbb E_t[L_t^{clip}(\theta)]
$$

Clip 裁的是 surrogate objective 中的 probability ratio，不是参数或梯度；它也不能保证真实 KL 一定低于某个值。

### Critic 如何学习

GAE 得到的 $\hat A_t$ 表示结果比旧 Critic 预测高了多少，所以构造 value target：

$$
\hat R_t=V_{old}(s_t)+\hat A_t
$$

例如旧预测为 $0.4$，Advantage 为 $0.3$，则新的回归目标是 $0.7$。Critic 用回归损失去拟合这个固定 target：

$$
\mathcal L_{value}=\mathbb E_t\left[(V_\phi(s_t)-\hat R_t)^2\right]
$$

计算 $\delta$、GAE 和 $\hat R$ 时必须用 rollout 当时保存的 old values，不能在 minibatch 内随着 Critic 更新而反复改 target。实际实现还可以对 value update 做 clipping。

### 完整 loss 和三个容易混淆的希腊字母

Actor 和 Critic 共享参数时，常见写法是：

$$
\mathcal L_{total}=\mathcal L_{actor}+c_v\mathcal L_{value}-c_e\mathcal H(\pi_\theta)
$$

$\mathcal H$ 是 token 概率分布的熵：

$$
\mathcal H(\pi_\theta(\cdot|s_t))=-\sum_a\pi_\theta(a|s_t)\log\pi_\theta(a|s_t)
$$

加 entropy bonus 是为了避免 Policy 过早集中到少数选择。由于优化器最小化 loss，所以写成 $-c_e\mathcal H$。LLM PPO 通常已有 sampling 和 reference KL，$c_e$ 经常取 0 或很小。Actor 和 Critic 若是独立模型/优化器，则可分别优化 $\mathcal L_{actor}$ 和 $\mathcal L_{value}$，不必真的拼成一个 total loss。

| 符号 | 含义 | 是否超参数 |
|---|---|---|
| $\gamma$ | reward/value 的时间折扣 | 是 |
| $\lambda$ | GAE 中后续 TD residual 向前传播的程度 | 是 |
| $\rho_t$ | 新旧 Policy 对已采样 token 的概率比 | 否 |

$c_v$、$c_e$、$\epsilon$ 和 KL 系数 $\beta$ 也是超参数。无可复现配置时，比盲猜单个系数更重要的是：标准化 advantage，监控 reward、reference KL、clip fraction、entropy 和 value explained variance，并用小规模实验扫描 learning rate、KL 强度和 PPO epochs。

### Old policy 不等于 reference policy

- Old policy：生成当前 rollout 的 Policy 快照，用来计算 $\rho_t$，限制一轮 PPO update 的步幅。
- Reference policy：长期冻结的 SFT/初始模型，用于 KL penalty，限制整个 RL 过程的累计漂移。

PPO 是 on-policy，但一批近期 rollout 可借助 ratio 和 clipping 做若干轮 minibatch update；不能像 off-policy replay buffer 一样长期重复使用旧数据。

## GRPO

### 本质：用组内相对表现取代 Critic baseline

PPO 用 Critic 估计每个前缀的 $V(s_{i,t})$，再用 TD residual 和 GAE 构造 token-dependent advantage。GRPO 不训练 Critic，而是对同一 prompt 采样 $G$ 条 responses，用同题其他回答的表现当 baseline。

若第 $i$ 条回答的最终 reward 为 $R_i$，常见的组内标准化 advantage 是：

$$
\hat A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon}
$$

因此两者的核心对照是：

- PPO：「这个 token 后的结果，比 Critic 对当前前缀的预期好多少？」
- GRPO：「这条回答，比同一道题采样出的其他回答好多少？」

GRPO 的本质区别不是「有没有最终 reward」或「能不能有步骤 reward」，而是 baseline 和 advantage 的构造方式。

### Outcome GRPO 中，同一 response 通常共享 Advantage

若一条 response $y_i=(a_{i,1},\ldots,a_{i,T_i})$ 只有一个最终 outcome reward，则通常将序列级 advantage 复制到其所有 token：

$$
\hat A_{i,1}=\cdots=\hat A_{i,T_i}=\hat A_i
$$

因此，高于组内平均的回答会整体获奖，低于平均的回答会整体受罚。这是 sequence-level credit assignment：正确回答中的无关或错误中间 token 也可能被鼓励；错误回答中的正确前半段也可能被惩罚。

这句话不能泛化为「所有 GRPO 的 token 永远共享 Advantage」。若引入 process reward，并据此构造位置相关的 return/baseline，就可以得到 $\hat A_{i,t}$。

### 共享 Advantage 不等于每个 token 的更新完全相同

GRPO 仍然在每个 token 上计算新旧 Policy 概率比：

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}
{\pi_{old}(y_{i,t}|x,y_{i,<t})}
$$

忽略 clip 时，一条 response 的 policy-gradient 形式近似为：

$$
\nabla_\theta L_i\approx
\frac{1}{T_i}\sum_t
\hat A_i\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})
$$

同一条 response 的 token 共享相同的好坏方向和 advantage 缩放系数，但每个 token 仍有不同的前缀、log-probability 梯度、ratio 和 clip 状态，因此实际参数贡献并不完全一样。

对比 PPO，忽略 clip 后是：

$$
\nabla_\theta L_i^{PPO}\approx
\frac{1}{T_i}\sum_t
\hat A_{i,t}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})
$$

例如 PPO 可能给一条回答的 token 分配 `[+0.8,+0.3,-0.6,+0.1]`，而 outcome GRPO 可能分配 `[+0.5,+0.5,+0.5,+0.5]`。最后对 token loss 求平均只是聚合梯度，不会消除这个差异：

$$
\operatorname{mean}_t(\hat A_{i,t}g_{i,t})
\neq
\operatorname{mean}_t(\hat A_{i,t})\operatorname{mean}_t(g_{i,t})
$$

### 步骤奖励不是 PPO 与 GRPO 的本质区别

只有最终「完成/未完成」二值 reward 时，PPO 和 GRPO 都会遇到稀疏奖励和 credit assignment 问题。PPO 的 Critic/GAE 能给出 token-dependent advantage，但这只是对后续影响的估计，不是真正的反事实因果归因；Critic 不准时也会误判。

Process reward 原则上两者都能使用。但只把每步 reward 相加成一个 $R_i$，然后仍让整条 response 共享 $\hat A_i$，并没有实现步骤级 credit assignment。要真正区分步骤，还要用后续步骤 reward 构造位置相关的 return 或 $\hat A_{i,t}$。

### Clipping、KL 与成本：不要从「没有 Critic」过度推论

GRPO 保留 PPO 风格的 token-level ratio 和 clipped objective：

$$
L_{i,t}^{clip}=\min\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right)
$$

DeepSeekMath 的原始 GRPO 目标还包含 reference-policy KL penalty，所以「GRPO 没有 KL」并不准确；后续 reasoning RL 实现可以根据任务移除它。

GRPO 省掉的是 value model 的参数、显存、通信和 value training，但并不必然减少全部计算。为了得到组内相对信号，它要求同一个 prompt 生成多个长 response，rollout 成本可能很高。

组内标准化的直接局限是：二值 reward 下，一组全对或全错时组内没有区分度，advantage 全为 0；group 很小时，均值和方差估计也很噪。原始 GRPO 的样本自身还参与 group mean，与 leave-one-out baseline 不同。

### PPO 是否天然比 GRPO 稳定

不一定。更准确的结论是：准确的 Critic 能让 PPO 获得更低方差、更细粒度的 advantage；不准的 Critic 会把 value error 传入 TD residual、GAE 和 Actor update，反而可能让 PPO 更不稳定。

Critic 学得好时，PPO 有几个优势：

- **降低方差**：$A_t=G_t-V(s_t)$ 减掉「这个前缀本来就能获得的回报」，留下超出状态平均预期的部分。
- **token-dependent credit**：同一条回答可以得到如 `[+0.5,+0.3,+0.1,-0.8]` 的 advantage，而不是整条统一奖罚。
- **不依赖当前 group 的区分度**：二值 reward 下，GRPO 的 `[1,1,1,1]` 和 `[0,0,0,0]` 组都没有相对信号；PPO 仍可以通过 $R-V(s_t)$ 产生信号。
- **可以跨 rollout 积累经验**：Critic 是由大量历史样本学出的 baseline，可能比当前小 group 的均值更稳定。

但 PPO 也多出了一整条失稳链路：

$$
V(s_t)、V(s_{t+1})\text{估计错误}
\rightarrow \delta_t\text{错误}
\rightarrow \hat A_t\text{错误}
\rightarrow \text{Actor 更新方向受污染}
$$

Actor 更新又会改变 rollout 分布，Critic 随后需要追踪新的回报分布。这种 Actor–Critic 耦合是非平稳的：Critic 更新太慢会滞后，太快又可能过拟合当前 rollout。PPO 还多了 value learning rate、value clipping、GAE、Critic 初始化及 Actor/Critic 更新比例等配置点。

GRPO 删除了 learned Critic 的误差和耦合，并通过组内中心化与标准化稳定 advantage 尺度；但它把问题换成了 group estimator 的方差：group 太小时统计量很噪，全对/全错时没有梯度，sequence-level advantage 的 credit assignment 也更粗。

因此不能背「PPO 比 GRPO 稳定」这个无条件结论。应该按下表回答：

| 条件 | 更可能的结果 |
|---|---|
| Critic 准确、value training 能跟上 Policy | PPO 的 advantage 更细、方差更低，可能更稳定 |
| Critic 不准或 Actor/Critic 失衡 | PPO 的 advantage 被污染，可能更不稳定 |
| GRPO group 足够大且 reward 有区分度 | GRPO 省去 Critic，训练链更简单，可能更稳定 |
| GRPO group 太小或频繁全对/全错 | 组内 baseline 高方差或直接无信号 |

面试版总结：

> PPO 和 GRPO 最本质的区别是 advantage baseline：PPO 学习状态相关的 Critic，GRPO 使用同一 prompt 多条回答的组内相对奖励。准确的 Critic 能给 PPO 提供低方差、token-dependent 的信号；但 Critic 也会引入估计误差和 Actor–Critic 耦合。GRPO 用更粗粒度、更依赖当前采样组的估计，换掉了 Critic 的训练复杂度，所以两者谁更稳定取决于 Critic 质量和 group 估计质量。

## DAPO

DAPO 仍然使用 group-relative advantage 和 token-level importance ratio。它主要解决 naive GRPO 在长 CoT 训练中的熵坍塌、无效采样、长度权重和截断噪声问题。

### Clip-Higher

普通 PPO/GRPO 经常使用对称 clip 区间 `[1-ε, 1+ε]`。DAPO 把上下界拆开：

$$
[1-\epsilon_{low},1+\epsilon_{high}],
\qquad \epsilon_{high}>\epsilon_{low}
$$

假设一个探索 token 在 old policy 下的概率只有 0.01，普通上界为 1.2，那么这次更新对它的有效提升很快会在大约 0.012 处被 clip。低概率 token 即使获得正 advantage，也很难得到明显的绝对概率增长。

DAPO 放宽 upper clip，允许这些低概率但成功的探索 token 获得更大的提升空间。Lower clip 没有同步大幅放宽，因为把某些 token 的概率压到接近 0 后，探索空间可能无法恢复。

### Dynamic Sampling

对于二值 outcome reward，全对组和全错组的 group-relative advantage 都是 0。DAPO 先过采样，再过滤这些没有组内区分度的 prompt group，直到 batch 中保留固定数量的有效 groups。

它解决的是固定训练 batch 逐渐被零梯度样本占满的问题。随着模型变强，简单题越来越容易全对；如果仍然按固定 prompt 数训练，有效 gradient batch size 会不断下降。

Dynamic sampling 不等于全错题在任何意义上都没有价值。它只说明在“组内二值 outcome reward”这个设定下，全错 responses 之间没有相对排序信号。如果有过程奖励或能区分错误程度的 reward，全错组也可能提供梯度。

### Token-Level Policy Gradient Loss

原始 GRPO 通常先对每条 response 内的 token loss 求平均，再对 responses 求平均。这样每条 response 总权重相同：一个 100-token response 和一个 5000-token response 各占一票。

DAPO 改为把整个 batch 中的有效 response tokens 放在一起平均：

$$
\frac{1}{\sum_i|y_i|}
\sum_i\sum_t L_{i,t}
$$

因此每个 token 权重相同，而不是每条 response 权重相同。长 response 对总梯度的贡献会更大。

这里的 Token-Level Policy Gradient Loss 指的是 loss aggregation，不是说 DAPO 首次使用 token-level ratio。GRPO 本身已经有 token-level ratio。

这种聚合更适合长 CoT 学习，但也可能强化长度偏好，所以需要和长度奖励、最大长度及数据分布一起考虑。

### Overlong Reward Shaping

长推理经常在最大生成长度处被硬截断。一个 response 在上限前正常结束，另一个只多写几个 token 就被截断并丢失最终答案，两者 reward 可能突然发生巨大变化。

DAPO 在接近最大长度的缓冲区内逐渐加入软惩罚，而不是只在截断点突然重罚。这样模型可以提前感受到继续变长的代价，减小边界附近的 reward noise。

### DAPO 中的 KL

DAPO 论文在可验证长推理任务中移除了 reference KL，并使用规则 outcome reward。理由是这里希望策略的推理能力可以显著偏离初始模型，reference KL 未必必要。

这不是一个可以无条件迁移到开放域 RLHF 的结论。对通用对话、安全和风格对齐任务，移除 KL 可能造成明显的语言分布漂移。准确说法应该是“DAPO 论文的具体 reasoning 设置移除了 KL”，而不是“DAPO 在定义上不能有 KL”。

## GSPO

GSPO 仍然对同一个 prompt 采样多个 responses，并使用组内标准化 reward。它主要修改的不是 advantage，而是 importance ratio 和 clipping 的粒度。

GRPO 对每个 token 独立计算 ratio，但采样出来的对象实际上是完整 response，reward 也通常属于完整 response。GSPO 认为，用单 token ratio 去修正 sequence-level 样本分布并不匹配，长序列中会出现较大的 gradient noise。

GSPO 对每条 response 计算长度归一化的 sequence ratio：

$$
s_i(\theta)=
\left(
\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}
\right)^{1/|y_i|}
$$

等价地：

$$
s_i(\theta)=
\exp\left(
\frac{1}{|y_i|}
\sum_t
[\log\pi_\theta(y_{i,t}|x,y_{i,<t})
-\log\pi_{old}(y_{i,t}|x,y_{i,<t})]
\right)
$$

它就是所有 token ratios 的几何平均。长度归一化很重要，否则完整序列概率是大量 token probability 的乘积，ratio 的尺度会强烈依赖 response 长度。

一条 response 共享同一个 sequence ratio，clipping 也以整条 response 为单位。GRPO 可能只 clip 一条 response 中的部分 tokens；GSPO 则倾向于整条 response 一起被接受或一起被 clip。这样 reward、advantage、importance sampling 和 clipping 都处在 sequence level。

这对 MoE 训练尤其重要。MoE 参数更新后，同一 token 的 expert routing 可能改变，token probability ratio 会产生明显波动。GSPO 只关注整段 response 的平均 likelihood 变化，对局部 routing 波动不那么敏感，也降低了训练引擎与 rollout 引擎之间 log-prob 精度差异的影响。

GSPO 的 clip threshold 数值远小于 GRPO 常见的 0.2，但两者不能直接比较。GRPO clip 的是单 token ratio；GSPO clip 的是整段平均 log-likelihood 对应的 sequence ratio，作用尺度不同。

GSPO 论文还讨论了 GSPO-token：继续使用 sequence-level ratio 和 clipping，但允许不同 token 使用不同 advantage。当整条 response 的 token advantages 都相同时，它与基本 GSPO 的目标和梯度等价。

## DPO

DPO 与前面的在线 RL 算法不在同一条训练流程上。它不使用 rollout、old policy、critic 或显式 advantage，而是在固定偏好数据上直接更新 policy。

一条数据是 $(x,y_w,y_l)$，分别表示 prompt、preferred response 和 rejected response。Policy 与冻结的 reference model 对两个 response 做 teacher forcing，计算完整 sequence log-prob：

$$
\log\pi(y|x)=\sum_t\log\pi(y_t|x,y_{<t})
$$

DPO loss 是：

$$
L_{DPO}=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
$$

它优化的不是 chosen 与 rejected 的 embedding distance，而是两条完整 response 相对 reference 的 log-likelihood margin。更新之后，policy 真正生成 chosen 类轨迹的概率会上升，生成 rejected 类轨迹的概率会下降。

Reference 不能简单理解成一个正则项的装饰。只最大化 $\log\pi(y_w)-\log\pi(y_l)$ 会鼓励模型任意改变分布；DPO 比较的是当前 policy 相对 reference 分别对两条 response 做了多大调整。它来自 KL-regularized RLHF 最优策略的重参数化。

### DPO 是否必须先做 SFT

数学上不需要。即使初始化时 $\pi_\theta=\pi_{ref}$，DPO logit 为 0，loss 梯度仍然非零，可以直接更新。

实践中通常先 SFT。原因不是 DPO 无法产生梯度，而是 DPO 的监督形式只告诉模型两条完整 response 中哪条更好。若 base model 还不会指令遵循、输出目标格式或生成基本推理过程，这种 pairwise signal 不如 SFT 的直接模仿有效。

常见流程是先用 chosen responses 做 SFT，然后复制 SFT checkpoint：一份作为可训练 policy，一份作为冻结 reference，再做 DPO。如果起点本身已经是 Instruct/Chat model，那么当前项目可以直接 DPO，但这个 checkpoint 通常已经在上游做过 SFT。

如果 base model 已经具备目标行为，DPO 数据覆盖充分，而且 chosen 并没有远离当前模型分布，直接从 base model 做 DPO 也可以。否则更稳的分工是：SFT 教会模型“怎么做”，DPO 教它“哪种做法更好”。

### DPO 为什么可以作为在线 RL 的冷启动

在稀疏 outcome reward 下，如果初始 policy 几乎采不到正确轨迹，GRPO 对同一 prompt 采出的 group 会经常全错。二值 reward 全部相同，group-relative advantage 就是 0，RL 没有可用信号。

DPO 可以使用 teacher model、搜索、rejection sampling 或人工构造的成功和失败轨迹，先提高成功轨迹的生成概率。它不需要解决完整的在线探索问题，只要把成功率从“几乎采不到”提升到“一个 group 中偶尔能采到”，后续 GRPO/DAPO/GSPO 就能比较成功和失败 response 并继续优化。

所以 DPO 冷启动的作用是改变 rollout distribution，而不是代替后面的在线 RL。

### DPO 在长轨迹上的问题

对单轮长 CoT，标准 DPO 在计算上完全可行。它使用 log-prob 求和，不会真的去计算一个极小的 sequence probability。

主要问题是 credit assignment。DPO 只知道整条 $y_w$ 优于整条 $y_l$，不知道哪一个推理步骤导致了结果差异。如果两条 response 有完全相同的前缀，这部分在 chosen/rejected margin 中会抵消；一旦它们较早分叉，后面的所有 token 都会被整体推高或压低，无法精确定位关键错误。

第二个问题是长度。Sequence log-prob 是 token log-prob 的和，chosen 和 rejected 长度差异很大时，loss 会混入长度信号。构造长度接近的 pair、使用长度归一化变体，或者比较共享前缀后的局部 continuation，通常更适合长推理。

第三个问题是 teacher forcing 与真实 rollout 的状态分布不同。DPO 只在离线数据给定的前缀上学习。如果模型实际生成时在中途走进一个数据未覆盖的错误状态，DPO 不会自动获得“如何从这个状态恢复”的训练数据。在线 RL 使用当前 policy 自己生成的轨迹，可以不断修正这种 distribution shift。

对多轮 agent trajectory，问题更严重。环境 observation 取决于之前采取的 action，整条 trajectory 的状态分布由 policy 和环境共同决定。标准 response-level DPO 把完整输出当成一个 contextual-bandit action，它的推导不能直接无条件搬到多轮 MDP。多轮场景通常需要 step-level preference、occupancy-aware 的 DPO 变体，或者真正的在线 RL。

因此在长轨迹推理上，DPO 更适合学习一个好的行为先验或作为 RL cold start，而不是单独承担长时序 credit assignment 和在线探索。

## 容易被继续追问的几个点

### 为什么有 reward model 还需要 critic

Reward model 通常只给完整 response 一个最终分数。Critic 为中间状态提供 expected return baseline，使 policy gradient 使用“比预期好多少”而不是绝对 reward，降低方差并提供 bootstrap。两者不能互相替代。

### GRPO 去掉 critic 后是否解决了 credit assignment

没有。它只是换了一种 baseline。若只有最终 outcome reward，一条 response 的 token 仍共享粗粒度 advantage。GRPO 解决的是 value model 成本和组内相对评价，不是完整的逐 token credit assignment。

### PPO clipping 和 reference KL 是否重复

不重复。PPO clipping 比较 new policy 与生成当前 batch 的 old policy，控制单轮更新；reference KL 比较当前 policy 与长期冻结的初始/SFT policy，控制累计漂移。

### DAPO 和 GSPO 的修改是否在同一维度

不完全在同一维度。DAPO 是一套长 CoT 训练配方，修改采样、clip 上界、loss aggregation 和长度 reward。GSPO 的核心修改更集中：把 token-level importance ratio 与 clipping 改成 sequence level。两者的技术动机可以同时存在，不能只按名称把它们看成简单前后替代关系。

### DPO 是否只是让 chosen 概率上升

不是。DPO 优化的是 chosen 与 rejected 相对 reference 的 margin。它同时包含对 preferred 和 rejected 的比较，而且 reference 决定了“相对于原模型改变了多少”。SFT 才更接近只最大化 chosen likelihood。

## 参考论文

Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015.

Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155), 2022.

Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023.

Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300), 2024.

Yu et al., [DAPO](https://arxiv.org/abs/2503.14476), 2025.

Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071), 2025.
