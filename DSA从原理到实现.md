# 从零读懂这个 DSA 算子：数学、数据搬运、流水线和精度

本文解释的是本目录 [dsa](dsa) 中的真实实现。它完成的是：**已经知道每个 query 要关注哪些 token 之后，计算这些 token 上的稀疏注意力**。它不负责选出这些 token，也不是完整的 DeepSeek 模型。

先记住一个贯穿全文的动作：用 query 给被选中的 token 打分，把分数变成权重，再按权重汇总它们的内容。数学上并不复杂；代码长，是因为要让矩阵计算、零散数据搬运和 softmax 同时工作，还要避免它们覆盖彼此尚未用完的数据。

建议按顺序阅读前八节，再对照代码阅读后半部分。最终时间见 [性能报告](PERFORMANCE_FINAL.md)，比较范围见 [公平性审计](FAIRNESS_AUDIT.md)，重构边界见 [重构说明](REFACTOR_NOTES.md)，编译器接口与 PR 的分析见 [编译器分析](COMPILER_ANALYSIS.md)。本文不把源码推导当成设备实测，也不把附件的历史时间当成本次重构后的时间。

## 1. 先认识 token、query、key、value 和 head

语言模型处理的基本单位是 token。一个 token 可能是一个字、一个词的一部分，或者一个符号。模型内部不会直接计算文字，而是把 token 表示成一串浮点数，即向量。

Attention 要回答的是：“处理当前这个 token 时，应该从其他哪些 token 取信息，分别取多少？”

- **Query，简称 Q**：当前 token 用来寻找信息的向量。
- **Key，简称 K**：其他 token 用来和 query 匹配的向量。
- **Value，简称 V**：真正被取走、汇总的内容向量。

最常见的匹配方式是点积：对应元素相乘，再全部相加。例如 `[1, 2]` 与 `[3, 4]` 的点积是 `1×3 + 2×4 = 11`。点积产生一个分数，不是最终权重。

一个 **head** 可以理解为一套独立的关注方式。同一个 query token 有很多 head，它们各自打分、各自得到结果。head 之间可以并行，但每个 head 内部仍要把所有被选 token 的贡献加起来。

**Prefill** 是一次处理一段已经给出的文本，因此 query 数通常很多。**Decode** 是生成后续 token 时的一步，通常每条序列只新增一个 query。两者可以计算同一个数学式，但能提供给硬件的并行任务数差别很大，这也是这里需要不同配置的原因。

## 2. 从普通 Attention 到这里的 Sparse MLA

### 2.1 普通 Attention 做什么

固定一个 query 和一个 head，假设有 `N` 个候选 token。它的计算可以写成：

$$
s_j=\alpha\,(q\cdot k_j),\qquad
p_j=\frac{\exp(s_j)}{\sum_{t=0}^{N-1}\exp(s_t)},\qquad
o=\sum_{j=0}^{N-1}p_j v_j.
$$

`s` 是分数，`p` 是 softmax 权重，`o` 是输出向量，`α` 是缩放系数。权重非负且总和为 1，所以输出是各个 value 的加权平均。

如果每个 query 都访问整段历史，历史越长，计算和读数据的成本就越高。**Sparse Attention** 把候选集合缩小：先给每个 query 选出一部分 token，再只在这个集合里算 attention。这里的“稀疏”指的是 token 选择，不代表向量的很多元素为零。

DeepSeek-V3.2-Exp 引入的 DSA 面向长上下文稀疏注意力。其官方仓库把索引计算和稀疏 attention kernel 分开提供；本目录实现的是后者。[DeepSeek-V3.2-Exp 官方说明](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

### 2.2 为什么这个算子只有一个 512 维 KV

MLA 是 Multi-head Latent Attention。理解本算子只需要抓住它的一点：多个 head 可以共享一份较小的 latent KV 表示，而不必为每个 head 都缓存完整的 K 和 V。

下面用列向量写一个简化的代数关系。假设一个 token 缓存的 latent 向量为 `c`，某个 head 的内容 key 为 `W_K c`，value 为 `W_V c`，则：

$$
q^T(W_K c)=(W_K^Tq)^Tc,
\qquad
\sum_j p_j(W_Vc_j)=W_V\left(\sum_jp_jc_j\right).
$$

左边是先展开 key/value 再做 attention；右边是先把 key 的投影“吸收”到 query 上，直接对 latent cache 算 attention，最后再投影输出。这个变换常被称为 **absorb**。它减少需要显式展开和缓存的数据，但不会自动省掉模型前后的投影计算。

本算子的输入 `q` 已经处在吸收后的 512 维空间，`kv` 就是 512 维 latent cache。于是同一个 `kv` 在 QK 计算里扮演内容 key，在 PV 计算里又扮演 value。输出也是 512 维 latent 加权和；它还不是经过模型输出投影后的最终隐藏状态。

位置关系另由 64 维 RoPE 部分提供。RoPE 是一种旋转位置编码：这里的 `q_rope`、`k_rope` 已经由上游准备好，kernel 只计算它们的点积，不在 kernel 中生成角度或旋转向量。内容点积和位置点积相加，形成最终分数。官方模型的吸收路径、位置分支与后续投影可在 [MLA.forward](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py) 对照查看。

### 2.3 “512+64”不自动决定真实模型的 scale

本附件约定 `α = 576**(-0.5) = 1/24`，TLA 和 AscendC 比较时都传这个值，因此双方计算契约一致。但不能由此推断所有 MLA 模型都应该这样设置。

例如上述官方模型按吸收前的 QK head 维度 `128+64` 设置 scale，长上下文配置还可能调整它；query 吸收到 512 维后仍保留原 scale。因此，**本 kernel 的固定 scale 是 benchmark 范围限制**。接入真实模型时应保留模型原本的 scale，并重新验证，不能因为张量变成 512+64 就擅自改成 `1/24`。[官方 softmax_scale 定义及使用](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)

## 3. 用输入输出把问题彻底说清楚

统一入口在 [dsa/api.py](dsa/api.py)。符号含义如下：

| 符号 | 含义 | 一个 decode 示例 |
|---|---|---:|
| `B` | batch，一次处理多少条序列 | 1 |
| `SQ` | 每条序列有多少个 query token | 1 |
| `H` | 当前设备上的 head 数 | 128 |
| `L` | KV cache 中有多少个 token | 8192 |
| `TOPK` | 每个 query 提供多少个被选索引 | 2048 |

输入输出是：

| 张量 | 形状 | 类型 | 作用 |
|---|---|---|---|
| `q` | `[B,SQ,H,512]` | BF16 | 吸收后的内容 query |
| `kv` | `[B,L,1,512]` | BF16 | 所有 head 共享的 latent key/value |
| `q_rope` | `[B,SQ,H,64]` | BF16 | query 的位置分支 |
| `k_rope` | `[B,L,1,64]` | BF16 | cache 的位置分支 |
| `indices` | `[B,SQ,1,TOPK]` | int32 | 每个 query 要访问的 cache 行号 |
| `output` | `[B,SQ,H,512]` | BF16 | latent attention 输出 |
| `softmax_max` | `[B,1,SQ,H]` | FP32 | 每行缩放后分数的最大值 |
| `softmax_sum` | `[B,1,SQ,H]` | FP32 | 减去最大值后的指数和 |

`kv` 和 `indices` 的 head 维是 1，意思是各个 head 共享它们，不是只计算一个 head。所有张量必须连续、位于同一张 NPU；入口检查形状和 dtype，不会把输入搬回 CPU 扫描一遍。

固定 `b, t, h`，把该 query 的索引记为 `I[j]`，实际问题就是：

$$
s_j=\frac{1}{24}\left(
\sum_{d=0}^{511}q_{b,t,h,d}\,kv_{b,I[j],0,d}
+\sum_{r=0}^{63}q^R_{b,t,h,r}\,k^R_{b,I[j],0,r}\right).
$$

对有效索引计算 softmax 后，用同一组 `kv[b,I[j],0,:]` 加权求和。没有第二套独立 value 输入，也没有额外的 causal mask。要表达因果选择，需要上游给出符合模型规则的索引；这份实现不是带任意因果模式的通用 API。

索引顺序可以改变，重复项却不能随意删除。例如 `[5,5,9]` 表示 token 5 有两份逻辑贡献，不能为了省时间改成 `[5,9]`。代码可能只搬一次相同数据再复制，但保留两份权重。

无效索引的地址会被夹到安全范围，随后在分数上用原始索引判断有效性，把无效项变成负无穷，使其权重为零。已有附件主要验证有效前缀加 `-1` 后缀的情况；**每个 query 至少一个有效索引**是本接口的使用前提，全无效行不在支持范围。

原生接口里的 `return_softmax_lse=True` 容易让人误读。本次所比较的接口实际返回 max 和 sum；本包保持同样的三输出语义。真正的对数归一化项可以由 `max + log(sum)` 得到，不能把 `sum` 本身当作 LSE。对应合同与官方证据见 [公平性审计第 2 节](FAIRNESS_AUDIT.md#2-比较的到底是什么算子)。

## 4. 稳定 softmax：为什么一定要减去最大值

直接算 `exp(s)` 有个问题：分数很大时，指数会溢出。可以给所有分数减去同一个常数，而不改变 softmax：

$$
\frac{e^{s_j-m}}{\sum_t e^{s_t-m}}
=\frac{e^{s_j}/e^m}{\sum_t e^{s_t}/e^m}
=\frac{e^{s_j}}{\sum_t e^{s_t}}.
$$

选 `m=max(s)` 后，所有 `s_j-m≤0`，最大的指数恰好是 1。这就是稳定 softmax。

为了后面能分块计算，我们暂时不除：

$$
m=\max_j s_j,\qquad
\ell=\sum_j e^{s_j-m},\qquad
u=\sum_j e^{s_j-m}v_j,\qquad
o=u/\ell.
$$

`m` 是一个标量，`ℓ` 是一个标量，`u` 是一个 512 维向量。这三个量足够描述已经处理过的 token。代码中的 `running_max`、`running_sum`、`numerator_ub` 分别对应它们。

## 5. Online softmax：只看一块，为什么还能得到全局结果

如果把全部 score 和全部 probability 都存到外部内存再计算，访存成本很高。FlashAttention 的关键思路是按小块处理，在片上保存中间状态，避免把大中间矩阵写回外部内存。它节省的是数据搬运，不是通过漏算 token 来变快。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

这里也是这个思路，但访问的是给定索引指向的稀疏 KV。每次拿 `BN` 个 token，称为一个 **tile**，也就是一小块。处理完一块，保留 `(m,ℓ,u)`，再处理下一块。这种归一化递推也称 online softmax。[Online normalizer 原论文](https://arxiv.org/abs/1805.02867)

假设旧状态为 `(m_old,ℓ_old,u_old)`，新 tile 的最大分数为 `m_tile`。两部分要先统一到同一个最大值：

$$
m_{new}=\max(m_{old},m_{tile}),\qquad
\alpha_{old}=\exp(m_{old}-m_{new}).
$$

新 tile 的指数权重以 `m_new` 为基准，旧状态也要乘 `α_old`：

$$
p_j=\exp(s_j-m_{new}),
$$
$$
\ell_{new}=\alpha_{old}\ell_{old}+\sum_{j\in tile}p_j,
\qquad
u_{new}=\alpha_{old}u_{old}+\sum_{j\in tile}p_jv_j.
$$

为什么旧状态必须重新缩放？因为旧的 `u_old`、`ℓ_old` 采用的是 `exp(s-m_old)`；现在共同尺度改为 `exp(s-m_new)`，两者正好差一个 `exp(m_old-m_new)`。如果省略这个乘法，遇到后面 tile 的分数更大时，前面 token 的相对权重就会错。

### 一个可以手算的小例子

取已经缩放好的分数 `[1,2,3]`，三个标量 value 为 `[10,20,40]`。先处理前两个：

```text
m_A = 2
l_A = exp(-1) + 1                 = 1.367879441
u_A = exp(-1) * 10 + 1 * 20       = 23.678794412
```

再处理第三个。新的最大值是 3，因此旧状态乘 `exp(2-3)`：

```text
alpha = exp(-1)                   = 0.367879441
m     = 3
l     = alpha * l_A + 1           = 1.503214724
u     = alpha * u_A + 40          = 48.710941656
o     = u / l                    = 32.404513384
```

直接对三个数一起算，权重为 `[0.090030573,0.244728471,0.665240956]`，结果相同。注意这些是实数公式和 Python 双精度示例；实际 kernel 的 BF16 P、硬件指数和浮点求和存在舍入，不能由这个恒等式声称设备输出与全 FP32 参考逐位相等。

下面这段只需要 Python，可独立运行：

```python
import math

scores = [1.0, 2.0, 3.0]
values = [10.0, 20.0, 40.0]
m, denominator, numerator = -math.inf, 0.0, 0.0
for begin, end in [(0, 2), (2, 3)]:
    next_m = max(m, max(scores[begin:end]))
    rescale = math.exp(m - next_m)
    weights = [math.exp(s - next_m) for s in scores[begin:end]]
    denominator = rescale * denominator + sum(weights)
    numerator = rescale * numerator + sum(
        p * v for p, v in zip(weights, values[begin:end])
    )
    m = next_m
print(numerator / denominator)  # 32.40451338379263
```

## 6. Split-KV：把一个 query 分给多个核，最后怎么合并

Online softmax 仍然允许一个任务从头到尾处理所有被选 token。但 decode 的 query 太少，只这样做可能让大多数核闲着。

**Split-KV** 把一个 query/head tile 的被选 token 分成几段，每段由不同任务独立计算。每个分片 `s` 输出 `m_s`、`ℓ_s` 和未归一化的 FP32 向量 `u_s`。然后 [dsa/merge.py](dsa/merge.py) 做：

$$
m=\max_s m_s,\qquad w_s=e^{m_s-m},
$$
$$
\ell=\sum_s w_s\ell_s,\qquad
u=\sum_s w_su_s,\qquad o=u/\ell.
$$

仍用上面的例子，分片 A 处理 `[1,2]`，分片 B 处理 `[3]`。A 的状态是 `(2,1.367879441,23.678794412)`，B 的状态是 `(3,1,40)`。把 A 乘 `exp(-1)`，B 乘 1，再相加，就得到同一个结果 `32.404513384`。

不能直接把两段已经归一化的输出取平均。这个例子那样做会得到 `28.655292893`，明显不同，因为两个分片包含的总权重不同。

Split 的收益是更多并行，代价是：主核输出更多 FP32 中间数据，增加一个 merge kernel，还要支付任务发射和任务之间的间隙。Prefill 已有很多 query，通常无需再 split；decode 则往往值得付出这笔成本。比较性能时，**主核和 merge 都必须计入**。

## 7. Ascend950 上是谁在干这些工作

本实现采用一个 Cube 与两个 Vector sub-block 配合的数据流。为了理解代码，可以把它看成一个小组：Cube 专注矩阵乘，两个 Vector 分担向量操作和数据整理；搬运引擎在不同存储之间传输数据。默认主核发射 28 个 Cube program，对应 56 个 Vector sub-block，这是本配置选择，不是对所有设备型号核心数量的概括。

| 名称 | 在本 kernel 中的角色 |
|---|---|
| GM | 输入、输出和部分工作区所在的设备全局内存；容量大，但不能当成免费读写 |
| L1 | Cube 使用的片上输入存储，保存 Q、被选 KV 和概率 P |
| L0A / L0B | 矩阵乘更靠近计算单元的输入存储；本代码由 MMAD/template 路径组织其搬运 |
| L0C | 矩阵乘的 FP32 累加结果，存放 QK 或 PV tile |
| UB | Vector 的片上缓冲，存索引、score、softmax 状态、gather 数据和输出累加值 |
| SIMD 寄存器 | Vector 正在计算的一小批数；这里 FP32 操作常一次处理 64 个元素 |
| Fixpipe | 把 L0C 结果按所需布局搬出；本代码用它把结果分给两个 Vector |

`tla.alloc(..., space="ub")` 声明 UB 存储，`tla.view`/`reshape` 描述已有存储的视图。**视图不是数据拷贝，也不是浮点类型转换**。一个 buffer 能被看到，不代表数据已经搬完；异步完成关系由编译器和显式同步共同维护。

Cube 喜欢块状矩阵布局。代码里常见的 **NZ** 是面向矩阵计算的物理排布，和普通按行连续的 **ND** 不同。理解时先把 NZ 想成由小矩阵块组织的数据，不要把四维物理 shape 当成四个数学维度。例如 Q 的 `(36,BM//16,16,16)` 仍表示逻辑上的 `BM×576` 矩阵，36 来自 `576/16`。

### 此处矩阵的 M 维是 heads

这是读代码时最容易搞错的一点。一个任务固定一个 query token，却同时处理 `BM` 个 head：

```text
Q: BM × 576      K: BN × 576
QK^T: BM × BN

P: BM × BN       V: BN × 512
PV: BM × 512
```

因此 `BM` 不是本任务的 query 数，而是 head tile 大小。这样做能让多个 head 复用同一批被选 KV。两个 Vector 各负责 `VM=BM/2` 个 head；同一份索引供这些 head 共同使用。

## 8. 先看任务分配，再读几百行 kernel

[dsa/config.py](dsa/config.py) 中每个 `Tuning` 控制一种调度。最关键的是 `BM`、`BN`、`split`：

- `BM` 大：同一批 KV 可被更多 head 复用，但 score 和输出状态占更多片上存储。
- `BN` 大：处理 TOPK 所需循环更少，但单次 gather、softmax、矩阵乘和临时存储更大。
- `split` 大：同一个 query 可以分给更多核，但分片状态和 merge 成本增加。

总任务数是：

$$
jobs=B\times SQ\times(H/BM)\times split.
$$

代码使用循环 `job = program_id, program_id+CORES, ...`。这意味着同一个物理执行小组做完一个任务，还会继续处理下一个任务，而不是为每个 query 单独发射一个 kernel。

以 `decode_b1` 为例：`B=1,SQ=1,H=128,BM=64`，若 `split=1`，只有 2 个任务；`split=14` 后有 28 个任务，可以覆盖这里配置的 28 个 Cube program。TOPK2048 配 BN64 一共 32 个 tile，14 个分片分别处理 2 或 3 个 tile，所有 tile 都被覆盖，没有只算前面的一部分。

这里的“不等长”指分片包含的 **完整 tile 数** 不同。它不表示已经支持任意长度的最后一个 tile。workspace 路径则要求每个分片含偶数个完整 BN tile，对应其两阶段循环。

## 9. 主路径：先整理到 GM 双槽，再让 Cube 转 NZ

16 个已选规格中有 15 个使用 [dsa/workspace.py](dsa/workspace.py)。它合并了附件中 fastgather、wide_fastgather、bn256 三份近似实现，通过编译期参数保留各自的调度。

先看完整过程：

```text
Vector：读取 indices → 零散读取 KV 与 RoPE → 拼成连续 576 列 → 写 GM 工作槽
Cube ：等待工作槽 → GM 到 L1，硬件 ND2NZ → QK → Fixpipe 写 score 到 UB
Vector：score 缩放和 mask → online softmax → BF16 P 打包到 NZ → 送 L1
Cube ：P × V → Fixpipe 写 PV 到 UB
Vector：u = alpha * u + PV → 下一 tile
最后 ：split=1 直接归一化；split>1 写 FP32 分片，交给 merge
```

同一条流程中，Vector 会提前准备下一 tile，让部分 gather 与当前 Cube 工作重叠。Q 只需为本 job 装入一次，随后在处理各个 KV tile 时重复使用。

### 9.1 配对 gather：少发搬运指令，不少读数据

`gather` 指的是按索引从很多位置收集数据。比如索引 `[100,7,42,42]` 指向四行，这些行不连续，不能用一次普通的大连续搬运全部解决。

代码先把相邻两项按大小排列，例如 `[100,7]` 变成 `[7,100]`。于是两行可以描述成：起点为第 7 行，行距为 `100-7` 行，行数为 2。`ops.load_pair` 用一条带动态行距的 DMA 描述这两行，每一行内部仍是连续的 512 个 BF16。

这并不是“把两行数据压缩成一行”。读取字节数没有减少，收益主要来自减少搬运指令和相关标量控制开销。也没有把全部 TOPK 全局排序；只在相邻二元组内部调整顺序。

同一套顺序同时用于内容 key、RoPE key 和 value，所以数学含义保留。浮点加法顺序可能改变末位，这应由精度验证约束。

快路径有条件：这一 tile 的配对索引都有效，且每对不是相同索引。workspace 代码先检查这些条件；如果不满足，就使用逐行安全读取，再在 softmax 里按索引 mask。它不会因为性能数据通常使用唯一索引，就省略真实的判断。

每个 Vector 负责 BN 的一半行，并以 16 行为一组搬运。`gather_key_even/odd` 和 `gather_rope_even/odd` 是轮换的 UB 暂存区，使生产和消费不必总挤在同一块内存上。

### 9.2 为什么还要绕一次 GM

零散读回的数据先按行拼成 `[KV512,RoPE64]`，成为连续的 576 列矩阵。Cube 再从 GM 读入 L1，利用现成的 ND2NZ 路径转换布局。这样 QK 可一次覆盖 576 维，减少 Vector 手工打包布局的工作。

代价也很明确：gather 后的数据要额外写一次 GM，再读一次 GM。以 TOPK2048 为例，一份被选 KV+RoPE 的逻辑数据量是 `2048×576×2` 字节，即 2.25 MiB；工作区往返额外产生约 4.5 MiB 的逻辑 GM 流量。实际到 HBM 的流量还受缓存影响，不能把这个估算当成硬件计数器结果。

工作区大小为 `[CORES,2,BN,576]`，每个 Cube program 有两个槽。28 核、BN128 时为 7.875 MiB，BN256 时为 15.75 MiB。它不随整个 `SQ` 线性增长，因为完成一个 job 后可以复用。

这里是 **GM 双槽**；workspace 路径只声明了一份 `keys_l1`。不要因为看到两个 GM 槽，就推断每一级内存都有两份 KV 缓冲。

### 9.3 一个 QK，两个 Vector

Cube 使用 `transpose_b=True` 算 `QK^T`，FP32 结果放到 L0C。随后 Fixpipe 的 `ROW_SPLIT` 把 head 行分成两半，每个 Vector 得到 `VM×BN` 的 score。

这不是先在 Vector 上把整个 K 转置一遍。转置标记描述矩阵乘要怎样解释操作数，由 MMAD/template 路径实现。因此研究 transpose 成本时，应检查实际生成的搬运和矩阵指令，不能只凭 Python 中有没有 `transpose` 字样判断。

### 9.4 softmax 为什么有 BN128 和 BN256 两段

一条 FP32 SIMD 向量通常装 64 个数。BN128 对应两组，BN256 对应四组。代码先做缩放、有效性 mask，再计算本行最大值、`exp(score-new_max)`、分母，以及旧状态的 rescale。

BN256 的求和采用原实现的分组顺序，例如先算 `(p0+p1)` 和 `(p2+p3)`，再相加并归约。浮点加法不满足严格结合律，重构没有把它随意改成另一种更短的写法。减少源码行数，不应偷偷改变已验证的算术顺序。

FP32 指数值用于分母，同时转换成 BF16 供 PV 矩阵乘。`vdintlv` 和 `vsstb` 把这些 BF16 按 L1 需要的 NZ 排列写到 UB，再通过 UB→L1 搬运交给 Cube。

`prob_padded_ub` 的物理行距是 `(VM+1)*16`，逻辑上每行只搬 `VM*16` 个元素。多出的空间属于布局 padding，不能当成额外 head 或额外概率。它影响实际 pitch 和存储组织；这份分析没有单凭 padding 的存在就证明某种具体的 UB bank conflict 被消除。

### 9.5 为什么 PV 有时一次 512 列，有时分两次 256 列

数学输出始终是 512 维，`PV_COLS` 只影响临时结果如何分块。较小 BM 使用 512 列 PV，BM128 配置使用 256 列片段。

BM128 时每个 Vector 负责 64 个 head，仅 FP32 输出分子就占：

```text
64 × 512 × 4 bytes = 128 KiB
```

若再放完整的 `64×512` FP32 PV 临时值，又是 128 KiB。分成 256 列，PV 临时值降为 64 KiB。算法仍计算全部 512 列，只是分两次搬出和累加。减少临时存储压力与增加分片搬运次数之间，需要按规格权衡。

### 9.6 UB 别名：为什么一个 byte buffer 能有三个名字

代码中的 `shared_bytes` 是一块原始字节存储，`byte_view` 把它分别看成：

- FP32 的 score tile；
- FP32 的 PV fragment；
- 最后输出的 BF16 fragment。

在 BM128/BN128/PV256 配置中，score 本来需要额外 32 KiB。它和 PV 生命周期分开后，能复用同一块较大的存储。这个操作没有数值转换，也没有额外拷贝；改变的是“同一段字节现在由谁使用”。

代价是必须严格管理最后一次使用。Cube 不能在 Vector 还在读上一份 PV 时，用新 score 覆盖这些字节；Vector 也不能在 Cube 还要读取打包 P 时随意复用相关空间。共享 UB 的分支多了一些特定等待，正是为了维护这个合同。**少分配一块 buffer 不等于可以同时保留两份数据。**

## 10. 小 decode 路径：直接打包 NZ，支持不等长 split

[dsa/ragged.py](dsa/ragged.py) 目前被 `decode_b1` 选中。它和 workspace 的根本区别是数据路径：Vector 直接把 gather 后的 KV 打成 NZ，送入 L1，省去中间 GM 工作区。

```text
Vector：indices → paired gather 到 UB → 手工 NZ packing → L1
Cube ：QK 的 512 维部分 + RoPE 的 64 维部分 → score
Vector/Cube：online softmax → P 到 L1 → PV → 更新分子
```

省下 GM 往返并不意味着它在所有规格都更快。它增加 Vector 的布局整理、多个小缓冲区和细粒度控制。在更多 query、大 BN 或更高吞吐场景下，硬件 ND2NZ 的路径可能更合适。配置表是在两种成本结构之间作选择。

### 10.1 为什么保留两次 QK MMAD

内容 KV 和 RoPE 分别打包、分别存 L1。Cube 先计算 512 维内容点积并清零累加器，再把 64 维 RoPE 点积加到同一个 FP32 L0C：第二次 `clear_accum=False`。

两部分之间没有把结果转成 BF16。它和一次 576 维逻辑 MMAD 的主要区别是布局、调度和可能的累加顺序，不是把 RoPE 丢掉，也不是额外降低中间结果精度。

PV 也有一个区别：ragged 先在 L0C 得到完整的 512 列结果，再用 Fixpipe 分成两个 256 列片段送到 UB；workspace 的 PV256 配置则每次计算 256 列。配置表里的 256 都表示单次搬出、消费的列数，不意味着两条路径的矩阵乘切法完全相同。

### 10.2 重复索引为什么只读一行却仍然正确

每个 Vector 负责 BN64 的一半，即 32 行，组织为 16 对。配对索引先排序、夹到安全地址，得到起点、行距和有效搬运行数。两个地址相同时，DMA 只读一行，随后 SIMD 把它复制成两行。

代码里 `65537` 看起来很突兀，其实是 `0x00010001`：同一个 int32 中放两个 16 位标记，后面按 16 位元素解释时能得到成对的选择标志。它是配合 BF16 数据整理的编码，不是改变 attention 的神秘常数。

RoPE 的两行暂存分配成 `(2,128)`，实际只使用每行 64 个元素，保留较大的物理行距。公共 DMA 接口必须保留这个 pitch；若错误地把目标当作紧凑 `(2,64)`，第二行会写错位置。

### 10.3 哪些东西真的双缓冲了

这条路径明确有 `kv_l1/kv_next`、`kr_l1/kr_next` 两套 L1，下一 tile 的数据可以准备到另一套。同时 `acc/acc_alt`、`m/m_alt`、`l/l_alt` 轮换，旧状态和新状态分开存放。

每轮按 phase 0/1 交替。下一 tile 的索引和 DMA 会提前发起，随后把布局打包安排在能与当前计算配合的位置。因为 split14 的分片可能有奇数个 tile，循环不能盲目执行第二个 phase；最后输出也必须按 tile 数的奇偶选择正确的状态缓冲。

这种处理是 `ragged` 这个名字的来源。不要为了统一接口删除奇偶分支，它会决定读哪一份最终结果。

## 11. 同步：ready、free 和 drain 分别在保护什么

矩阵乘和 DMA 不是 Python 执行一行就立刻完成一行。Cube、Vector 和搬运流水可以异步推进。因此，正确性必须同时回答三个问题：

1. **Ready：数据产生完了吗？** 消费者不能读尚未写完的数据。
2. **Free：旧数据用完了吗？** 生产者不能覆盖消费者还在使用的槽。
3. **Drain：最后一轮结束了吗？** 离开循环或开始下一个 job 前，不能留下未完成的读取或残余同步状态。

两份缓冲只解决“有两个地方可写”；它不能自动回答这三个问题。

本代码常见的 flag 含义如下，具体位置应对照两个主核：

| flag | 本实现中传达的事件 |
|---|---|
| `0 / 5` | 当前/下一数据槽准备好，允许 Cube 使用 |
| `1` | score 已经通过 Fixpipe 写到 Vector 可读的 UB |
| `2` | BF16 P 已经送到 L1，允许 Cube 做 PV |
| `3` | 一片 PV 已经写到 UB，允许 Vector 累加 |
| `4` | Vector 已消费完 PV，共享空间可以复用 |
| `7 / 8` | workspace 路径在 job 结束时的 C/V 握手 |

flag 编号只是本程序分配的同步槽，不是某个数学常量。`MODE.BLOCK` 这里用于协作小组的同步，不是每算一块就让全部 28 个 Cube 一起等待的全局 barrier。

### 用一个 PV 临时槽理解 free credit

开始时 Vector 发一个 `flag4`，表示“这个槽最初是空的”。Cube 在需要覆盖该槽前取得这个许可，写完后发 `flag3`。Vector 等 `flag3`，读完并更新 FP32 分子，再发 `flag4`。

```text
空槽许可 → Cube 写 PV → PV ready → Vector 读 PV → 空槽许可 → 下一片
```

共享 score/PV 空间时，Cube 在写 score 前就要取得许可，因此第一片 PV 不再重复消费同一份许可；后续片段仍要等待。代码里的条件不是可以随手合并的重复写法，它对应不同的槽位占用过程。

循环结尾 Cube 再等最后一个 `flag4`，确保最后一次消费已完成。这就是 drain。workspace 的 7/8 握手还把 job 输出的完成关系与下一 job 的资源复用衔接起来。

### 为什么很多 wait 都写 PIPE.S

这里保留了附件的 `block_wait_flag(..., PIPE.S, ...)`。生产者的 set 则出现在 MTE3、F、V 等不同流水上。PIPE 描述同步指令关联的执行流水和顺序关系，不是给 flag 加的任意标签。

等待放在标量侧，可能使后续指令发射等待；但仅把 `PIPE.S` 文本换成某个搬运 PIPE，并不能自动得到安全重叠。必须逐条证明：究竟哪次写完成后允许谁读、哪次读完成后允许谁覆盖、每次 set/wait 是否正确配对，再看编译结果和设备执行。

这次重构没有改这些等待。静态 AST 等价证明的是前后保留相同结构，不是对整个硬件协议的形式化无死锁证明。进一步调同步需要单独的 IR、trace 和设备精度证据。

## 12. 读 SIMD 代码时，不要被指令名字吓住

先把 SIMD 看成“一条指令同时操作很多数”。这里大部分 FP32 运算每次处理 64 个元素；一个 head 有 BN128 个分数，就分两次拿进来。

| 代码 | 可以怎样理解 |
|---|---|
| `vlds` / `vsts` | 从 UB 读一组数到寄存器，或写回 UB |
| `dist="BRC_B32"` | 读一个 FP32 标量并广播，给这一组所有元素使用 |
| `dist="ONEPT_B32"` | 只写一个标量，例如一行的 max 或 sum |
| `vmuls` | 一组数乘同一个标量，这里用于 score scale |
| `vexpdif(a,b)` | 原生指数差运算，对应 `exp(a-b)` 的硬件实现 |
| `vmax` | 两组向量逐元素取较大值 |
| `vreduce_max` / `vreduce_sum` | 把一组元素归约成最大值或和 |
| `vcmp` / `vsel` | 比较生成条件，再按条件选值；用来 mask、处理重复行 |
| `vtruncf(..., bfloat16)` | FP32 到 BF16 的转换；精确舍入含义看 IR 属性，不能只按名字猜测“截断” |
| `vdintlv` / `vintlv` | 拆分交错元素或重新交错；用于偶奇索引和 BF16 打包 |
| `vsstb` | 按特定块间距存数据，用来组织 Cube 所需布局 |

例如 workspace 的 softmax 可以先按下面的伪代码理解，然后再对应到指令：

```text
读当前 head 的 BN 个分数
乘 scale，无效索引对应的分数置 -inf
new_max = max(old_max, 当前 tile 最大值)
alpha = exp(old_max - new_max)
p = exp(score - new_max)
new_sum = alpha * old_sum + sum(p)
保存 new_max、new_sum、alpha
将 p 转为 BF16，按 NZ 打包并送到 L1
```

看明白这一层后，再去研究寄存器数、归约次序、padding 和布局，通常比从第一条 `vlds` 开始硬追更容易。

## 13. 精度：哪些是 BF16，哪些一直保持 FP32

BF16 使用 16 位存储，能节省带宽和矩阵乘输入空间，但有效尾数少于 FP32。FP32 占 32 位，在归约和状态累积中保留更多精度。这份实现的分工是：

| 阶段 | 精度 |
|---|---|
| Q、KV、RoPE 输入 | BF16 |
| QK 的 L0C 累加 | FP32 |
| score 缩放、mask、max、exp、sum、rescale | FP32 |
| 送入第二次矩阵乘的 P | BF16 |
| PV 累加和 online 输出分子 | FP32 |
| split 中间输出及 merge | FP32 |
| 最终 output | BF16 |
| 返回 max、sum | FP32 |

一个细节很重要：分母是 FP32 指数的和，而 PV 使用转换后的 BF16 P。因此最终结果不是“先把完整 FP32 softmax 归一化，再用全 FP32 做 PV”的逐位复刻；它遵循这里约定的低精度矩阵乘、较高精度状态路径。

这不是 TLA 独有的精度捷径。审计到的官方 AscendC BF16 分支同样把 FP32 指数转换为 BF16 后做 PV，并使用 FP32 中间状态；相关固定源码、转换属性和版本限制列在 [公平性审计第 3 节](FAIRNESS_AUDIT.md#3-精度路径是否对齐)。**同一精度类别**也不等于每条指令和每个中间数逐位一样：不同 BN、split、归约树、QK 分段都会影响舍入。

本实现没有为了快而冻结最大值、跳过重标定、裁剪大指数、略去 RoPE、把 FP32 分子降为 BF16，或让不同 split 直接平均。若继续优化，这些都不能只凭“随机测试没出问题”就改掉。

精度检查也必须说清标准。附件主要使用全张量最大绝对误差除以参考最大绝对值；这能描述一种误差尺度，但不能替代逐元素容差，更不能保证小幅值行没有较大相对误差。合理的复验应同时看 output、max、sum，检查每行误差、非有限数，并覆盖均匀分数、尖峰分数、重复索引和边界索引。新输入也要跑，不能只复用调参时的随机种子。

## 14. 全部 16 个配置：名字和参数怎么读

下表来自 [dsa/config.py](dsa/config.py)。`H` 是本设备上的 head 数；名称中的 `tp8` 表示该测试选了较少的本地 heads，不代表本 kernel 内部执行 tensor parallel 通信。这里没有跨设备 collective。

| 规格 | B | SQ | H | L | TOPK | BM | BN | split | 路径 | PV 搬出列数 | score/PV 共用 UB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| decode_b1 | 1 | 1 | 128 | 8192 | 2048 | 64 | 64 | 14 | 直接 NZ | 256 | 否 |
| decode_b8 | 8 | 1 | 128 | 32768 | 2048 | 128 | 128 | 2 | GM 双槽 | 256 | 是 |
| decode_b32 | 32 | 1 | 128 | 32768 | 2048 | 128 | 128 | 4 | GM 双槽 | 256 | 是 |
| decode_tp8 | 8 | 1 | 16 | 32768 | 2048 | 16 | 128 | 8 | GM 双槽 | 512 | 否 |
| prefill_q16 | 1 | 16 | 128 | 8192 | 2048 | 128 | 128 | 4 | GM 双槽 | 256 | 是 |
| prefill_q64 | 1 | 64 | 128 | 32768 | 2048 | 128 | 128 | 1 | GM 双槽 | 256 | 是 |
| prefill_tp8 | 1 | 128 | 16 | 32768 | 2048 | 16 | 256 | 1 | GM 双槽 | 512 | 否 |
| prefill_k512 | 1 | 128 | 64 | 32768 | 512 | 64 | 128 | 1 | GM 双槽 | 512 | 否 |
| decode_b4_h128 | 4 | 1 | 128 | 8192 | 2048 | 128 | 128 | 8 | GM 双槽 | 256 | 是 |
| prefill_b4_h128 | 4 | 8192 | 128 | 8192 | 2048 | 128 | 128 | 1 | GM 双槽 | 256 | 是 |
| decode_b4_h64 | 4 | 1 | 64 | 8192 | 2048 | 64 | 128 | 8 | GM 双槽 | 512 | 否 |
| prefill_b4_h64 | 4 | 8192 | 64 | 8192 | 2048 | 64 | 128 | 1 | GM 双槽 | 512 | 否 |
| decode_b4_h32 | 4 | 1 | 32 | 8192 | 2048 | 32 | 128 | 8 | GM 双槽 | 512 | 否 |
| prefill_b4_h32 | 4 | 8192 | 32 | 8192 | 2048 | 32 | 128 | 1 | GM 双槽 | 512 | 否 |
| decode_b4_h16 | 4 | 1 | 16 | 8192 | 2048 | 16 | 128 | 8 | GM 双槽 | 512 | 否 |
| prefill_b4_h16 | 4 | 8192 | 16 | 8192 | 2048 | 16 | 256 | 1 | GM 双槽 | 512 | 否 |

`decode_b1` 的 score 虽不和 PV 共用 UB，仍有其他字节存储复用，例如 RoPE pack/P pack，以及 PV/最终输出。因此表格最后一列只描述 score/PV 这一对，不表示整个 kernel 没有别名。

这张表不是“理论上任何情况下的最优解”。它保留附件选择的配置，重构没有借机重新调参。合法的形状特化和 benchmark 特征之间也要分清：只按维度选择配置是正常优化，按参考答案、随机种子或是否处在计时阶段改变结果则不允许。

## 15. 怎样按文件阅读和运行

建议依次读：

1. [api.py](dsa/api.py)：看清输入合同、配置选择、工作区和 split 输出分配。先确认自己理解完整调用包含哪些任务。
2. [config.py](dsa/config.py)：找到目标规格，写下 BM、BN、split、PV_COLS、SHARE_SCORE。它们决定接下来哪些编译期分支存在。
3. [merge.py](dsa/merge.py)：短且直接，对照第六节公式理解分片如何还原整体结果。
4. [workspace.py](dsa/workspace.py)：按“分配与初始化 → 首槽预取 → 稳态 → 排空与输出”四段读。先追一轮 BN tile，再看下一槽如何提前准备。
5. [ragged.py](dsa/ragged.py)：与主路径比较 GM 往返、NZ 打包、L1 双缓冲、old/new 状态和奇偶收尾。
6. [compat.py](dsa/compat.py)：最后再看接口如何生成 IR。它是编译器边界，不是 attention 数学的一部分。

在已激活、能够使用 NPU 的 Python 环境中，进入 `x` 目录，基本调用为：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export DSA_API_MODE=legacy
python your_program.py
```

`your_program.py` 中：

```python
from dsa import forward

# q/kv/q_rope/k_rope/indices 由调用方准备，形状和类型见第三节。
output, maximum, denominator = forward(q, kv, q_rope, k_rope, indices)

# 验证工具可以同时取得这一次调用真正编译、执行的 kernel 对象。
outputs, kernels = forward(
    q, kv, q_rope, k_rope, indices, return_kernels=True
)
```

第一次调用包含编译开销，不应用作稳定运行时延。验证程序见 [tools/review.py](tools/review.py)、[tools/run_matrix.py](tools/run_matrix.py)；用 `--help` 查看当前命令行选项。它们负责原版、重构版和原生实现的对照，不被 kernel 包导入。

Host 入口自身也有开销。最终实现把固定形状与不可变 `Tuning` 的合法性放在导入时验证；每次调用仍完整检查输入 shape、dtype、设备、连续性和可变的 core 数，自定义配置也仍逐次检查。同一次调用只读取必要次数的元数据，并减少临时 Python 容器。这里缓存的是已经验证的固定配置，不是输入值或计算结果。对照工具用 PyTorch meta 张量和假的 kernel 发射检查这些 host 行为，不碰 NPU；其开销数据与限制见 [重构说明](REFACTOR_NOTES.md#host-入口怎样减少开销同时保留检查)。

### 兼容层的三个模式到底有什么区别

在导入 `dsa` 之前设置 `DSA_API_MODE`：

- `legacy`：默认模式，保留附件的 typed IR 桥接，适合验证重构前后等价。它不修改磁盘上的安装文件，但在进程内包装 CodeGenerator，并解析构造的 MLIR。
- `public`：只用正式 TLA API，所需接口缺失就报错，不悄悄退回 legacy。
- `auto`：按可用 API 选择路径，适合便利使用；严肃的性能报告仍须记录实际选择了哪条路径。

因此“没改 site-packages 文件”和“完全只用公开 API”是两个不同问题。附件能在某个未改安装目录的 wheel 上通过自带适配器运行，不代表随便装一个普通版本就能直接使用。

本次检查的 main-dev 基线已具备 `cube.mmad`、`simd.vmuls`、`simd.vexpdif` 等能力；本次编译器工作补齐 `scalar_load`、`byte_view` 和动态行距/行数 GM→UB DMA。具体基线、接口约束、测试及社区 PR 状态以 [COMPILER_ANALYSIS.md](COMPILER_ANALYSIS.md) 为准。接口存在、成功编译、设备精度通过、性能无损是四件事，不能相互替代。

## 16. 为什么重构没有把所有重复代码都消掉

通常写 Python，会把重复部分抽成小函数。但这里的 Python 是生成设备程序的前端，抽函数可能影响 IR 边界、内联、别名分析和调度。旧适配路径还依赖 AVE/HIVM 的内联行为。

因此这次只把确实共享的三条 workspace 实现合成一份编译期模板，把 host 分派、merge、兼容层分别整理；ragged 的不同数据流独立保留。首轮 gather 与稳态 gather、奇偶状态输出等重复块暂时保留，因为它们出现的位置就是调度的一部分。

[AST 检查脚本](tools/check_refactor_ast.py) 会按选定 constexpr 专门化原版与重构版，统一局部改名后比较整个函数体，而不是仅比较关键算式。五种 workspace 专门化加 ragged、merge，共七项比较覆盖这 16 个配置。结果见 [ast-equivalence.json](validation/ast-equivalence.json)。

这说明没有在重构时改掉这些设备语句和顺序，但 AST 本身不能证明后端 IR、机器码、同步行为或性能相同。最终 host 入口更新后的 [16 规格设备记录](validation/bitwise-final-all16.json) 提供了另一层证据：legacy 路径的 output/max/sum 在检查 dtype、shape 后，按整数位视图比较全部相同；各规格主核及 merge 的可执行 `.text` 指纹也全部相同。

逐位检查覆盖本次实际输入，可执行代码段相同也不保证每轮延迟一样。Host 检查与发射、缓冲地址、缓存状态和共享设备干扰等仍需要从实测中区分。因此不能直接把“相同 `.text`”写成“所有性能指标已无损”；完整的原版/重构版/AscendC 时间、指标和测量范围见 [最终性能报告](PERFORMANCE_FINAL.md)。

## 17. 如何判断快得是否公平

这里最重要的是明确比较对象：**给定相同索引、BF16 输入、同样 scale 和输出要求的 sparse MLA attention**。它不是完整 DSA layer，不含 indexer、TopK、KV cache 构建、RoPE 生成和最终投影。

在这个范围内，双方允许采用不同 tiling 和流水深度，因为这些正是实现质量的一部分。需要对齐的是数学问题、精度要求、核心资源和计时范围。一个实现有 split+merge，另一个只有一个任务，也必须比较完整工作。

附件中的统计主要是 **一次调用所有设备任务时长之和**。这能反映 kernel 的设备工作量，但不包含 CPU 分配、发射、任务之间空隙等全部成本。把两个任务拆开后，即使总 task duration 下降，完整调用区间也可能受额外开销影响。因此性能表应同时明确指标，不能把不同口径拼成一个“加速比”。

共享服务器上的 1% 差异还可能只是波动。合理的无损验证需要原版和重构版交错运行、相同输入和软件栈，保存全部样本及中位数，并检查每个规格的核心使用和所有任务。历史数据只反映它自己的环境；详细审计、原始记录复算和边界见 [FAIRNESS_AUDIT.md](FAIRNESS_AUDIT.md)。

## 18. 还能往哪里优化，先验证什么

目前不能从某个 decode 规格获胜就推断所有 prefill 都会获胜。两类场景的限制不同：decode 常缺少并行任务，prefill 则更可能被持续的 gather、布局转换和 C/V 配合限制。以下是可以继续验证的方向，不是已经得到性能收益的宣告。

### 18.1 先把耗时拆成可观察的阶段

应在同一规格下看 trace、搬运和计算计数器，回答：Cube 是在等索引数据，还是 Vector 在等 QK？GM 工作区往返占多少？softmax/pack 占多少？PV 临时槽的 free 等待是否让后续工作无法发射？

看到 `PIPE.S` 多、或者某段 Python 很长，都不能单独确定瓶颈。需要把指令和具体 producer/consumer 对上，再决定改哪一处。

### 18.2 把直接 NZ 的收益推广到更多规格

目标是省下 GM 中转，但同时控制 Vector pack 成本。值得研究更高效、可复用的布局转换，以及 BN128/256 下片上缓冲的容量和流水。不能直接把 BN64 改成 256 就期待线性加速：UB 占用、DMA 数、向量寄存器和同步距离都会变化。

### 18.3 继续减少稀疏访存的控制成本

paired gather 已经把两行合成一条搬运描述。若真实索引存在连续段或更大规则块，可以在调用内识别并合并，但检测本身也有成本；排序和重排必须同时覆盖 key、RoPE、value 与 mask，并保持重复项语义。

性能验证要加入不同连续性、重复率和索引分布，不能只在有利的随机分布上报告收益。也不能把索引预处理移到计时外，让比较范围不一致。

### 18.4 按并行度决定 split，而不只看主核微秒数

small decode 可以继续探索 split 与 BM 的组合，使任务数足够又不产生过多 merge。prefill 应关注整段持续吞吐，并优先避免无必要的分片中间结果。所有决策都以主核加 merge 的完整成本为准。

### 18.5 让编译器理解内存别名和流水，而不是隐藏它们

正式的 scalar load、byte view、动态 strided DMA 能让 agent 直接表达所需行为，不用拼接 MLIR 文本。进一步还需要：清晰的布局构件、可靠的零开销内联、真实 alias/memory-effect 分析，以及能显示 ready/free/drain 依赖的诊断。

编译器若要自动复用 UB，必须证明上一份数据最后一次读已结束；若要自动移动 wait，必须证明新的等待仍覆盖必要的硬件完成关系。无法证明时应报告未知，不能靠“看起来两个 buffer 在交替”判断安全。具体建议和本次代码落点见 [编译器分析](COMPILER_ANALYSIS.md)。

## 19. 使用前最后确认边界

本包适合研究和复验上述 16 个 BF16 稀疏 MLA 规格。使用其他情况前，要补相应合同和测试：

- scale 目前固定为 `1/24`，真实模型可能需要其他值。
- 它只接受连续输入，不是任意 stride、任意 dtype 的通用实现。
- head、BN tile 和 split 有整除条件；不支持任意残缺 tile。
- 每个 query 至少一个有效索引；全无效行不在支持范围。
- 不包含 paged KV cache、额外 causal 模式、indexer、TopK、反向或跨设备通信。
- 返回的是 latent 输出，不含模型最终输出投影；量化 KV cache 的解码等也不在这里。
- 附件历史结果、静态重构证据、新环境的设备结果必须分别标明，不能互相冒充。

读完代码时，可以用三个问题检查自己是否真正掌握了实现：每一块数据现在放在哪；下一位使用它的人在等什么完成；这一块存储什么时候才允许覆盖。再把每一轮的 `m、sum、numerator` 对回 online softmax 公式，几百行硬件代码就能还原成一条可追踪的计算过程。
