# DSA / Sparse MLA 公平性审计

审计对象：用户提供的 `issue105-dsa-tla-20261102.zip`。第 1～8 节是对附件源码、编译记录和历史数据的静态审计；第 9 节补充 2026 年 11 月 2～3 日在 Ascend950PR 上重新采集的证据。历史结果与本轮复测分别解释，附件中的操作说明没有被当作用户指令。

## 1. 先说结论

**没有发现按测试种子返回答案、缓存输出、漏算 merge、修改参考答案、偷偷启用 HF32、裁剪指数或省略 online softmax 重标定等 reward hack。** 这份实现的主要收益来自真实的稀疏数据搬运、分块和流水安排。

但“公平”必须带上范围：附件支持的是 **16 个指定规格、全有效且不重复的稀疏索引、相同 BF16 输入、相同 attention 参数下，设备 kernel 任务耗时之和的比较**。它不能直接证明完整 Python API 延迟、所有输入分布、整个 DSA 模型或全部 AscendC 支持范围都更快。

历史性能表的确有 **6/16 个规格达到或超过 AscendC，六个全部是 decode**。其中 `decode_b4_h128` 只有约 **1.010×**，应当看成接近持平；其余五个 decode 优势约为 1.50×～2.18×。所有八个 prefill 规格仍慢于 AscendC。

精度方面，附件把 FP32 softmax 的指数结果转为 BF16 后送入 PV。**这不是 TLA 单方面增加的低精度捷径**：附件引用的官方 AscendC 源码中也明确采用 BF16 P、FP32 运算状态和 FP32 matmul 累加，并使用相应的舍入方式。需要保留的证据边界是：公开源码 revision 不等于当时已安装 CANN 二进制的精确来源，因此不能据此宣称两者每条指令和每个中间舍入都完全一致。

本轮已经补齐 16 个规格的精度、严格位级等价、可执行代码段和核心配置证据。**相同代码段不等于每次测量都绝对零退化**；共享环境和采样顺序的影响必须保留在结论里。新增证据及三批计时的区别见[第 9 节](#9-本轮重新验证得到的证据)。

## 2. 比较的到底是什么算子

本实现计算的是已经给定稀疏索引的 MLA attention 阶段。对某一个 batch、query 和 query head，令 `I[j]` 是第 j 个选中的 KV token：

```text
score[j] = (dot(Q, KV[I[j]]) + dot(QRoPE, KRoPE[I[j]])) / 24
max      = max_j score[j]
sum      = Σ_j exp(score[j] - max)
output   = Σ_j exp(score[j] - max) * KV[I[j]] / sum
```

本 benchmark 约定使用 `1/sqrt(576)=1/24`，两边传入同一系数。KV 同时提供 key 的内容部分和 value，这符合 MLA absorb 模式。但不能据此推导所有 MLA 模型都应按吸收后的 512+64 维重新缩放：真实模型应保留其原始 attention scale，可能还包含 YaRN 修正。官方 DeepSeek-V3.2-Exp 实现按原 QK head 维度计算 scale，吸收到 latent 空间后继续使用它。[模型实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)

原生调用在附件 `code/evaluate.py::native`：

- `query=q, key=kv, value=kv, sparse_indices=ids`；
- `query_rope=qr, key_rope=kr`；
- `attention_mode=2`、`sparse_mode=0`；
- `return_softmax_lse=True`，其他布局参数采用默认 BSND。

官方接口说明确认：`attention_mode=2` 对应 MLA absorb，`sparse_mode=0` 不额外增加 causal mask；`return_softmax_lse=True` 实际请求的是 `softmax_max` 和 `softmax_sum`，而不是一个 log-sum-exp 张量。因此附件对三个输出的解释是正确的。[CANN 9.1 接口文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnSparseFlashAttention.md)、[官方仓库接口定义](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/sparse_flash_attention/docs/torchapi_sparse_flash_attention.md)。

两边都没有计算 lightning indexer 或 TopK。索引生成不计时，对这个 attention 子算子的对比是合理的；若声称“整个 DSA 端到端加速”，就必须另计 indexer、TopK、KV cache 更新及模型其他部分。

## 3. 精度路径是否对齐

### 3.1 明确核对到的 AscendC 源码

审计读取了官方 `cann/ops-transformer`，并额外 fetch 了附件引用的固定 commit：

```text
74dc469b5744bea3b5bd06d249fb59747195bddd
```

以下结论依据这个固定 revision，而不是随时变化的 master：

| 项目 | 附件 TLA | 官方 AscendC 源码证据 | 判断 |
|---|---|---|---|
| 输入与输出 | BF16 Q/KV/RoPE，BF16 output | `sparse_flash_attention.cpp:95-100` 的 BF16 实例采用 `bfloat16_t, bfloat16_t, float, bfloat16_t` | 类型层级一致 |
| QK 累加 | FP32 L0C | Cube service 使用模板累加类型 `T=float` | 没有发现 TLA 降低累加精度 |
| softmax | FP32 max、sum、指数与重标定 | `ProcessVec1Vf` 要求 `T=float`；VF 中使用浮点寄存器和 `ExpSub` | 同一精度类别 |
| PV 的 P | FP32 exp 后 `vtruncf` 转 BF16 | `CastStoreExp128` 的 BF16 分支显式把 FP32 exp 转为 BF16 | BF16 P 不是 TLA 独有近似 |
| P 的舍入 | 保存的 IR 为 `<round>, true` | `castTraitZero/One` 使用 `CAST_ROUND` 和 `SAT` | 舍入与饱和策略相符 |
| PV 与输出状态 | FP32 累加，最终除以 denominator 后转 BF16 | 原生模板使用 FP32 中间类型、BF16 输出 | 没有发现额外低精度输出状态 |
| HF32 | 新算子 IR 不带 `enable_HF32`，参考关闭 `allow_hf32` | 计数器记录原生与 TLA 均为 `HF32 Eligible=NO` | 已有记录未显示 HF32 捷径 |

可复核的官方文件：

- [BF16 kernel 实例](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp)，95～100 行。
- [Cube matmul 路径](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla_arch35.h)，260～270 行的 QK，346～348 行的 PV。
- [FP32 softmax 类型约束和重标定](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h)，119～132、147～167 行。
- [softmax exp、sum 及 P 转换调用](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/common/op_kernel/arch35/vf/vf_basic_block_aligned128_no_update_sfa.h)，59～72 行。
- [FP32 exp 转 BF16 并存储](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/common/op_kernel/arch35/vf/vf_basic_block_128_common_sfa.h)，25～46 行。
- [转换舍入属性](https://gitcode.com/cann/ops-transformer/blob/74dc469b5744bea3b5bd06d249fb59747195bddd/attention/common/op_kernel/arch35/vf/vf_basic_block_utils.h)，48～59 行。

### 3.2 合并 576 维与分成 512+64 并不等于降精度

`fastgather`、`wide_fastgather` 和 `bn256` 把内容与 RoPE 放到同一个 576 维 L1 布局中，由一次逻辑 MMAD 求 QK。`ragged` 则先做 512 维乘积，再用 `clear_accum=False` 把 64 维 RoPE 乘积累加到同一个 FP32 L0C。它没有在两部分之间转成 BF16。

官方 Cube 路径在 `HAS_ROPE` 时明确设置 `mm1SingleK=576`。附件 FP32 reference 则调用两次矩阵乘法，再把两个结果相加。因此正常的归约顺序差异可能导致末位不同；允许各自采用更好的 tiling 时，不应要求所有中间结果逐位相等。这里要验证的是同一数学问题、同一精度类别，以及双方在同一误差要求下都合格。

这段说明不能替代安装二进制的版本定位：附件使用的编译器是 NPU-IR `ff04a33ad9cd`，公开 AscendC 源码是上述 `74dc469b`。若要做到二进制级精度路径确认，还应锁定实际 CANN 算子包版本和 tiling key，检查其反汇编或匹配版本源码。

### 3.3 现有精度判据有一个重要限制

附件 `validate_extended.py:47-78` 用的是：

```text
E = max(abs(actual - reference)) / max(abs(reference))
```

其中 output 阈值为 0.005，max/sum 阈值为 0.0001。检查覆盖全部元素，并额外记录 RMS，但 RMS 不参与通过条件。

这个全张量归一化指标能说明绝对误差相对整个张量的最大尺度较小，**不能替代逐元素 `atol + rtol*abs(reference)`**。例如一个张量里最大值为 1000，另一个本来接近 0 的元素误差达到 1，也可能仍通过 0.005 的判据。不同 query/head 的数值尺度差异较大时，最好再按行计算误差。

附件报告已经承认这是研究约定，而不是 CANN 官方验收标准。因此现有“全部通过”是真实的，但只能解释为“通过了这套公开写明的研究阈值”。它不等于已经证明所有模型场景的精度，也不等于逐元素 allclose。

## 4. 哪些优化是正常优化，哪些说法需要收紧

### 4.1 没有发现问题的部分

- **按形状选择 BM/BN/split**：这是正常的静态特化。host dispatcher 使用张量维度，不使用随机种子或参考答案。
- **相邻两个索引排序后使用一条双行 DMA**：同时重排对应 key/value 的位置。attention 对所选 token 顺序不敏感，改变求和次序不等于删掉 token。
- **重复索引**：重复 token 仍参与多次权重计算。某些路径只搬一次数据再复制，减少的是访存，不是数学上的重复权重。
- **快慢 gather 路径**：代码实际检查索引是否有效、是否存在相邻重复，并在不满足快路径条件时回退。它没有把“测试数据通常有效”当成省略运行时检查的理由。
- **BF16 P 和 FP32 在线状态**：官方源码采用同类路径；没有发现跳过 `exp(old_max-new_max)` 或冻结 max 的实现。
- **split merge**：精度检查包含合并结果；性能记录在 split 大于 1 时包含 `sparse_mla_kernel` 和 `merge_partials` 两个任务。
- **typed AVE/HIVM 适配器**：附件的私有 Python 适配发射真实的 native op，并由正常编译流水内联。审计的 25 个二进制/IR 记录中没有残留扩展 helper call。这是公共 API 缺口的兼容方案，不是改写答案或二进制指令。

附件报告中“不会根据输入值选择算法”需要更准确地表述：**dispatcher 只按形状选 kernel，但 kernel 内会按索引内容选择等价的 gather 快慢路径**。后者是合法优化；不应把它说成完全不存在数据相关控制流。

### 4.2 不能夸大的范围

- `normal` 性能输入来自标准正态随机数，索引是 `randperm` 截取，全部有效且唯一。因此表格反映的是这一类随机稀疏访存，而不是重复率高、连续块多、长尾分布等所有模型输入。
- `uniform`、`peaked`、`duplicates` 和 `invalid` 做了精度检查，但没有进入最终性能比较。这不是造假，只是性能覆盖范围有限。
- TLA 的 `invalid` 检查支持有效前缀加 `-1` 后缀；附件记录中原生对这 16 条检查未通过同一参考判据。它们没有被拿来计算“加速比”，所以最终性能表没有借原生错误输出获利。仍不能据此直接断言 AscendC 有 bug，需要核对确切版本契约。
- 全无效行、任意 tail、paged/causal、indexer、反向均不在附件验证范围内。不要把这个研究 kernel 当成完整原生 API 的无条件替代。

## 5. 计时是否公平

### 5.1 已确认的事实

`bench_extended.py:90-139` 对同一规格复用相同输入，轮换三个实现的测量顺序。每个实现每轮预热，随后测 40 次，共三轮。每个规格在同一张 NPU 上比较，前七组来自 NPU6，另外九组来自 NPU5。不同规格可以使用不同卡，但不能拿不同规格的绝对时间推断硬件或算法优劣。

最终计时实际调用的是 `screen.py::profile`。`screen.py:62-72` 从 trace 中提取 AI Core/Vector 等 kernel 任务，然后把一次调用里的 `dur` 相加。原始记录证明：原生为一个 `SparseFlashAttention` 任务；TLA 在 split 大于 1 时是主核加 merge 两个任务，merge 没有被遗漏。

### 5.2 这不是端到端调用延迟

任务时长之和排除了 CPU wrapper、内存分配、CPU 到设备的分发开销，以及两个设备任务之间的空隙。这是可以公开报告的设备工作量指标，但必须使用这个名字。把一个 kernel 拆成两个 kernel，可能增加任务间空隙；当前指标不会把这部分成本算进去。因此它不能单独证明用户真正看到的 API 延迟也有同样的 2× 优势。

后续应同时记录两类互不混用的指标：

1. **sum of device task durations**：继续用于和附件表格衔接，并纳入所有任务。
2. **一次完整调用的设备区间或端到端延迟**：从本次第一个任务开始到最后一个任务结束，或在同一 stream 上用事件包围完整调用，必要时另外报告同步后的主机 wall time。应该保留 stream/correlation 信息，避免只按任务数量机械分组。

CANN 9.1 文档对 `return_softmax_lse=True` 的 graph 模式有支持限制。不能为了图计时把原生的 max/sum 关闭、却让 TLA 继续返回它们，也不能把 graph 与普通调用的结果放进同一列。若当前安装版本不支持该组合，应使用普通调用的事件或时间线测量。[CANN 9.1 接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnSparseFlashAttention.md)。

### 5.3 核心数与共享环境

TLA wrapper 默认使用 28 个 Cube program，对应 56 个 Vector sub-block；merge 使用 56 个 Vector program。附件 `pipe-counters.json` 对 `prefill_q16` 给出了直接证据：原生和 TLA 主核都是 28 AIC / 56 AIV，merge 为 56 AIV。

但附件最终性能 JSON 的每轮 trace 摘要只保存任务名和时长，没有保存所有 16 个规格的 Block Num、Mix Block Num 和 tiling key。已有资料没有显示 TLA 偷用了更多核心，也没有证据证明原生被故意限制。这个附件证据缺口已在本次复测中通过逐规格 CANN CSV 补充，见第 9.2 节；内部 tiling 允许各自选择。

设备快照显示这是共享服务器。轮换顺序与多轮中位数能缓解部分漂移，但不构成置信区间，也不能消除温度、频率、缓存状态和其他卡活动的影响。1% 左右的差异不宜称为显著提升。新旧 kernel 性能无损验证应使用同一机器、同一软件环境和交错顺序，不能把本次测量直接与附件的绝对微秒数相减。

## 6. 原始数据是否支持附件报告

本次静态复算得到：

- `manifest.json` 的 **181 项文件**全部存在，字节数和 SHA-256 全部匹配。
- `performance-raw.json` 有 **48 条记录 = 16 规格 × 3 实现**。
- 每条记录 **3 轮 × 40 个样本**，合计 **5760 个样本**。
- 所有单轮中位数、三轮中位数与 `performance.json` 一致。
- `accuracy-selected.json` 有 **96 条记录、204 次 TLA 检查**，全部通过附件阈值。
- 非 invalid 的 AscendC 精度记录全部通过；invalid 原生失败记录为 16 条。
- 所有最终 optimized 性能记录的 binary SHA-256，都能在同一规格的精度验证集合中找到。
- `ir/audit.json` 对应的 **25 个 npubin** 与记录的 SHA-256 全部一致。
- 全部精度记录中的 kernel 源码 SHA-256 与交付的对应 Python 文件一致。

因此没有发现“测的是快版本，验证的是另一个版本”或手工篡改汇总数值的证据。

下表是附件历史记录的复算结果，不是此次重构后的实测。单位为微秒，加速比是 `AscendC / TLA`，大于 1 才表示 TLA 更快。

| 规格 | AscendC | 附件最优 TLA | AscendC / TLA |
|---|---:|---:|---:|
| decode_b1 | 52.693 | 26.1785 | 2.013 |
| decode_b8 | 53.9995 | 82.1365 | 0.657 |
| decode_b32 | 144.202 | 253.414 | 0.569 |
| decode_tp8 | 61.671 | 41.062 | 1.502 |
| prefill_q16 | 99.2295 | 125.280 | 0.792 |
| prefill_q64 | 235.9025 | 395.268 | 0.597 |
| prefill_tp8 | 272.4775 | 308.769 | 0.882 |
| prefill_k512 | 78.0545 | 112.8045 | 0.692 |
| decode_b4_h128 | 53.168 | 52.6175 | 1.010 |
| decode_b4_h64 | 62.5565 | 37.395 | 1.673 |
| decode_b4_h32 | 60.8355 | 31.0115 | 1.962 |
| decode_b4_h16 | 60.3225 | 27.667 | 2.180 |
| prefill_b4_h128 | 124619.674 | 153876.737 | 0.810 |
| prefill_b4_h64 | 76528.9535 | 96089.3715 | 0.796 |
| prefill_b4_h32 | 73917.1625 | 83334.333 | 0.887 |
| prefill_b4_h16 | 71909.221 | 75159.5475 | 0.957 |

`decode_b1` 的附件基线和最终配置使用同一份 ragged 实现。26.220 与 26.1785 微秒的差别来自两次测量，不应记作这轮修改的优化收益。`decode_b4_h128` 最终 TLA 比附件基线略慢约 0.03%，也不存在可声称的改进。

## 7. 静态审计提出的复测要求

以下是审查附件时提出的要求；哪些已经完成、哪些仍有边界，见第 9 节。

| 优先级 | 事项 | 为什么要补 |
|---|---|---|
| 必须 | 同机同软件环境运行原始附件、重构版、AscendC | 排除 torch_npu、编译器、频率和卡型号变化 |
| 必须 | 所有 16 规格输出三张量，并核对每项实际 kernel/binary | 防止只验证主输出或换了未验证的编译变体 |
| 必须 | 增加逐 query/head 误差、绝对误差、RMS 和共同阈值 | 全张量最大值不能掩盖小尺度行的错误 |
| 必须 | 同时报告任务耗时之和与完整设备区间 | split merge 的启动间隔影响实际延迟 |
| 必须 | 逐规格记录 AIC/AIV 核数、tiling key、实际任务列表 | 把“相同资源”从推断变成证据 |
| 建议 | 额外随机种子、强抵消、大动态范围、混合重复率、乱序有效索引 | 避免只对常见正态测试分布表现良好 |
| 建议 | 小规格 FP64/CPU 参考交叉验证 | 区分 FP32 reference 的归约误差与 kernel 错误 |
| 建议 | 反复交错 A/B/B/A 测量，报告配对比值和波动 | 共享环境下不能要求每次微秒数完全相同 |

不应为了通过复测临时放宽某一方的精度阈值，也不应在看到结果后删除不利规格。若在当前共享环境下只能证明“没有发现超过噪声的性能退化”，就应按这个结论写，而不是承诺所有未来运行都绝对没有一纳秒损失。

## 8. 针对附件历史数据的准确表述

> 这是一份针对 BF16 D512+RoPE64、已给定稀疏索引的 Sparse MLA attention 实现。附件 16 组规格中，6 组 decode 在三轮设备任务耗时汇总中达到或超过已安装 AscendC，其中一组接近持平。未发现答案缓存、漏算 merge 或额外降低 P/累加精度等 reward hack；BF16 P 与官方 AscendC 源码的数值路径一致。当前记录不等于完整调用延迟，也不等于完整 DSA 模型加速。全张量研究精度阈值已通过；完整原生 API 范围和安装二进制逐指令精度同一性尚未证明。

原始证据目录：`../build/tla/issue105_review_input_20261102/issue105-dsa-tla/`。官方源码审计 checkout：`../build/tla/issue105_ascendc_audit_source_20261102/`；对附件版本应使用 `git show 74dc469b:<path>`，不要把 checkout 的最新 master 当成同一个 revision。

## 9. 本轮重新验证得到的证据

本节依据取回的本轮 JSON、二进制 receipt 和 CANN profiler CSV 摘要，与前面的附件历史表分开。复测使用 Ascend950PR、lpc 环境中的 `torch_npu 2.7.1.post7.dev20260707`；附件环境是 post8，不能把两个环境的绝对微秒数直接相减。原始 TLA、重构 TLA 与原生 AscendC 在本轮同一环境接受同一组输入。

### 9.1 精度和重构等价已覆盖全部 16 个规格

[accuracy-all16](validation/accuracy-all16/) 保存 16 个规格的完整三输出精度。输入族为 `normal`，seed 为 `20261102`。原始 TLA、重构 TLA、AscendC 全部通过同一套公开写明的附件研究阈值；没有为某一方放宽判据。另记录了逐行误差和 RMS，具体值见各性能报告的精度表。

重构版 output 的最坏全张量归一化误差约为 **0.003640**，最坏逐行归一化误差约为 **0.005590**。后一个值大于 0.005，所以不能声称“逐行误差也全部小于 0.005”；0.005 是本次约定的全张量判据，逐行指标只作补充报告。AscendC 在本轮具有相同的最坏 output 逐行误差。这些数据仍不构成官方 CANN 精度认证，也不能代替真实模型精度评估。

早期 JSON 的 `original_refactored_bitwise_equal` 字段采用 `torch.equal`，尽管名字带有 bitwise，本质上检查的是数值相等，不能独自排除正负零等位模式差别。为补上这个缺口，[严格位级复验](validation/bitwise-final-all16.json) 先检查 dtype/shape，再将 BF16 output 看成 int16、FP32 maximum/denominator 看成 int32，比较其完整位模式。**16 个规格、三个输出全部位级相同。**

同时逐项核对了该严格复验中的 original/refactored 主核与 merge `binary_sha256`：它们与 `accuracy-all16`、`timing-all16`、`paired-all16`、最终 `balanced-all16` 对应记录全部一致。精度和这三批性能比较使用的是同一批实际二进制；不是仅拿相同 Python 文件名作关联。所有 main/merge 的非空 executable section 哈希也在原版和重构版之间一致。完整二进制文件可以因函数名等信息不同而不同，这与可执行代码段相同不矛盾。

### 9.2 核心配置与 HF32 的证据已从单规格扩展到全部规格

本轮 Chrome trace 的 args 没有足够的 Block 字段，因此没有从任务名或 grid 推算核数。补充证据来自 CANN `op_summary*.csv`，通过 profiler 进程 ID 关联规格，并保留 CSV 路径、SHA-256 和匹配行数：

- [第一批核心配置](validation/core-configurations.json)：16/16 个规格。
- [逐调用交错批次核心配置](validation/core-configurations-paired.json)：16/16 个规格。
- [最终六种顺序平衡批次核心配置](validation/core-configurations-balanced.json)：16/16 个规格。

三批记录中，原始 TLA、重构 TLA 与 AscendC 主核均报告 `Block Num=28`、`Mix Block Num=56`；TLA merge 报告 `Block Num=56`、`Mix Block Num=0`。所有已记录任务的 `HF32 Eligible` 都为 `NO`。这支持比较采用了相同的主核任务配置，且没有发现一方通过 HF32 获利。这里描述的是 profiler 配置字段，**不等于证明运行中每一时刻都有 28 个物理 AIC 和 56 个物理 AIV 同时活跃**。

### 9.3 三批计时都保留，不能选择性删除不利轮次

第一批在每轮内按实现分组测量，[PERFORMANCE.md](PERFORMANCE.md) 保留全部 16 个规格。共享环境中出现过非常大的轮间波动：即使原版与重构版 `.text` 相同，分组测量的三轮中位数仍可能相差数倍。不能把这样的比值直接解释成重构导致数倍性能退化，也不能删掉较慢轮次后声称无损。

第二批把三个实现放进同一个 profile，逐次交错调用，完整记录在 [PERFORMANCE_PAIRED.md](PERFORMANCE_PAIRED.md)。它缩短了相邻测量的时间间隔，但采用的三个循环排列没有完全平衡“前一个运行的是谁”。在这批结果中，小规格重构版有约 0～3.7% 的观测变慢，最大例子是 `decode_tp8`；这需要如实保留，不能被相同 `.text` 直接抹去，也不能没有进一步证据就全部归因于噪声。

因此又增加全排列平衡采样：三个实现的六种排列完整覆盖，每轮每种实现 30 次，共三轮。**这批全部 16 个规格已经完成**，最终结果单独汇总到 [PERFORMANCE_FINAL.md](PERFORMANCE_FINAL.md) 和 [summary-final.json](validation/summary-final.json)，不与前两批或附件历史数据合并。平衡顺序可以减少一种已识别的偏差，仍不意味着共享设备从此没有波动。

最终重构耗时相对原版的观测变化范围为：任务耗时之和 **−1.192%～+0.142%**，任务起止区间 **−0.787%～+0.971%**，eager 调用 **−0.784%～+2.499%**。变化率按 `100×(重构/原版−1)` 计算，负数表示更快。第二批 `decode_tp8` 约 3.726% 的观测变慢仍保留；最终平衡批次该规格的任务耗时之和变为约 1.192% 更快。这说明采样方法和运行状态会影响观察结果，不能把任一批次的比值直接解释成相同可执行代码发生了同等幅度的算法变化。

按最终任务耗时之和，TLA 胜出的六组是：`decode_b1` 2.018×、`decode_tp8` 1.201×、`decode_b4_h128` 1.028×、`decode_b4_h64` 1.414×、`decode_b4_h32` 1.605×、`decode_b4_h16` 1.779×，比值均为 AscendC/TLA。AscendC 在其余十组领先，即 `decode_b8`、`decode_b32` 和全部八组 prefill。1.028× 这种较小优势尤其需要结合轮间波动理解，不能自动视为显著优势。

完整 eager 口径只有 `decode_b1`、`decode_b4_h64`、`decode_b4_h32`、`decode_b4_h16` 四组比值大于 1，分别约为 1.278×、1.001×、1.154×、1.216×；其中 1.001× 接近持平。不能用 kernel 任务口径的六组胜出替代完整调用口径的结果。

三种指标的边界始终不变：`task_sum_us` 累加已选 kernel 任务，`task_span_us` 计算这些任务从开始到结束的区间；它们都不覆盖未被分组纳入的所有设备活动。`eager_us` 则包围完整 wrapper 调用并等待同步，纳入调度、分配和完成等待等成本。三者必须分别读，不能把 kernel 任务区间冒称完整 API 延迟。

目前能够确认的是：本次重构在所测 16 个规格上保留了实际可执行计算和严格位级输出，最终平衡计时的变化范围如上。结论以三种计时边界及保存的原始轮次为限，不对这些小幅差异作没有统计证据的显著性判断；**不承诺所有运行环境、所有调用均绝对零退化**。

### 9.4 公共 API 路径也做了独立设备验证

旧 wheel 的 legacy 路径与新编译器的 public 路径分开验证，没有把换编译器后的性能混进重构前后比较。`validation/public/` 中 8 条记录全部通过：四个代表规格 `decode_b1`、`decode_b8`、`decode_tp8`、`prefill_tp8` 的 normal 输入，以及前两个规格的 duplicates/invalid 输入。它们覆盖四类实际 kernel 分支，并记录 `no_dsa_codegen_monkeypatch=true`。invalid 仅作边界验证，不拿来宣称相对 AscendC 的加速。

三个新增公共接口的原生 UT 为 **121 项通过**，包含 79 项 host 合同和 14 个配置的 IR/compile/run 三阶段；每个设备配置使用两个种子。源码、精度日志和 pre-commit 证据见[编译器分析](COMPILER_ANALYSIS.md)。社区改动已提交 [Triton-Ascend PR #2940](https://github.com/triton-lang/triton-ascend/pull/2940)。这些证据说明公共接口实现已完成对应验证；不能扩大为全部 16 规格在新编译器上都已性能无损。
