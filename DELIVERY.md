# DSA 审计、重构和编译器接口交付说明

这份交付保留了你附件中选定的高性能算法，重点是把它讲清楚、整理成可维护的代码，并建立能追溯到实际二进制的比较。先读本文件，再按 [README](README.md) 运行；完整中文讲解见 [DSA 从原理到实现](DSA从原理到实现.md)。

## 1. 和 AscendC 的比较是否公平，有没有 reward hack？

**未发现答案缓存、按种子特化、遗漏 merge、偷偷启用 HF32 或单方面降低中间精度等 reward hack。** BF16 P、FP32 softmax 状态及 matmul 累加与核对到的官方 AscendC 源码属于相同精度路径。两边计算同一个已给定稀疏索引的 Sparse MLA attention 子算子，输入、scale、输出语义相同；没有把 indexer/TopK 计入其中一方。

公平结论有明确范围。本轮全部 16 个规格的三输出通过同一套研究误差阈值，CANN profiler CSV 记录主核 `Block Num=28 / Mix Block Num=56`，HF32 均为 `NO`。这些是任务配置字段，不是瞬时物理活跃核数。精度阈值也不是官方 CANN 认证；源码路径一致不等于已经证明安装二进制逐条指令完全相同。

详细证据、历史数据与本轮复测的区别，见 [FAIRNESS_AUDIT.md](FAIRNESS_AUDIT.md)。

## 2. 重构了什么，性能有没有损失？

新的实现位于 [dsa/](dsa/)，通过 [forward](dsa/api.py) 统一调用。主核分成 ragged 与 workspace 两类，共享调度配置和 merge；入口检查、数据布局、同步协议与兼容层都有对应注释。目录结构和逐项重构方法见 [REFACTOR_NOTES.md](REFACTOR_NOTES.md)。

已经完成的等价证据是：

- 全部 16 个规格的原始/重构 main 和 merge 可执行代码段相同。
- [严格位级复验](validation/bitwise-final-all16.json) 对 BF16/FP32 输出转成对应整数位视图后比较，16 个规格、三个输出全部相同。
- 严格复验的每个 kernel 完整 binary SHA 与原始精度、第一批计时、逐调用交错计时及最终平衡计时逐项匹配。早期 `torch.equal` 字段仅说明数值相等，没有被单独当成位级证明。

这支持“重构保留了已测路径的实际计算”，但不能承诺共享机器上每次调用都绝对零退化。第一批分组计时受到明显干扰；第二批逐调用交错后，仍有小规格出现约 0～3.7% 的观测变慢，且三种循环顺序没有完全平衡前序调用影响。我们保留这些结果，并增加六种排列完整覆盖的平衡采样。

**最终平衡采样已完成全部 16 个规格**，详见 [PERFORMANCE_FINAL.md](PERFORMANCE_FINAL.md) 和 [summary-final.json](validation/summary-final.json)。三个实现的六种排列完整覆盖，每轮每种实现 30 次，共三轮。重构耗时相对原版的观测变化为：kernel 任务耗时之和 **−1.192%～+0.142%**，所选任务起止区间 **−0.787%～+0.971%**，完整 eager 调用 **−0.784%～+2.499%**；负数表示更快。这是本次测量范围，不是绝对零退化或统计显著性的承诺。

按 kernel 任务耗时之和，TLA 在六组 decode 优于 AscendC：`decode_b1` **2.018×**、`decode_tp8` **1.201×**、`decode_b4_h128` **1.028×**、`decode_b4_h64` **1.414×**、`decode_b4_h32` **1.605×**、`decode_b4_h16` **1.779×**。AscendC 在另外十组领先，包括 `decode_b8`、`decode_b32` 和全部八组 prefill。完整 eager 口径只有四组 TLA 比值大于 1，其中 `decode_b4_h64` 约 **1.001×**、接近持平；不能把六组 kernel 优势说成六组完整调用优势。

最终表分别列出上述三种计时。前两者只覆盖所选 kernel 任务，不是所有设备活动或完整 API 延迟；eager 则包含完整 wrapper 调用与同步成本。

前两批数据分别保存在 [PERFORMANCE.md](PERFORMANCE.md) 和 [PERFORMANCE_PAIRED.md](PERFORMANCE_PAIRED.md)；附件历史表独立保留，不能与本轮绝对时间混算。所有不利轮次都保留，没有挑最快样本作为结论。

## 3. 从零开始读懂这个算子

[DSA 从原理到实现](DSA从原理到实现.md) 是完整中文教程，依次解释 attention、MLA、稀疏索引、RoPE、online softmax、split merge、Ascend950 的存储与执行单元，再对应到真实 kernel 的 gather、布局转换、UB 复用、MMAD 和同步流水。

教程特别区分数学恒等式与浮点实现：BF16 P 为什么合理、什么时候需要 FP32 状态、重排索引为什么保留 attention 数学、重复索引为什么不能去重。它也解释了为什么附件固定的 `1/24` 只是本 benchmark 约定，真实 MLA 模型应保留模型自身的 scale。

## 4. 普通 Triton-Ascend 能否运行，社区需要补什么？

本次检查的旧 wheel 可以在**不改安装文件**的情况下借助 `legacy` 适配层运行，但适配层使用了进程内 CodeGenerator 扩展和 typed IR 构造，因此不能称为“仅靠现成公开 API”。这一区别与精度是否公平是两回事。

新的 `public` 路径去掉这套私有适配。当前 main-dev 已有 MMAD、subview 和所需 SIMD 运算；本次补齐三个公共接口：`tla.scalar_load`、`tla.byte_view`、`tla.load_gm_to_ub_strided`。具体合同、实现位置、为何需要这些接口，以及未来编译器如何更好地支持算子开发，见 [COMPILER_ANALYSIS.md](COMPILER_ANALYSIS.md)。

代码已提交社区：[Triton-Ascend PR #2940](https://github.com/triton-lang/triton-ascend/pull/2940)。最终接口测试 **121 项通过**，修改文件通过 pre-commit；原生构建和日志保存在 `validation/compiler/`。公开 API 路径还有 **8 条设备验证通过**，覆盖四类代表性 kernel，以及 ragged/workspace 的重复索引和部分无效索引输入。

旧编译器的 16 规格重构比较与新编译器的 public 验证分别保存。public 精度通过不等于已经测完新编译器全部规格的性能，也不等于本次接口 PR 自身带来加速。复现命令、版本要求和模式选择都在 [README](README.md)。

最终 PR head 为 `564a1e5b014adbc91b533c81dc699ec4425c1c99`；产品代码之后仅追加 README 拼写修复。最终 head 的社区 pre-commit 和 title 检查通过，文档与集成 CI 尚在运行，PR 尚未合并。详见 [CI 状态](validation/compiler/CI_STATUS.md)。
