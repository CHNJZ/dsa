# DSA 对 Triton-Ascend 的要求：哪些已经具备，哪些还需要补齐

本文以附件 `issue105-dsa-tla-20261102.zip` 中实际选中的四类 kernel 为依据，
区分“安装目录没有改动”“只使用正式公开接口”和“编译器本身支持某个操作”。
这三件事不能混为一谈。代码检查基线是 2026 年 11 月 2 日重新获取的
`triton-lang/triton-ascend main-dev@0988a7ed24e68253cd3c08eac023ba3d3c9a496e`。
设备运行和性能结果以本目录最终验证记录为准；下文不会把源码检查当成设备验证。

## 1. 现成的、没有修改安装文件的包能运行吗？

附件不是仅由公开 TLA API 写成的程序。它自带两层扩展：

- `tla_native_ext.py` 在运行时替换 `CodeGenerator.__init__`，把编译模块挂到
  builder 上；再解析临时 MLIR 文本，把已有的 AVE/HIVM 操作插入编译模块。
- `dsa_ext.py` 用同一个入口构造 `memref.load`、`memref.view`、带动态行距的
  `memref.reinterpret_cast`、带动态行数的 `memref.subview` 和 `hivm.hir.load`。

因此，一个安装目录完全未改动的兼容 wheel，仍然可能运行附件。但是运行时
已经应用了本地编译器适配代码，不能称为“仅靠公开 TLA 接口”。这类适配依赖
Python CodeGenerator 的内部结构、MLIR 类型打印格式、dialect 注册情况以及
inliner/canonicalizer 的行为；换一个 wheel，兼容性没有保证。

本次实际检查的安装包是 `triton_ascend 3.6.0.dev20261031+git8ca956c1`，
Python 3.11.15、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post7.dev20260707。
`validation/environment-installed.json` 记录了 TLA 源文件、`code_generator.py`
和 `libtriton.so` 与本机 wheel RECORD 中哈希一致，未发现这些已检查安装文件
被改写。RECORD 不是独立签名的上游证据，未记录哈希的 pyc 也不在认证范围内。
这套旧安装缺少 `scalar_load`、`byte_view`、strided DMA、`subview`、SIMD
`vmuls`/`vexpdif` 等公开入口；MMAD 也还没有当前 main-dev 的完整参数。
所以旧 wheel 可以依靠附件自带适配层运行，但不能去掉适配层后原样使用全部
公开接口。安装版本、当前源码能力和自带适配层应分别写清楚。

这些适配确实构造原生指令，而不是在计时外预计算答案，也没有用近似数学函数
冒充原始操作。是否构成不公平比较，还必须核查算子合同、精度和计时范围，见
`FAIRNESS_AUDIT.md`。使用私有编译器接口和改变数学精度是两类不同的问题。

## 2. 以当前 main-dev 为准，已经具备了什么？

最新源码包含以下正式能力，不需要在这次 PR 中重复实现：

| 附件中的适配 | 当前正式入口 | 说明 |
|---|---|---|
| `ext.matmul` | `tla.cube.mmad` | 已支持 A/B 转置及 actual extents；按转置后的逻辑维度检查输入 |
| `ext.vmuls` | `tla.simd.vmuls` | 原生向量乘标量 |
| `ext.vexpdif` | `tla.simd.vexpdif` | 原生融合指数差指令；不能随意换成先减再 exp 并宣称逐位等价 |
| buffer 子视图 | `tla.subview` | 直接表达 `memref.subview`；保存原来的行距 |
| 连续子视图 reshape | `tla.reshape` | 中间类型和结果类型保留 offset |

这批能力分别已包含在 main-dev 的 SIMD 更新和 `0988a7ed2` 的 subview/MMAD
更新中。不能因为较早安装的 wheel 没有它们，就说“当前社区源码仍然缺失”。

当前基线的 LLVM 源码指纹为 `f6ded0be897e2878612dd903f7e8bb85448269e5`，
NPU-IR 子模块为 `4eb12142e16fc8c7ed33c3b568658b773ae0d1c9`。
这两个依赖与前一轮远端隔离 native build 相同，可以复用依赖和构建缓存。

## 3. 这次补齐的三个接口

### 3.1 从 UB 读取一个标量：`tla.scalar_load`

```python
index = tla.scalar_load(index_buffer, position)
index = tla.scalar_load(index_matrix[row, column])
```

SIMD 程序常先在 UB 中准备一批索引、标志或计数，随后标量控制流需要拿其中
一个值决定从 GM 读哪里、读几行。原来只能通过适配代码手写 `memref.load`。
新入口直接在原生 C++ builder 中生成这个操作。

本次公开合同限于 int32 UB 数据。每个维度需要一个 Python 整数，或 signed
int32/int64 标量 tensor。静态越界会在前端报错，动态越界由调用者避免。源 buffer 的 offset 和 stride 都保留。
它在 vector core 的标量控制流中使用，不是 `tla.scope.simd()` 中的向量加载。

读取标量不等于等待数据。生产者如果是 DMA 或 SIMD 写入，必须先完成。
普通自动同步模式中，编译器应当从真实内存读写建立依赖；手工管理流水线时，
程序仍须提供正确的 ready/free 协议。单元测试专门覆盖“SIMD 写 UB，随后标量
读同一存储”和“DMA 写 UB，随后直接标量读取”两种场景。

这里刻意不接受窄整数、unsigned tensor 或 bool 索引。`arith.index_cast` 会做
符号扩展，直接转换 uint8(200) 会变成负数。Python 常量限定在 signed int64
范围内，且先正确转换成 int64，避免 2**31 默认推导成 uint32 后发生同类错误。
这些拒绝条件和转换都有回归测试；原有 view 接口的合同没有被扩大或改变。

### 3.2 一块字节存储的不同类型视图：`tla.byte_view`

```python
storage = tla.alloc((1024,), tl.int8, space="ub")
fp32_tile = tla.byte_view(storage, (128,), tl.float32)
bf16_tile = tla.byte_view(storage, (128,), tl.bfloat16)
```

DSA 的不同阶段有时不需要同时保留 score、PV 临时值和最终 BF16 输出。
它们可以复用同一块 UB，但这必须在 IR 中表达成真实别名关系。新接口直接生成
`memref.view`，既不搬运数据，也不转换数值。

源必须是 offset 为零、连续的一维 int8 UB buffer。shape、dtype 和 byte offset
都是编译期值；所需字节数不能超过原分配，起点必须按结果元素大小对齐。
更严格的 DMA/SIMD 硬件对齐规则仍由具体使用操作约束。

两个视图互相重叠，并不表示两个值可以同时保持有效。复用前必须确认前一个
阶段最后一次读取已经完成。测试先写 FP32、读出，再用 BF16 覆盖部分相同字节，
最后检查完整底层存储和未触碰区域，验证别名和生命周期没有被隐藏。

### 3.3 动态行距和行数的 GM→UB 搬运

```python
tla.load_gm_to_ub_strided(base, destination,
                        offset=first_row_offset,
                        row_stride=distance_in_elements,
                        rows=number_of_rows)
```

这个接口适合把两条相距较远、各自连续的 KV 行放进一个 UB 小块。`rows` 默认为
目标 buffer 的全部行数，也能是运行时标量，因此重复索引场景可以只读一行，
随后由 kernel 生成第二份逻辑数据。它不负责排序索引或去重，也不改变重复索引
在注意力中的权重。

源是一维、单位步长的 GM view；目标是二维、行内连续的 UB view。数据类型相同，
目标行距可以大于逻辑列数。offset 和 row stride 都以元素为单位。三个动态
参数同样只接受 signed int32/int64 标量；Python 整数须能放入 signed int64。
前端检查已知的 source/destination 容量、非负 offset、行数以及行距；动态值
必须由调用者保证有效，具体 DMA 数值范围和硬件限制仍归后端检查。

这里有一个容易写错的细节：`memref.reinterpret_cast` 中的 offset 会替换源
descriptor 的 offset，而不是相对当前 view 自动累加。实现先用
`memref.extract_strided_metadata` 读取源 offset，再加上本次相对 offset。
目标通过 `memref.subview` 限定行数，保留目标已有的 pitch 和 offset。
测试把指针加法、GM entry offset、source subview offset 和本次相对 offset
叠在一起，并检查非紧凑目标及其保护区。

完整 DSA 集成验证还暴露了一个容易漏测的情况：当源是零 offset 的 GM entry，
canonicalizer 会把新的 reinterpret_cast 直接接到 pointer_cast 上，而现有
Triton-to-Linalg 的 pointer-entry 转换要求 shape 静态。直接把动态 rows 写进
reinterpret_cast 会触发断言；最初仅有多层 offset 的测试没有覆盖这条路径。
这不是用了错误的 `triton.jit`：原 kernel 使用的 `TLAJITFunction` 就是正式
TLA 入口，离线最小复现也带有 `hacc.mode = "tla"`。

修复只调整新 builder 的构造：先生成静态行容量的 source descriptor，再用
动态 rows 的 `memref.subview` 表达本次实际要读的范围。唯一的 DMA 读取这个
子视图，不读取整个容量。没有修改通用 Triton-to-Linalg pass，也没有要求
调用者为未读取的行预留有效内存。新增 FP16/BF16 零 offset 测试，尤其让一行
拷贝位于 source allocation 的末尾，假设多读第二行就会越界；目标保护区也
保持检查。最终还需核对生成的 DMA count，不能只凭输出相等证明没有多读。

我们没有为 DSA 新造 `load_pair` 和 `load_rows` 两个专用算子，也没有顺手把
整个 TLA buffer 类型系统改成动态 shape。动态尺寸只存在于这一次搬运产生的
原生 memref view 中，减少对其他算子的影响。

## 4. 对 agent 来说，为什么这比解析 IR 文本更好？

原来的适配把几种责任混到 kernel 文件中：检查 Python 类型、拼 MLIR 字符串、
创建临时文件、解析 helper 函数、处理 block 所有权、绕开 AVE inliner 的限制。
这些工作不属于注意力算法。agent 修改其中任何一处，都可能产生难以定位的
编译失败，甚至把 view offset 或内存依赖弄丢。

正式入口把职责分开：

1. Python API 检查用户可见的 shape、dtype、内存空间和静态边界。
2. C++ builder 生成已有的 MLIR/AVE/HIVM 操作，不依赖类型的文本打印格式。
3. 内存规划和同步分析继续看到真实的别名、读写和循环关系。
4. `tla_ut` 分别验证 IR、二进制生成和真实 NPU 数值结果。

这不会自动保证任何 kernel 都正确，但能把大量错误变成接口处清楚的诊断，
而不是等到大 kernel 编译失败或 NPU 上出现随机错误。

## 5. 还值得做哪些编译器工作？

以下是后续方向，不能当成本次已经实现或已经带来加速的成果。

### 5.1 给内存复用建立可检查的合同

应该让工具能回答：两个 view 是否属于同一 allocation？实际字节范围是什么？
前一阶段的最后一次读是否完成？动态 offset 无法证明时应明确报告未知。
不要用“地址不同”推断生命周期安全，也不要因为用了 byte view 就把 alias
关系当成 no-alias。现有 `memref.view` 可以表达事实，后续工作应优先完善其分析。

### 5.2 帮助识别索引搬运可以合并的条件

这份 DSA 显式把两条行地址转换成起点和正 stride，再发一次 DMA。编译器若要
自动做这类合并，必须证明行宽、步长、边界、重复索引行为和目标布局合法，且
减少搬运指令的收益大于标量地址计算开销。可以先从受限的“两行、相同长度、
相同 dtype”模式做优化，再用不连续、重复、倒序和边界索引测试守住语义。

### 5.3 为显式 C/V 流水线提供诊断

agent 最容易混淆的是“槽位可以重用”和“读取已经结束”。工具应分别展示
Cube/Vector 的生产、消费、ready、free 和 drain，而不是只展示两个缓冲区地址
在轮换。更好的 IR 检查和可视化应给出具体依赖证据，不应凭一个编译选项宣称
流水线一定安全或一定更快。

### 5.4 把布局转换做成可组合、可验证的构件

当前 kernel 中 BF16 pack、NZ 排布、转置和分片散落在较长的 SIMD 代码里。
通用构件需要同时写清逻辑矩阵形状、物理存储形状、每个子核负责的行，以及
padding 的数值。可以先提供可执行参考和标准样例，再考虑自动选择实现。
尤其不能把某个 DSA 的固定 shape 当成所有矩阵布局的通用规则。

### 5.5 编译收据应直接记录实际生效的 TLA 路径

附件里缓存 metadata 的 `enable_tla_mode` 曾与最终编译命令不一致，导致复查者
必须从 IR 和命令行反推实际路径。一个可靠的收据应记录输入源码指纹、实际
native 库、最终选项、目标芯片和生成二进制之间的关系。这个问题有助于复现和
审计，但不应和本次三个内存接口塞进一个未经验证的大重构。

## 6. PR 与验证边界

独立开发目录是 `build/tla/pr_dsa_interfaces_20261102`，分支
`codex/tla-dsa-interfaces`。核心改动放在现有 TLA language 目录和原生 builder，
三阶段测试放在 `tla_ut/vector/`、`tla_ut/dma/`，拒绝非法参数的 host 测试放在
`tla_ut/memory_frontend_contract.py`。

社区 PR 已提交：[triton-ascend #2940](https://github.com/triton-lang/triton-ascend/pull/2940)，
提交为 `4a96387550889968145e01e2a66fab99f5cf9341`。可重现的 git format-patch
位于 [compiler/0001-tla-dsa-memory-apis.patch](compiler/0001-tla-dsa-memory-apis.patch)，
应用、构建与复验步骤见 [compiler/README.md](compiler/README.md)。

这些接口的单元测试通过，不等于完整 DSA 的性能必然不变。重构前后必须在同一
设备、同一编译器和同一计时方法下比较；切换到新编译器的结果要单独记录，
不能把编译器版本差异藏进“代码重构无损”的结论中。

### 已完成的验证

本次在远端以当前 main-dev 基线真实重建 C++ native extension，让 Python 加载
独立 `build/lib.linux-x86_64-cpython-311/triton`，没有覆盖 lpc 的安装目录。
最终源码已经对照 `validation/compiler/source.sha256` 逐个检查，13 个文件
全部一致；C++ 显式执行了仓库要求的 `clang-format -i -style=file`，随后修改
文件的完整 pre-commit 和 `git diff --check` 通过。

最终接口测试 **121 项通过**：79 项 host 合同，加上 14 个配置分别执行 IR、
compile、run 三阶段。每个 run 配置使用 seed 7 和 19，总共 **28 组输入**，
全部通过 exact equality，覆盖 typed output、完整 byte backing、未触碰的
保护区、动态行数以及 SIMD/DMA→scalar 数据依赖。strided DMA 的 rows 显式
禁止按数值专门化，并检查实际编译签名仍为 i32/i64，因此 rows=1 的末行用例
也确实走运行时 count 路径，不会悄悄退化为编译期常量。

原始最终日志是 `validation/compiler/native-device-ut-final.log`，结果为
`121 passed, 7 warnings in 22.42s`。单独的 CPU 离线检查也有 14 个配置通过，
覆盖 AST→TTIR→交给 NPU-IR 的 ttadapter 边界，不只是 mock 调用检查。

完整 DSA 的四种代表性公开 API 路径也已在设备验证：`decode_b1`、`decode_b8`、
`decode_tp8`、`prefill_tp8`，覆盖直接 NZ、不等长 split、不同 PV 宽度/UB 复用、
BN256 等分支；需要 merge 的规格连同 merge 一起验证。它们都通过既有 benchmark
的数值检查。该门槛是研究验证门槛，不能当成官方 CANN 精度认证。

公开路径没有导入旧 DSA 扩展模块，也没有应用 DSA 的 CodeGenerator monkeypatch。
官方 Ascend backend 自身会在第一次编译时延迟包装 CodeGenerator 以添加目标
属性；验证脚本核对了该包装来自安装包，而不是把所有正常初始化都误判为本地
补丁。四条完整路径也通过了不替换 IR 的离线 lowering，记录在
`validation/compiler/public-native-offline.log`。

接口精度通过和完整 DSA 数值通过，不等于新编译器版本一定带来加速。重构前后
的性能检查单独记录，不能把编译器版本差异藏进“重构无损”的结论。本 PR 本身
没有完整 DSA 性能提升声明。

命令和日志保存在 `validation/compiler/`，英文社区 PR 说明在
`compiler/PR_DESCRIPTION.md`。最终测试覆盖了零 offset 的新增回归，旧阶段的
115 项记录只保留为工作历史，不作为最终版本的验证依据。

截至 `2026-11-03T01:28:39.974Z`，最终 PR head `564a1e5b014adbc91b533c81dc699ec4425c1c99` 的全仓 pre-commit 和 title 检查通过；文档与集成 CI 仍在运行。首轮文档门禁发现测试 README 原有的 `npuir` 拼写问题，本次以独立文档提交修正为 `NPU-IR`，同款五项文档检查和全仓 pre-commit 已复验通过。

121 项验证对应产品提交 `4a96387550889968145e01e2a66fab99f5cf9341`；后续仅 README 拼写改变，产品、UT 和 native 库没有改变，因此没有重跑无关设备测试。两份源文件清单、最终 CI 快照及日志见 [CI_STATUS.md](validation/compiler/CI_STATUS.md) 与 [源码对应记录](validation/compiler/source-provenance.json)。尚在运行的社区检查不属于已通过结论。
