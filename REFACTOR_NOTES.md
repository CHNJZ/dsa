# DSA 重构说明：让结构清楚，同时保留已测调度

这次重构保持原附件选择的 16 个规格、输入输出含义和数值运算不变。
它整理代码结构，不借机重新调参。是否保持性能仍须以同机、同输入、
同编译器下的原版/重构版配对实测判断；AST 相同只是其中一层证据。
最终时间和测量边界见 [性能报告](PERFORMANCE_FINAL.md)。

## 从哪里开始读

1. `dsa/api.py`：唯一的 host 入口。它校验张量、查配置、分配工作区，
   然后发射主核；需要分片时再发射 merge。这里没有导入基准或精度脚本。
2. `dsa/config.py`：16 个形状与调度表。BM 是 head tile 大小，BN 是
   被选 KV 行的 tile 大小，split 是 KV 方向的任务分片数。分派只依赖形状。
3. `dsa/workspace.py`：大部分规格使用的主核。沿着注释中的初始化、
   首槽预取、稳态、排空四个阶段读，不必一开始就追每条 SIMD 指令。
4. `dsa/ragged.py`：小 decode 的另一种数据流。它直接在 UB 打包 NZ 到 L1，
   允许 split 之间拥有不同数量的 BN64 tile。
5. `dsa/merge.py`：用每个分片的最大值重新缩放 FP32 分子和分母，最后
   归一化并转 BF16。merge 是计算的一部分，计时必须包含它。
6. `dsa/compat.py`：编译器接口边界。算法文件不再散落文本 IR 构造代码。

调用方式：

```python
from dsa import forward

output, softmax_max, softmax_sum = forward(q, kv, q_rope, k_rope, indices)

# 验证时显式拿到本次调用的主核和 merge 编译对象，不依赖可变全局状态。
outputs, compiled_kernels = forward(
    q, kv, q_rope, k_rope, indices, return_kernels=True
)
```

`output` 是 BF16 `[B,SQ,H,512]`，最大值和分母均是 FP32 `[B,1,SQ,H]`。
两个统计量指的是 `softmax_max` 和 `softmax_sum`，不是对数形式的 LSE。
每个 query 至少需要一个有效索引；全无效行不在这份算子的支持范围。

## 四份主核为什么整理成两份

附件的 `dsa_kernel_fastgather.py`、`dsa_kernel_wide_fastgather.py` 和
`dsa_kernel_bn256.py` 使用同一个 GM 双槽数据流。它们的区别可以用
三个编译期参数表达：

| 参数 | 含义 | 保留的行为 |
|---|---|---|
| `BN` | 一个 KV tile 的行数 | 128 或 256；索引搬运量和 softmax 的归约树对应变化 |
| `PV_COLS` | 一次 PV 结果搬运的列数 | BM128 为 256；已选的小 BM 为 512 |
| `SHARE_SCORE` | score 和 PV 是否共用 UB 字节分配 | BM128 共用，且保留额外的 free-credit 等待；小 BM 分开 |

这些值是 `tl.constexpr`。选定配置后，未选中的代码分支不会留成设备上的
运行时判断。BN128 和 BN256 的 softmax 分别保留原来的操作顺序，尤其是
浮点归约括号，没有为了统一代码改成另一种求和次序。

`ragged.py` 没有强行并入这份模板，因为它的关键区别不是 tile 参数：
它由 Vector 自己打 NZ，QK 分成 512 维和 RoPE64 两次 MMAD，状态还采用
old/new 双份缓冲。它需要独立的 ready/free 时序。把这些差异藏进几十个
条件参数会更难读，也更难证明正确。

## 保持不变的内容

- 输入、RoPE、KV 和概率进入矩阵乘时仍是原来的 BF16 路径；QK/PV 累加、
  online softmax 状态和分片分子仍为 FP32。没有减少 rescale 或更改 exp 指令。
- 16 个规格的 BM、BN、split、28 个 Cube block，以及对应 56 个 Vector
  merge block 保持原配置。显式配置和 core 参数只用于有记录的复验。
- 所有 gather、GM 工作区写回、ND2NZ、softmax、PV 和 merge 都仍在调用内。
  没有缓存输入值、预先整理索引或跳过副输出。
- 逐条保留原来的 ready/free 标志和等待位置。特别是共享 score/PV UB 的
  配置，不能因为“score 已经算完”就删掉它的复用等待。
- `sanitize_overflow=False`、`enable_auto_bind_sub_block=False`、
  `enable_preload=False` 与附件一致；这些选项没有作为本次重构的新调优变量。

## Host 入口怎样减少开销，同时保留检查

最终 `api.py` 仍在每次调用检查所有输入的形状、dtype、设备和连续性，
以及调用者传入的 core 数。优化只减少同一次调用中重复读取 shape/device
和不必要的 Python generator、tuple、list 构造；分配顺序和 kernel 参数不变。

固定配置表由形状和不可变的 `Tuning` 组成，在模块导入时完整校验一次，
随后按形状查询。调用者显式传入的自定义 `Tuning` 仍逐次完整校验。
这里没有跳过验证的开关，也没有缓存张量、输入数值、索引整理结果或答案。

[Host 对照工具](tools/check_host_api.py) 使用真实 PyTorch meta 张量，将设备
类型断言从 NPU 替换为 meta，并用 stub 代替 kernel 发射，因此不运行 NPU。
它通过 86 项对照，检查 16 规格两种返回模式的分配和发射参数，以及非法输入
和配置的拒绝行为。完整 preflight 的中位数由 2.794 降到 1.446 微秒；
含 meta 分配与 stub 的 wrapper 由 25.925 降到 24.150 微秒。
这只是 host microbenchmark，不能当成真实 NPU eager 延迟的改进数值。
原始样本与最终 API 源码指纹见 [host-api-review.json](validation/host-api-review.json)。

## 为什么还有重复的 gather 和 epilogue

workspace 主核中，预取第一槽和稳态预取下一槽看起来相似，但它们位于
不同的流水位置。当前版本直接保留这两个块，以便逐条比较生成的 IR。
ragged 主核按最终 tile 数的奇偶输出不同状态缓冲，也保留原分支结构。

这里没有随意把 SIMD 块包装成普通 JIT helper。旧 wheel 的 AVE inliner
支持有限；函数抽取可能引入无法内联的边界，或者改变后端看到的调度和
别名关系。代码复用必须建立在编译器能消除抽象成本的基础上。将来若
编译器提供可靠的内联和 view/alias 跨函数语义，这些块才适合继续抽取。

## 兼容层的三个显式模式

在 Python import `dsa` **之前**设置：

```bash
export DSA_API_MODE=legacy   # 默认：与附件相同的 typed IR 桥接，供等价复验
export DSA_API_MODE=public   # 必须具备全部公开 API；缺失时立即报错
export DSA_API_MODE=auto     # 根据 API 可用性选择，报告必须记录实际选择
```

`legacy` 没有改动磁盘上的 site-packages，但会在当前 Python 进程中包装
CodeGenerator，并通过已有 dialect 构造 typed IR。因此“没有修改安装包”
不能等同于“只使用公开 TLA API”。这两件事必须分开描述。

`public` 模式不包装 CodeGenerator，也不解析或拼接 MLIR 文本。它使用
已有的 `cube.mmad`、`simd.vmuls`、`simd.vexpdif`，以及本次拟补齐的
`scalar_load`、`byte_view`、`load_gm_to_ub_strided`。主核数学和同步调度
与 legacy 模式相同；仍需独立做编译、精度和性能验证。

## 等价证据及其边界

运行：

```bash
python3 tools/check_refactor_ast.py
```

脚本不需要 NPU 或 Triton。它将原版和重构版中纯 constexpr 表达式及分支
专门化，归一化局部变量改名和兼容模块名称，再比较整个函数体的 Python
AST。它保留操作、控制流和语句先后次序，不按文本长度猜测等价。

已比较的五种 workspace 专门化覆盖全部 15 个 workspace 规格：

- 原 fastgather：BM128 / BN128 / PV256 / 共享 score。
- 原 wide_fastgather：BM16、32、64 / BN128 / PV512 / 独立 score。
- 原 bn256：BM16 / BN256 / PV512 / 独立 score。

另对 ragged 和 merge 的函数体作同样的比对，共 7 项全部一致。机器可读
结果与源文件 SHA-256 在 [ast-equivalence.json](validation/ast-equivalence.json)。
该文件本身不证明后端 IR、机器码或运行时间相同。

最终 host 入口更新后，又在 legacy 路径对全部 16 个规格检查了完整
output/max/sum：先检查 dtype 和 shape，再按 int16/int32 原始位比较，
全部逐位相同；对应主核和 merge 的 ELF 可执行 `.text` 指纹也全部相同。
记录见 [bitwise-final-all16.json](validation/bitwise-final-all16.json)。
这些结果证明本次输入上的输出等价和相应可执行代码段一致，不代表任意输入、
任意编译器版本或每轮计时都相同。设备时间、完整调用时间及共享环境波动，
仍应分别以 [最终性能报告](PERFORMANCE_FINAL.md) 的实测为准。
