# DSA 所需的 TLA 内存接口补丁

社区 PR：[#2940 — Add scalar reads, byte views and strided DMA](https://github.com/triton-lang/triton-ascend/pull/2940)。

- 最终提交：`564a1e5b014adbc91b533c81dc699ec4425c1c99`；共两个提交。
- 产品实现与 121 项验证对应提交：`4a96387550889968145e01e2a66fab99f5cf9341`；后续仅改 README 拼写。
- 原始基线：`main-dev@0988a7ed24e68253cd3c08eac023ba3d3c9a496e`。
- 补丁：[0001-tla-dsa-memory-apis.patch](0001-tla-dsa-memory-apis.patch)。
- 分析与 PR 正文：[PR_DESCRIPTION.md](PR_DESCRIPTION.md)；完整中文说明见 [COMPILER_ANALYSIS.md](../COMPILER_ANALYSIS.md)。

补丁只新增 `scalar_load`、`byte_view`、`load_gm_to_ub_strided` 三类公开接口、
原生绑定、测试和说明。当前 main-dev 已有的 MMAD、subview、SIMD 接口没有重复实现。
不改变 DSA 算法、数学精度、索引权重或计时范围，也不包含完整 DSA 的性能提升声明。

## 应用并构建

在独立 checkout 中应用，避免影响已有开发目录。将 `/path/to/x` 换成解压后的
`x` 目录绝对路径：

```bash
git clone --no-checkout https://github.com/triton-lang/triton-ascend.git triton-ascend-dsa
cd triton-ascend-dsa
git checkout -b codex/dsa-memory-review 0988a7ed24e68253cd3c08eac023ba3d3c9a496e
git am /path/to/x/compiler/0001-tla-dsa-memory-apis.patch
git submodule update --init --recursive
```

`git am` 在不同提交环境可能产生不同 commit hash，但应用的是同一源码补丁。
两个提交标识记录在补丁文件头。设备验证时的 13 文件 SHA-256 保留在
[source.sha256](../validation/compiler/source.sha256)，最终交付对应
[final-source.sha256](../validation/compiler/final-source.sha256)。唯一变化为 README 中
`npuir` 改为 `NPU-IR`；产品和测试代码没有变化。也可直接获取已提交 PR：

```bash
git fetch origin pull/2940/head
git checkout --detach 564a1e5b014adbc91b533c81dc699ec4425c1c99
```

这次包含 C++ native builder，不能只复制 Python 文件到旧 wheel。请按对应提交的
[官方源码构建说明](https://github.com/triton-lang/triton-ascend/blob/4a96387550889968145e01e2a66fab99f5cf9341/docs/en/installation_guide.md)
准备 Linux/NPU、CANN、Python 和 LLVM 依赖，再在隔离环境中构建。
本次构建使用 Python 3.11，LLVM revision
`f6ded0be897e2878612dd903f7e8bb85448269e5`，NPU-IR gitlink
`4eb12142e16fc8c7ed33c3b568658b773ae0d1c9`。

本次用 `python setup.py build` 生成独立 `build/lib.linux-x86_64-cpython-311`，
通过 `PYTHONPATH` 选择该包，没有覆盖 lpc 的 site-packages。目录名依赖实际 Python
版本与平台；运行前检查 `triton.__file__` 和 `triton._C.libtriton.__file__`，确保
Python API 与新 native 库来自同一个构建。

## 验证接口与完整 DSA

在 NPU 环境先加载 CANN 环境，再使用已构建的新包运行仓库三阶段测试：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
TLA_TEST_STAGE=run python -m pytest -q -s \
  third_party/ascend/unittest/tla_ut/vector/scalar_load.py \
  third_party/ascend/unittest/tla_ut/vector/byte_view.py \
  third_party/ascend/unittest/tla_ut/dma/strided_load.py \
  third_party/ascend/unittest/tla_ut/memory_frontend_contract.py
```

结果为 **121 项通过**：79 个 host 合同及 14 个配置的 IR、compile、run。
设备阶段共 28 组输入全部 exact equality。动态 rows 禁止按数值专门化，包含
rows=1 时源 allocation 最后一行的边界测试。命令与原始日志见
[验证说明](../validation/compiler/COMMANDS.md)。

完整 DSA 需在导入前设置 `DSA_API_MODE=public`。四条代表路径以及四条额外边界
用例都通过已有数值门槛；这些是研究验证结果，不是官方 CANN 精度认证。
运行入口和限制见 [x/README.md](../README.md)，公平性结论见
[FAIRNESS_AUDIT.md](../FAIRNESS_AUDIT.md)。

## 社区 CI 状态

截至 `2026-11-03T01:28:39.974Z`，最终 head 的 GitHub 全仓 pre-commit 和 PR title 检查均通过；
Documentation 和 Integration Tests 仍在运行。旧 head 的文档门禁曾被 README 原有
拼写 `npuir` 阻断，已经用单独文档提交改成 `NPU-IR`，同款文档门禁五项以及全仓
pre-commit 均已本地复验通过。未将尚在运行的社区 CI 记为通过。

详细证据见 [CI_STATUS.md](../validation/compiler/CI_STATUS.md)、
[最终快照](../validation/compiler/PR_STATUS_FINAL.json) 和
[源码验证边界](../validation/compiler/source-provenance.json)。PR 已提交、尚未合并。
