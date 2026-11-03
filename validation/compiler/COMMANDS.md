# 编译器补丁验证记录

基线：`triton-ascend main-dev@0988a7ed24e68253cd3c08eac023ba3d3c9a496e`。
本目录 `source.sha256` 对应 13 个修改文件，`changed-files.txt` 是文件列表。

已完成：真实 native extension 构建、79 个 host 合同检查、14 个真实 IR 配置、
完整 pre-commit，以及设备完整三阶段 **121 passed**（79 host + 14 × 3）。
14 个 run 配置各使用两个 seed，28 组输入全部 exact equality。

设备三阶段使用 `/home/l00898962/experiments/dsa-review-20261102/venv/bin/python`；
host/格式检查使用下列命令中的 `tla-mmad-subview-20261102/venv`。二者继承
用户指定 lpc 的 Python 3.11 和 PyTorch，没有覆盖 lpc 的 site-packages。

## 离线 IR

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/home/l00898962/experiments/tla-mmad-subview-20261102/build-src/build/lib.linux-x86_64-cpython-311
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /home/l00898962/experiments/dsa-review-20261102/precommit-dsa-api
/home/l00898962/experiments/tla-mmad-subview-20261102/venv/bin/python -m pytest \
  --noconftest -q -k test_ir \
  third_party/ascend/unittest/tla_ut/vector/scalar_load.py \
  third_party/ascend/unittest/tla_ut/vector/byte_view.py \
  third_party/ascend/unittest/tla_ut/dma/strided_load.py
```

结果 `14 passed, 28 deselected`。后面 28 项是设备 compile/run 阶段，本命令刻意不执行。

## Host 合同与格式检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/l00898962/experiments/tla-mmad-subview-20261102/venv/bin/python \
  -m pytest --noconftest -q third_party/ascend/unittest/tla_ut/memory_frontend_contract.py
PRE_COMMIT_HOME=/home/l00898962/experiments/tla-mmad-subview-20261102/precommit-cache \
  /home/l00898962/experiments/tla-mmad-subview-20261102/venv/bin/pre-commit \
  run --files $(cat ../changed-files.txt)
git diff --check
sha256sum -c ../source.sha256
```

结果：79 项 host 检查通过；pre-commit 全部适用项通过；diff 检查无错误，
13 个源码哈希全部匹配。`ruff` 根据仓库原有 include/filter 跳过，不代表单独跑过 ruff。

## 设备三阶段与最终源码

`native-device-ut-final.log` 记录最终 **121 passed, 7 warnings in 22.42s**，包含：

- scalar load：direct、linear、strided、matrix 四个配置；
- byte view：byte offset 为 0、128 两个配置；
- strided DMA：FP16/BF16 各 static、dynamic、wide、zero_base 四个配置。

每个配置的 run 阶段用 seed 7、19，28 组输入逐位检查数据及保护区。
DMA rows 使用 `do_not_specialize`，运行时检查编译签名仍是 i32/i64，避免
rows=1 自动成为 constexpr。零 offset 用例包含 source allocation 最后一行，
仅该行有效；这覆盖了完整 DSA 集成时发现的 pointer-entry 静态 shape 要求。

启动设备检查前取消 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，按仓库的
`tla_ut/README.md` 运行 `TLA_TEST_STAGE=run` 三阶段。最终远端 build-src 的
13 个源文件已逐项通过 `sha256sum -c source.sha256`，对应当前交付版本。
旧的 `native-device-ut.log` 是早期 115 项测试，不能替代最终日志。

## 完整公开 API 路径

`public-native-offline.log` 是四个完整主核与所需 merge 的正式 native lowering
结果：`decode_b1`、`decode_b8`、`decode_tp8`、`prefill_tp8`。每条记录的
`prototype_ir_rewrite` 都为 false，且 replacements 为 0；没有通过修改待测 IR
绕过转换问题。这四条路径随后也完成设备数值验证，结果放在主验证目录中。

设备 public smoke 检查禁止 DSA 自己的 CodeGenerator 补丁和旧 helper 模块。
官方 Ascend backend 的正常延迟初始化会给 CodeGenerator 添加目标属性；脚本
核对来源为 stock backend 后允许这一正常行为。

这些结果证明接口与所测 DSA 路径的数值正确性，不是完整 DSA 的性能提升结论。
