# PR #2940 最终 CI 与验证边界

检查时间：`2026-11-03T01:28:39.974Z`。最终 head：`564a1e5b014adbc91b533c81dc699ec4425c1c99`，PR 为 OPEN，未合并。

- [最终 head 全仓 pre-commit](https://github.com/triton-lang/triton-ascend/actions/runs/41559838008)：SUCCESS。
- [最终 head PR title](https://github.com/triton-lang/triton-ascend/actions/runs/41559837997)：SUCCESS。
- [最终 head Documentation](https://github.com/triton-lang/triton-ascend/actions/runs/41559838129)：IN_PROGRESS。
- [最终 head Integration Tests](https://github.com/triton-lang/triton-ascend/actions/runs/41559838162)：IN_PROGRESS。

最新机器可读记录见 [PR_STATUS_FINAL.json](PR_STATUS_FINAL.json)。以下旧快照保留为历史，不能代替最终 head 状态。

## 已修复的文档门禁问题

实现提交的 Documentation workflow 41557291819 后续失败于测试 README 第 7 行的 `npuir` 拼写。该文字原已存在于基线；本 PR 追加 README 内容后触发了整文件检查。独立文档提交将它修正为正式名称 `NPU-IR`，未扩大拼写白名单。

修复后的同款 docs-ci-v2 五项检查全部通过，见 [文档门禁结果](docs-fix-gate-report.md) 和 [完整日志](docs-fix-gate.log)；再次执行全仓 pre-commit 全部适用 hooks 通过，见 [日志](docs-fix-precommit-all.log)。新一轮云端文档/集成检查尚未结束，不冒称通过。

设备 121 项验证的源提交为 `4a96387550889968145e01e2a66fab99f5cf9341`。最终 head 仅比它多一行 README 拼写修改。原 [source.sha256](source.sha256) 保留设备测试时的清单，新 [final-source.sha256](final-source.sha256) 记录交付清单，[source-provenance.json](source-provenance.json) 明确两者对应关系。产品和测试源码相同。

---

## 以下为修复前历史快照

# PR #2940 CI 快照

检查时间：`2026-11-03T01:11:17.130396+00:00`。

PR：[Add scalar reads, byte views and strided DMA](https://github.com/triton-lang/triton-ascend/pull/2940)。
提交：`4a96387550889968145e01e2a66fab99f5cf9341`；与交付提交一致：`True`。

## 已确认通过

- PR title format 检查通过。
- GitHub pre-commit 对全仓运行 `python3 -m pre_commit run --all-files --verbose`，全部适用检查通过，包含 ruff、yapf、clang-format、mypy。
- CI 检查的是 GitHub 合成 merge `db1dfaa`：把本 PR head 合入 `0ee9f1a3d5e91807aba88d84f124628eff52acd5`。它与本地按原始基线进行的改动文件检查是两份不同证据。

[完整 pre-commit CI](https://github.com/triton-lang/triton-ascend/actions/runs/41557291878)，命令和 hook 结果摘录见 `ci-precommit-excerpt.log`。

## 当前其余检查

- [label](https://github.com/triton-lang/triton-ascend/actions/runs/41557287442/job/125659151321)：SUCCESS。
- [RTD Preview](https://github.com/triton-lang/triton-ascend/actions/runs/41557287224/job/125659150800)：SUCCESS。
- [runner-preparation / prepare](https://github.com/triton-lang/triton-ascend/actions/runs/41557292068/job/125659164213)：SUCCESS。
- [code-review](https://github.com/triton-lang/triton-ascend/actions/runs/41557287297/job/125659151154)：SUCCESS。
- [Check PR title format](https://github.com/triton-lang/triton-ascend/actions/runs/41557291725/job/125659162568)：SUCCESS。
- [pre-commit (code formatting)](https://github.com/triton-lang/triton-ascend/actions/runs/41557291878/job/125659163853)：SUCCESS。
- [restricted files labeler](https://github.com/triton-lang/triton-ascend/actions/runs/41557287328/job/125659151200)：SUCCESS。
- [docs/readthedocs](https://triton-ascend--2940.org.readthedocs.build/)：SUCCESS。
- [Docs gate](https://github.com/triton-lang/triton-ascend/actions/runs/41557291819/job/125659163191)：IN_PROGRESS，尚不能记为通过。
- [integration-tests-ascend / integration-tests-ascend (linux-aarch64-a3-800i-4, py3.10)](https://github.com/triton-lang/triton-ascend/actions/runs/41557292068/job/125659207441)：IN_PROGRESS，尚不能记为通过。
- [integration-tests-ascend / integration-tests-ascend (linux-aarch64-a3-800i-4, py3.11)](https://github.com/triton-lang/triton-ascend/actions/runs/41557292068/job/125659207397)：IN_PROGRESS，尚不能记为通过。
- [integration-tests-ascend / integration-tests-ascend (linux-amd64-a5-2, py3.12)](https://github.com/triton-lang/triton-ascend/actions/runs/41557292068/job/125659207481)：IN_PROGRESS，尚不能记为通过。

本次未发现失败 CI；尚未结束的集成测试不包含在已通过结论中。没有等待长时间集成测试完成，也没有发送 PR 评论或 review。后续状态以 GitHub 为准。

## 代码复核边界

已核对提交只包含 13 个预期文件、原生绑定有公开 builder 挂载、静态容量与动态子视图保留实际行数和别名、signed i32/i64 合同和零 offset 回归已包含；未发现新的阻塞问题。检查不替代尚在运行的集成测试，也不意味着已获得社区批准。
