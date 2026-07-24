# Mini-SGLang MetaX TODO

> Historical planning record from before the C500/C550 validation. Current
> completion and blockers are tracked by `README.md`, `REPORT.md`, and the
> Gate 0/Gate 1.x verdicts in this directory.

## 当前唯一主线

```text
理解通用框架链路
-> 真卡运行原生 CUDA 基线
-> 保留第一个有效失败
-> 只做一项最小修复
```

## 2026-07-23

| 优先级 | 任务 | 可见输出 | 完成标准 | 状态 |
| --- | --- | --- | --- | --- |
| P0 | 核对上游与 Ascend 分支 | commit 和硬件差异清单 | 能解释哪些 Ascend 改动不能复制到 MetaX | 已完成 |
| P0 | 梳理 Launch -> Scheduler -> Engine 初始化 | 源码注释和调用关系 | 能说清进程、Manager 和 Engine 初始化顺序 | 已完成第一轮 |
| P0 | 形成 MetaX 兼容边界 | 通用层/P0 验证层清单 | 不把 MACA/MCCL 实现当成项目任务 | 已完成 |
| P0 | 整理日报、Jira 和 Gate 0 | `daily_report_2026-07-23.md`、`jira_draft.md`、`gate0.md` | 已完成/候选/待真卡项分开 | 已完成 |
| P1 | 继续走读 `Engine.forward_batch()` | 一页输入/输出与调用链笔记 | 能说清 model -> sampler -> CPU token 回传 | 下一项 |
| P1 | 走读 Attention 和 KV Cache | backend 接口、KV 布局和 JIT 依赖表 | 每个 P0 风险能定位到源码文件 | 待开始 |

## 服务器到位前

1. 完成 `Engine.forward_batch()` -> Attention -> KV Cache -> Sampler 的通用链路笔记。
2. 整理一张 vendor API inventory：包名、版本、Mini-SGLang 需要的 symbol、验证命令和预期结果。
3. 准备一份基于干净 `ascend-port=85e3886` 、未加 MetaX fallback 的 CUDA 路径运行说明，并保留 `main=9a91cfa` 对照方法。
4. 保留当前 `metax-port` 候选 diff，但不在真卡证据前将全部 fallback 视为必要适配。
5. 不再新增大范围 MetaX 分支；如果没有真卡失败，就先增加笔记和检查项，不增加运行时代码。

## 服务器到位后

| 顺序 | 动作 | 必须保留的证据 | 停止/决策点 |
| ---: | --- | --- | --- |
| 1 | 记录 GPU、驱动、MACA、Python、PyTorch、`sgl_kernel`、FlashInfer、Triton 和容器版本 | 原始输出 | 版本明显不匹配时先解决环境 |
| 2 | 运行 `torch.cuda`/bf16/API/symbol 预检 | 每项 PASS/FAIL | 只记录问题，不立即全面 fallback |
| 3 | 运行干净 `85e3886` 的 CUDA 路径，必要时用 `9a91cfa` 对照 | 命令、commit、完整日志、首错 | 首错出现后停止，先建最小复现 |
| 4 | 将首错分类 | 环境/配置/依赖/API/算子判断 | 前一类未排除时不进入后一类 |
| 5 | 引入一项最小修复 | diff 和同 case A/B 日志 | 失败改变后再处理下一点 |
| 6 | 完成两次 greedy 请求 | 输入、输出、进程存活和显存状态 | 形成 Gate 0 verdict |

## 每日记录模板

```markdown
### YYYY-MM-DD

今日唯一目标：

已完成：
- 动作：
- 输出：
- 证据（commit/命令/日志/测试）：

当前结论：
- 已证实：
- 候选推断：
- 尚未验证：

问题/阻塞：
- 现象：
- 边界：

下一个最小动作：
- 动作：
- 预期输出：
- 停止条件：
```
