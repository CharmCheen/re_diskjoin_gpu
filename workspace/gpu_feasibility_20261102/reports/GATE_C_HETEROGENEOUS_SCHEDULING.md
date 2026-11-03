# GATE C — GPU×CPU 异构调度 / batching 后调度（SKIPPED_BY_GATE，2026-11-02）

**状态：SKIPPED_BY_GATE**（本 Gate 未执行任何调度实验，无伪造数据。）

## 跳过依据

1. **Runbook 明确限制**：`GPU_EXPERIMENT_RUNBOOK.md` Stage 3 规定"本轮不要开展 GPU 调度算法的创新实验，除非 Gate 2 确认的价值已被完整测量"。
2. **调度空间被 Amdahl 证据压缩（MEASURED，Gate A/B）**：
   - 原生 16T 管线中 IO/缓存路径占 91%（2.12s / 2.32s），验证仅 8% ⇒ CPU/GPU 流水重叠的可用并行窗口极小；
   - drop-in offload（mode1）warm E2E 仅 1.05x，说明即使调度完美，同管线天花板 ≈1.09x（Gate A 上界）；
   - batching 已在 b≥1024 饱和（188.6→23.5ms 后不再下降），batch 内调度无剩余收益空间。
3. **在本轮有价值测量的路径（VRAM 常驻）中，调度已被平凡方案占优**：mode2 直接整批提交即达 warm 29.6x，13,011 任务的 kernel 时间 23.5ms 中 launch 开销占比已 <10%（B1 数据），精细调度的理论空间 ≤ 该残差（INFERRED）。

## 若未来开展本 Gate 的前提条件

- 原生管线中 IO 与 GPU 重叠（异步 H2D + 双流），或 out-of-core 流式常驻方案落地；
- 负载中存在足够多的大任务（≥10⁵ cmp/任务，见 B2 break-even）使调度决策有实际影响；
- 冻结任务图 + 精确 GT 的消融协议（同 AGENTS.md Gate 4 要求）。

## Decision

SKIPPED_BY_GATE —— 本轮不执行。证据显示当前负载形态下调度研究空间受限，不构成独立研究贡献的直接依据。
