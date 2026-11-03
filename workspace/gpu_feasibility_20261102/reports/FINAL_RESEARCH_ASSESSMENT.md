# FINAL RESEARCH ASSESSMENT — DiskJoin GPU 加速可行性（amax，2026-11-02）

**判定：MIXED，偏向 ENGINEERING_ONLY —— 不建议以当前形态作为 SIGMOD/VLDB 研究项目继续；建议停止或重新定位。**

证据等级标注：**MEASURED**（本轮真实执行）／**INFERRED**（分析推断）／**NOT_TESTED**。

## 研究问题逐条回答

> 问题按本轮任务书（RQ1–RQ6 及后续 Gate C/D 追问）转述；问题原文以用户 prompt 为准。

**Q1 GPU 在本机是否真实可用且结果正确？**
是（MEASURED）。GPU0（RTX 5090, sm_120）CUDA smoke 通过；GPU 验证输出与 native/CPU-replay/numpy 四实现**逐位一致**（p1 190,149 对、bigann100k 3,855,811 对多重集比对）；p3 的 ≤32 对差异全部为 ε²=0.2 的 1–2 ulp 边界 tie，与 Gate 4B0 历史记录一致（`reports/correctness_report.md`）。

**Q2 验证 kernel 本身相对 CPU 的真实加速是多少？**
验证-only：GPU 23.5ms vs CPU 16T 193ms = **8.2x**，vs CPU 1T 1315ms = **56x**（MEASURED，冻结任务图、同候选集、≥5 reps 中位数）。microbench 平台吞吐 d=16 7.2e10、d=128 6.9e9 cmp/s。

**Q3 drop-in GPU offload 在原生 E2E 管线中的收益？**
**1.05x（16T，warm 2.22s vs native 2.32s）；1.65x（1T）**（MEASURED，5 reps）。**未达预注册 1.20x 门槛（16T 口径）**。与 Gate A Amdahl 上界 1.09x/1.79x 吻合：原生 16T 管线中 IO/缓存占 91%，验证仅 8%，替换验证不可能有大收益。

**Q4 真正的瓶颈是什么？**
MEASURED：bigann100k 16T 下 IO/O_DIRECT 路径 2.12s（91%）串行主导（5,192 次小读，~390μs/次）；1T 下 IO 50% + 验证 44%。另有 **CUDA 上下文创建 3.3–4.6s/进程**（Persistence Off，驱动级，常驻守护进程仅降至 3.0–3.3s），使所有短进程 cold E2E 失效。

**Q5 VRAM 常驻路径的收益与可扩展性？**
收益 MEASURED：mode2（VRAM 常驻+跳 IO）warm **29.6x**（0.0785s vs native 2.32s），其中"跳 IO"贡献 ~14.7x（mode3 CPU DRAM 常驻 0.158s），GPU 验证再贡献 2.0x。可扩展性 **不成立**（INFERRED）：该收益依赖全量常驻，100k×128d 仅 51.6MB，而论文目标 100M×128d ≈ 51.2GB > 单卡 32.6GB ⇒ 常驻路径在目标规模不可用，需新的流式/分片方案（NOT_TESTED）。

**Q6 batching / 任务粒度结论？**
MEASURED：batching 增益 8.0x（b=1 188.6ms → b=1024 23.5ms），b≥1024 饱和；bigann100k 全部任务 ≤92K cmp（p50 12K），单任务 GPU 处理处于 launch 开销主导区间（b=1 时加速比仅 1.02x）。break-even：GPU 需任务 ≥~10⁵ cmp 才在单任务层胜出 16T CPU（INFERRED，由 microbench 固定开销 ~25–60μs 推得）。

**Q7 多卡扩展？**
**BLOCKED_ENV**（MEASURED）：GPU1/2 `cudaSetDevice` 返回 out of memory（32GB 空闲下，驱动级），无 sudo 无法修复；且 Runbook 本轮禁止三卡 benchmark。三卡间无 NVLink（SYS 互联），即使修复也需按 NUMA 分域设计（`reports/GATE_D_MULTI_GPU.md`）。

**Q8 是否值得作为 SIGMOD/VLDB 项目继续？**
**不建议以"GPU 加速 DiskJoin"的当前形态继续**（判定依据如下）。理由：
1. 核心机制收益（验证 8.2x）真实，但在原生管线中被 IO 瓶颈压到 1.05x —— 这本身是对论文的有价值负结果，但不足以支撑一篇系统论文；
2. 唯一大收益路径（VRAM 常驻 29.6x）在 out-of-core 目标规模（100M）内存不可行，**除非**提出新的流式常驻/IO-GPU overlap 调度方案 —— 该方案属本轮 NOT_TESTED，且其创新点是"IO 调度"而非"GPU 验证"；
3. 异构调度（Gate C）空间被 Amdahl 证据与 batching 饱和压缩，独立成文的贡献存疑；
4. 竞品风险：RAPIDS cuVS 等成熟 GPU 距离/join 原语存在（本轮环境未装，未对比 —— NOT_TESTED），纯 kernel 层面难以构成壁垒。

**若继续研究的唯一可信方向**（需重新立项，非本轮延续）：面向 out-of-core 的 GPU 常驻流式执行（数据分片预取 + IO 与 GPU 重叠 + 常驻进程消除 3.3s init），并以 100M 级真实负载和完整 Recall 协议验证 —— 这需要新的授权、磁盘资源与管理员修复 GPU1/2。

## 本轮达成 vs 未达成

| 项 | 状态 |
|---|---|
| 环境审计 + sm_120 smoke | PASS（MEASURED） |
| Gate A CPU 剖析 + Task Trace | PASS（MEASURED，与旧服务器逐位复现） |
| Gate B 正确性（四实现多重集比对） | PASS（MEASURED） |
| Gate B 验证-only 性能 + batching + microbench | PASS（MEASURED，≥5 reps 中位数） |
| Gate B E2E 四模式（同管线同协议） | PASS（MEASURED）；**预注册 1.20x 门槛未达标（16T）** |
| Gate C 异构调度 | SKIPPED_BY_GATE |
| Gate D 多卡 | BLOCKED_ENV |
| 100M 规模外推 | NOT_TESTED（磁盘与内存约束） |
| cuVS/库基线对比 | NOT_TESTED（环境无 torch/cupy/cuvs） |

## 给用户的一页总结

GPU 当前**有效但被掩盖**：RTX 5090 上验证阶段真实加速 8.2x（16T CPU）且逐位正确；但在原生 DiskJoin 管线中端到端仅 1.05x（16T），因为 91% 的时间在串行 O_DIRECT IO；把数据常驻 VRAM 跳过 IO 可达 29.6x，可这在 100M 目标规模下单卡装不下。CUDA 进程 3.3–4.6s 的初始化开销和多卡驱动故障是额外环境阻塞。结论：GPU 验证本身不足以构成研究贡献；除非围绕 out-of-core 流式常驻/IO 重叠重新立项，否则建议停止。

## 证据索引

- 环境：`ENVIRONMENT.md`；来源审计：`SOURCE_AUDIT.md`
- Gate A：`reports/GATE_A_CPU_PROFILE.md`（剖析、任务分布、Amdahl）
- 正确性：`reports/correctness_report.md`
- Gate B：`reports/GATE_B_GPU_VERIFICATION.md`（B0/B1/B2、E2E 四模式、break-even）
- 汇总数据：`results/summary/e2e_summary.csv`、`results/summary/gpu_replay_summary.csv`
- 原始数据：`results/raw/`（≥5 reps/配置）；日志：`logs/`
- 历史：`workspace/HANDOFF.md`（Gate 0–4B1）
