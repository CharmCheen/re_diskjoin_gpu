# GATE A — CPU Bottleneck Profiling & Task Trace（MEASURED，amax，2026-11-02）

## Hypothesis
原生 DiskJoin 的 Verification（距离计算）占端到端 search 时间的比例决定 GPU offload 的 Amdahl 上界；若比例小，单纯 GPU 化验证收益有限。

## 环境
见 `ENVIRONMENT.md`（amax，2×Xeon Gold 6526Y，OMP 线程扫描均 `numactl --cpunodebind=0 --membind=0`，编译器优化/AVX2 保留）。

## Workload（复用已验证冻结资产，未重新生成）
| 负载 | n | d | 桶数 | 任务数 | 距离计算 | 匹配 pair | 性质 |
|---|---|---|---|---|---|---|---|
| p1 | 4,000 | 16 | 120 | 1,484 | 1,833,576 | 190,149 | 合成（Gate 4A），mem_budget=0.1（缓存无 miss） |
| p3 | 4,000 | 16 | 120 | 1,685 | 3,274,616 | 59,151–59,183 | 合成（Gate 4A，ε 边界点负载） |
| bigann100k | 100,000 | 128 | 1,000 | 13,011 | 154,113,486 | 3,855,811 | 真实 BIGANN 前缀（Gate 5A 遗留），mem_budget=0.0048（缓存 28 槽，5,192 次 O_DIRECT miss） |

amax 复现性检查：p1 与 bigann100k 的 dist_comp / pair 数 / 采样计数与旧服务器 gate5a 记录**逐位一致**。

## A1 结果：执行阶段分解（bigann100k，search-only，`[gate-gpu]` 计时，5 reps 中位数）

| OMP 线程 | 候选生成 (HNSW) | IO+缓存 (O_DIRECT) | CPU 验证 | 总 Join |
|---|---|---|---|---|
| 1 | 0.212 s | **2.02 s (50%)** | ~1.76 s (44%) | 4.00 s |
| 16 | 0.014 s | **2.12 s (91%)** | ~0.19 s (8%) | 2.32 s |
| 32 | — | ~2.1 s | ~0.15 s (mode3) | — |

- **验证-only 时间（B0 replay，5 reps）**：1.315 s (1T) / **0.193 s (16T)** / 0.259 s (32T，超线程+NUMA 反而变慢)。
- **Amdahl 上界**：GPU 把验证降到 ~0（23–53 ms）后：
  - vs 原生 1T 管线：4.00 → ≥2.24 s ⇒ **≤1.79x**
  - vs 原生 16T 管线：2.32 → ≥2.13 s ⇒ **≤1.09x**
- **真正的主导瓶颈（16T 时）是串行 IO/缓存路径（2.1s，91%）**：5,192 次 O_DIRECT 读、每次 ~390μs（~165MB/s 有效带宽，小读延迟主导）。

p1/p3（T=1）：验证 12.4ms / 15.9ms，占 Join 33.5ms / 37.7ms 的 37%/42%（缓存无 miss 时 IO 占比小）。

## A3 交付：Task Trace（`traces/`）
每个负载：`tasks.txt`（Gorder 执行序任务图：slot, target, deg, neighbors）、`cpu_tasks.csv`（**task_id, slot, k, a, b, m, n, comparisons, matches, sampled** 逐任务真值计数）、`trace.txt`（缓存 A/P/H 决策序列）。trace 可映射回全局向量 id（assignment 顺序），支持固定候选集跨设备重放。p1/p3/bigann100k 均已验证与原生计数一致。

## 任务规模分布（bigann100k，GPU batching 依据）
comparisons/task：min 1,378 / p25 7,224 / **p50 12,096** / p75 12,669 / p90 17,898 / **max 92,568**。
⇒ 全部任务都是小任务（≤348×348 矩阵），单任务 GPU kernel 处于 launch 开销主导区间。

## Gate A Decision
- **PASS，含关键负结果**：在 16 线程优化 CPU 基线下，验证仅占 8%，单纯 GPU offload 的 E2E 天花板 ≈1.09x；1T 基线下 ≈1.79x。
- 唯一值得 GPU 投入的路径必须**同时**处理 IO 路径（VRAM 常驻）或扩展到 >100M 数据规模（out-of-core）。继续 Gate B 以量化验证-only 与两条 E2E 路径的真实收益。

## 命令与日志
- 原生 sweep：`logs/cpu_*_t*_r*.stdout/.time`，`results/raw/cpu_native_sweep.txt`
- 阶段分解：`logs/e2e_m0_t{1,16}.stderr`（`[gate-gpu] mode=0` 行）
- B0 replay：`results/raw/bigann_cpu_{summary,tasks}_t{T}_r{rep}.*`
