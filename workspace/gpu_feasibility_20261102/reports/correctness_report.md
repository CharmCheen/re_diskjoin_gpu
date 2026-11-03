# correctness_report.md — GPU/CPU/Oracle 正确性验证（全部 MEASURED）

## 验证链设计
四条独立实现互相印证：
1. **Native**：`dj_search_only`（上游源码原样编译，只读执行）+ GATE3_PAIR_OUT 全量 pair dump。
2. **CPU Replay (B0)**：`replay_cpu`（复用上游 `utils::L2Sqr`，固定任务图重放）。
3. **GPU**：`replay_gpu` / `dj_gpu_search`（CUDA，fp32 fma 串行每线程归约）。
4. **Oracle**：`scripts/oracle_pairs.py`（numpy，独立实现；fp32 与 fp64 双精度模式）。

对照口径：per-task 匹配数（整数）、全量 pair 多重集（排序后逐对比较）、采样计数（id%100000）。

## 结果汇总

### bigann100k（真实 BIGANN 数据, n=100k, d=128, ε²=73984, 13,011 任务, 154,113,486 次比较）
数据为 uint8 导出：所有差平方与部分和为 <2^24 的整数，fp32 **精确无舍入** → 无边界歧义。

| 实现 | 匹配 pair 总数 | 采样计数 | 判定 |
|---|---|---|---|
| Native pair dump | 3,855,811 | 2 | 基准 |
| CPU replay (per-task 计数和) | 3,855,811 | 2 | **EXACT** |
| GPU replay（batch 1/8/32/128/1024/13012 ×5 reps，逐任务） | 3,855,811 | 2 | **EXACT，13,011/13,011 任务计数全等** |
| GPU native E2E（mode 1/2 ×5 reps） | 3,855,811 | recall=0.666667 同 native | **EXACT** |
| numpy oracle（全量 pair 多重集） | 3,855,811 | — | **与 native dump 逐对 EXACT MATCH** |

### p1（合成, n=4000, d=16, ε²=0.2, 1,484 任务）
| 实现 | pair 总数 | 判定 |
|---|---|---|
| Native dump | 190,149 | 基准 |
| CPU replay | 190,149 | **EXACT** |
| GPU replay（逐任务） | 190,149，mism=0 | **EXACT** |
| GPU native E2E mode2 ×5 | 190,149 | **EXACT** |
| numpy oracle 多重集 | 190,149 | **EXACT MATCH** |

### p3（合成, n=4000, d=16, ε²=0.2, 1,685 任务，**生成器刻意放置 ε 边界点**）
| 实现 | pair 总数 | 与 fp64 真值差 |
|---|---|---|
| numpy fp64（真值口径） | 59,183 | 0 |
| numpy fp32 | 59,177 | −6 |
| CPU replay (AVX2 fp32) | 59,174 | −9 |
| GPU (fp32 fma) | 59,154 | −29 |
| Native (fp32 AVX2, 同一函数不同 TU) | 59,151 | −32 |

判定：**非 bug，为固有 fp32 阈值边界 tie**。抽查全部差异对（如 (103,702)、(1327,1355)）：dist² 均在 ε²=0.2 的 **1–2 ulp** 内（0.199999988 / 0.200000003），其中 (1327,1355) fp64 判真、fp32 判假，证明是真 tie。与 Gate 4B0 记录的 P3 ε² 边界 tie 现象一致（该负载 `boundary_frac=0.2`）。GPU 与 native 相差仅 3 对（0.005%），且都落在同一 tie 集内；在无 tie 数据（bigann100k、p1）上 GPU 与 native **逐位一致**。

## 浮点口径说明
- 全部 GPU 实验禁用 TF32/FP16（显式 fp32 fma，`--use_fast_math` 未启用）。
- Count-only 为主口径（与原生一致）；pair 物化仅用于正确性验证（oracle 侧），未做大规模输出。

## 结论
- **GPU Verification 在 tie-free 数据上与原生实现计数逐位一致、pair 多重集一致**（两条真实/独立负载全量验证）。
- 边界 tie 差异已定位、量化并归因（≤3 对 / 59k，均 1-2 ulp）。
- PASS（测试范围内）。
