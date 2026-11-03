# GATE B — GPU Verification 实现与 CPU/GPU 对照（MEASURED，amax，2026-11-02）

## Hypothesis

- H-B1：GPU 可将 Verification（距离计算）阶段加速 ≥8x（相对 16T CPU）且逐位正确。
- H-B2：验证-only 加速在原生 E2E 管线中的收益受 Gate A Amdahl 上界约束（16T ≈1.09x，1T ≈1.79x）。
- H-B3：E2E 收益的主要杠杆是把数据与候选常驻 VRAM、跳过 IO 路径，而非仅替换验证 kernel。
- 预注册继续门槛：**原生 E2E（同管线、同协议）≥1.20x** 才值得作为研究项目继续。

## 环境

见 `ENVIRONMENT.md`。GPU0（RTX 5090, sm_120），`CUDA_VISIBLE_DEVICES=0`；nvcc 12.8 `-arch=sm_120 -ccbin g++`；fp32 fma、严格 `< eps²`、禁 TF32/FP16；采样计数沿用 `DiskJoin.h` 约定（id%100000）。

## 方法

- **正确性协议**：见 `reports/correctness_report.md` —— p1 / bigann100k 输出 pair 多重集在 native / CPU-replay / GPU / numpy oracle 四实现间逐位一致（190,149 / 3,855,811 对）；p3 仅 ≤32 对 ε²=0.2 的 1–2 ulp 边界 tie（与 Gate 4B0 已记录现象一致）。
- **受控比较**：OMP=1 候选生成 + 冻结任务图（`traces/*/tasks.txt`），CPU 侧 `numactl --cpunodebind=0 --membind=0`；正式数据 ≥5 reps 取中位数。
- **E2E 四模式**（`src/DiskJoin_gpu/`，环境门控 `GATE_GPU=0/1/2/3`，同一二进制、同 build）：
  - mode0 = 原生管线（无 GPU，对照）；
  - mode1 = GPU 验证 + 原生 O_DIRECT IO（drop-in offload）；
  - mode2 = 数据/候选 VRAM 常驻，跳过 IO 路径；
  - mode3 = 数据 CPU DRAM 常驻，跳过 IO（区分"跳 IO"与"用 GPU"的贡献）。
  - 冷（含 CUDA init）与 warm（扣 `gpu_init_s`）双口径呈现。

## B0 — CPU 验证基线（bigann100k，B0 replay，5 reps）

| 配置 | 验证时间 | 吞吐 |
|---|---|---|
| 1T | 1.315 s | 1.17e8 cmp/s |
| 16T | **0.193 s** | 8.0e8 cmp/s |
| 32T | 0.259 s（超线程+NUMA 反而变慢） | — |

## B1 — GPU 验证-only（bigann100k，13,011 任务，154,113,486 cmp）

| 变体 | 时间 | vs CPU 16T |
|---|---|---|
| naive per-task H2D（pageable, b512） | 0.756 s | **0.26x（慢 3.8x）** |
| naive per-task H2D（pinned, b512） | 0.731 s | 0.26x |
| resident + batching b=1 | 0.1886 s | 1.02x |
| resident b=8 | 0.0538 s | 3.6x |
| resident b=32 | 0.0319 s | 6.1x |
| resident b=128 | 0.0254 s | 7.6x |
| resident b=1024 | **0.0235 s** | **8.2x** |
| resident b=13012（全量） | 0.0235 s | 8.2x（已饱和） |

- **batching 增益 8.0x**（188.6→23.5ms）；b≥1024 后完全饱和（任务图天然按 slot 分组，13,011 任务仅 ~1,012 个 slot 组）。
- kernel-only 23.4ms ⇒ **6.56e9 cmp/s**（d=128），与 microbench 平台 6.86e9 一致；vs CPU 16T 8.2x、vs CPU 1T 56x。

## B2 — microbenchmark（正式 5 reps，`results/raw/micro_d{16,128}_formal.csv`）

| 维度 | 平台吞吐（大 batch） | 单任务固定开销（launch+H2D/D2H） |
|---|---|---|
| d=16 | 7.2e10 cmp/s | ~25–60 μs |
| d=128 | 6.9e9 cmp/s | ~25–60 μs |

- batching 收益随任务变小而放大：d=128, m=n=128：27.8→1.51 μs/任务（batch1→128，**18x**）；d=16, m=n=128：18x。
- **Break-even 推断**：bigann100k p50 任务仅 12K cmp（GPU kernel ~2μs，低于固定开销）⇒ 不 batching 必然被 launch 开销主导，与 B1 b=1 1.02x 观测一致。CPU 16T 吞吐 0.8e9 cmp/s，GPU 需要任务 ≥ ~10⁵ cmp 才在单任务层面胜出。

## E2E 四模式对照（bigann100k，5 reps 中位数，`results/summary/e2e_summary.csv`）

| 模式 | 1T cold / warm | 16T cold / warm | vs native 16T (warm) | matches 校验 |
|---|---|---|---|---|
| 0 native | 4.00 / 4.00 s | 2.32 / 2.32 s | 1.00x | native 自身 |
| 1 GPU验证+原生IO | 5.73 / 2.42 s | 5.53 / **2.22 s** | **1.05x** | 3,855,811 ✔ |
| 2 VRAM常驻跳IO | 3.61 / 0.276 s | 3.41 / **0.0785 s** | **29.6x** | 3,855,811 ✔ |
| 3 CPU DRAM常驻 | 1.96 / 1.93 s | 0.188 / 0.158 s | 14.7x | mode3 不重报（真值计数由任务图旁路校验） |

- **CUDA init 3.3–4.6s/进程**（Persistence Off，驱动级）使所有 cold 口径不可用；warm 口径为唯一有效比较，已在 HANDOFF 与报告中双口径标注。
- mode1 vs mode0：**16T 仅 1.05x（未达预注册 1.20x 门槛）；1T 1.65x** —— 与 Gate A Amdahl 上界 1.09x/1.79x 吻合（验证 23.5ms 未完全归零）。
- mode2 vs mode3 = **2.0x**：即使同为跳 IO，GPU 验证仍比 CPU 16T 验证快一倍（0.0785 vs 0.158s），验证加速在跳 IO 管线内真实兑现。
- p1/p3（T=1，缓存无 miss 的小负载）：native 32.9 / 37.7ms → mode2 warm **6.7 / 6.9ms（4.9x / 5.5x）**，matches 190,149 / 59,154 与正确性报告一致。

## Limitations

1. mode2 的 29.6x **依赖全量 VRAM 常驻**：本轮 100k×128d 仅 51.6MB；论文目标 100M×128d ≈ 51.2GB > 单卡 32.6GB，**该路径不能直接外推到 out-of-core 规模**（INFERRED）。
2. mode1 的 IO 路径未做任何 GPU 侧优化（未测试 GPUDirect Storage、批量预读等）；1.05x 是"纯 drop-in 验证 offload"的下界结论，不是"GPU 不能加速 E2E"的结论。
3. CUDA init 3.3–4.6s 为本机驱动配置问题（Persistence Off 且无 sudo），常驻守护进程仅降至 3.0–3.3s；其他机器上 cold 开销可能不同。
4. microbench `eff_ratio` 列为占位诊断（全 0，未用于结论）；kernel 吞吐以 B1 kernel-only 与平台值交叉验证。
5. 全部结论限于单卡 GPU0；GPU1/2 不可用（见 GATE_D）。

## Gate B Decision

- **PASS（机制级）**：H-B1 成立 —— 验证-only GPU 加速 **8.2x（16T）/ 56x（1T）**，四实现逐位正确。
- **H-B2 成立**：drop-in E2E 1.05x（16T），被 IO 瓶颈压死，与 Amdahl 预测一致；**未达预注册 1.20x 原生 E2E 门槛**。
- **H-B3 成立（限定规模）**：VRAM 常驻 + GPU 验证 = warm **29.6x**（bigann100k, 16T），其中跳 IO 贡献 ~14.7x、GPU 验证再贡献 2.0x；但常驻前提在 100M 规模不成立。
- **初步研究判定（MIXED / ENGINEERING-leaning）**：GPU 验证收益真实但被 IO 主导的原生管线掩盖；单卡上要兑现收益必须重构 IO/调度路径，且 out-of-core 规模需要流式/分片常驻方案 —— 这属于系统工程而非算法创新。最终判定见 `reports/FINAL_RESEARCH_ASSESSMENT.md`。

## 命令与日志

- GPU replay batch 扫描：`logs/gpu_replay_*.log`，汇总 `results/summary/gpu_replay_summary.csv`
- naive H2D：`results/raw/bigann_gpu_{pageable,pinned}.csv`
- microbench：`results/raw/micro_d{16,128}_formal.csv`，`logs/micro_*.log`
- E2E 正式：`results/raw/e2e_bigann_formal.csv`、`e2e_p1p3_formal_raw.txt`，`logs/e2e_*.log`
- 实现：`src/gpu_verify.cu`、`src/replay_gpu.cu`、`src/DiskJoin_gpu/lib/DiskJoin.h`（`GATE_GPU` 门控）
