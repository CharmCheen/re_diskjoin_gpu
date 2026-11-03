# HANDOFF_GPU — DiskJoin GPU Feasibility 轮次交接（2026-11-02，amax）

本轮全部产物：`workspace/gpu_feasibility_20261102/`（107MB）。结论详见 `reports/FINAL_RESEARCH_ASSESSMENT.md`。

## 最终结论（一页版）

- **GPU 验证机制有效且正确**：验证-only 8.2x（vs 16T CPU）/56x（vs 1T），四实现输出多重集逐位一致（bigann100k 3,855,811 对；p1 190,149 对）；p3 仅 ε²=0.2 的 1–2 ulp 边界 tie。
- **但原生 E2E drop-in 仅 1.05x（16T，warm）**，未达预注册 1.20x 门槛；根因是 IO/O_DIRECT 路径占 91%（Gate A）。
- **VRAM 常驻 + 跳 IO = warm 29.6x**（mode2 vs native），其中 GPU 验证额外贡献 2.0x（vs mode3 CPU DRAM 常驻）；但 100M×128d ≈ 51.2GB 装不进单卡 32.6GB，该路径不能外推到论文规模（INFERRED）。
- **判定：MIXED / ENGINEERING-leaning，建议停止或重新立项**（新方向 = out-of-core 流式常驻 + IO-GPU overlap，属新的研究问题，NOT_TESTED）。

## 已冻结/已交付

| 资产 | 路径 | 说明 |
|---|---|---|
| 环境 | `ENVIRONMENT.md` | GPU0 可用；GPU1/2 驱动故障；CUDA init 3.3–4.6s/进程 |
| 来源审计 | `SOURCE_AUDIT.md` | 本地代码语义（同桶 id2<id1、严格 <ε²、采样计数、无 per-pair pruning） |
| CPU 剖析 + 任务图 | `traces/{p1,p3,bigann100k}/`、`reports/GATE_A_CPU_PROFILE.md` | tasks.txt / cpu_tasks.csv / trace.txt，可跨设备重放 |
| 正确性 | `reports/correctness_report.md` | 四实现比对方法与结果 |
| Gate B | `reports/GATE_B_GPU_VERIFICATION.md` | B0/B1/B2 + E2E 四模式 + break-even |
| Gate C / D | `reports/GATE_C_*.md` / `GATE_D_*.md` | SKIPPED_BY_GATE / BLOCKED_ENV |
| 汇总数据 | `results/summary/*.csv` | e2e_summary.csv（核心表）、gpu_replay_summary.csv |
| 原始数据 | `results/raw/` | ≥5 reps/配置 + `/usr/bin/time` JSON |
| 实现 | `src/` | gpu_verify.cu、replay_gpu.cu、replay_cpu.cpp、DiskJoin_gpu（GATE_GPU=0/1/2/3 门控） |
| 命令与清单 | `command_manifest.tsv`、`MANIFEST_SHA256.txt` | 见本轮 README |

## 下一步最小行动（若用户决定继续）

1. **如接受"停止"判定**：归档本目录即可；无需进一步实验。
2. **如需突破 1.20x 门槛**（新立项）：实现 IO 与 GPU 验证重叠（双流 + 异步 H2D + 预读线程），先在 bigann100k 冻结任务图上 replay 验证机制，再集成 native。预估：1–2 天编码 + 0.5 天测量，磁盘增量 <100MB。
3. **如评估 out-of-core 方向**（需重新授权）：需要 ≥100GB 磁盘余量生成 10M/100M 前缀数据 —— 当前根分区 99% 满（11G 可用），**硬阻塞**。
4. **管理员协作项**（无 sudo，无法自助）：修复 GPU1/2 `cudaSetDevice` OOM（`CUDA_VISIBLE_DEVICES=1 /tmp/gvutest2` 可复现）；评估开启 Persistence Mode 以消除 3.3–4.6s/进程 CUDA init。

## 环境与复现要点

- 编译：`/usr/local/cuda/bin/nvcc -arch=sm_120 -ccbin g++`（nvcc 不在 PATH）；g++ 11.4 + `-fopenmp -O3 -march=native`。
- 运行：必须 `CUDA_VISIBLE_DEVICES=0`；GPU1/2 不可用。
- 计时口径：**必须用 warm（扣 gpu_init_s）**；cold 含 3.3–4.6s CUDA init，不可用于比较。
- 受控协议：冻结任务图 + OMP=1 候选生成 + CPU 侧 `numactl --cpunodebind=0 --membind=0`；≥5 reps 中位数。
- 正确性协议：输出 pair 多重集比对 + numpy fp64 oracle；p3 边界 tie 已定性，不掩盖。

## 未完成 / 阻塞清单

- [BLOCKED] 多卡（GPU1/2 驱动 OOM）。
- [SKIPPED] 异构调度创新（Runbook 限制 + 价值空间受限）。
- [NOT_TESTED] 10M/100M 规模（磁盘 11G 可用，硬阻塞）。
- [NOT_TESTED] cuVS/库基线对比（环境无 torch/cupy/cuvs；安装被 Runbook 禁止）。
- [环境] Persistence Mode Off；cudahold 常驻探针（占 507MiB）已于收尾阶段终止。
