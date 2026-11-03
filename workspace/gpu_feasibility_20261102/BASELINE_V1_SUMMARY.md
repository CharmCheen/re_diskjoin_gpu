# BASELINE_V1 — DiskJoin GPU Baseline（冻结摘要）

**冻结日期**：2026-11-02
**归档 ID**：GPU_BASELINE_V1
**目录**：`workspace/gpu_feasibility_20261102/`（107 MB，全部只读）
**完整性**：`MANIFEST_SHA256.txt`（629 文件）核验通过
**证据等级**：核心表格全部为 MEASURED（原始日志在 `logs/`，原始数据在 `results/raw/`，≥5 reps/配置）
**综合判定**：**Correctness PASS / Kernel Speedup 8.2× / Native E2E Gate FAIL**

后续任何 GPU 优化研究（Kernel Tiling、Kernel Fusion、GPU-aware Batching、SSD–GPU 协同）
都必须与本基线比较，且在独立工作目录进行。**本目录不得修改。**

## 1. 固定实验条件（Baseline V1）

| 条件 | 值 |
|---|---|
| Workload | BigANN100K（n=100,000, d=128, ε²=73984） |
| Frozen Task Graph | 13,011 Tasks（`traces/bigann100k/`） |
| Comparison Count | 154,113,486 |
| CPU 配置 | 16 threads（`numactl --cpunodebind=0 --membind=0`） |
| GPU | RTX 5090（CUDA_VISIBLE_DEVICES=0；GPU1/2 驱动故障不可用） |
| GPU Input | VRAM Resident（批模式口卫） |
| GPU Batch Size | b = 1024 |
| Verification Kernel | FP32 Squared L2 + 阈值 |
| 计时口径 | warm（扣 3.3–4.6 s/进程 CUDA init，驱动 Persistence Off） |
| 测量协议 | ≥5 Repetitions，取中位数 |
| 上游参考 SHA | `reference/DiskJoin` @ `aa86151f97527b1dcf9a032d523c1d67d892454a`（未修改） |

## 2. 核心结果（bigann100k，5 reps 中位数）

| 资产 / 配置 | 结果 | 保存优先级 |
|---|---|---|
| CPU Verification Baseline | 193 ms（16 Threads） | 必须 |
| GPU Verification Baseline | 23.5 ms → **8.2×** | 必须 |
| GPU Batch Sweep | b=1：188.6 ms → b≈1024：23.5 ms | 必须 |
| Native End-to-End | CPU 2.32 s；GPU 2.22 s（1.05×） | 必须 |
| DRAM / VRAM Resident | 0.158 s / 0.0785 s（warm） | 必须 |
| I/O Profiling | O_DIRECT 路径 ≈ 2.12 s（占 E2E 91%） | 必须 |
| Correctness | 3,855,811 对匹配，四实现输出多重集逐位一致 | 必须 |
| 任务图与数据 | 13,011 Tasks / 154,113,486 次比较 | 必须 |
| 环境与原始日志 | 命令、编译参数、驱动/native 版本、5 reps | 必须 |

## 3. 三个后续对照锚点

1. **Kernel 优化**：能否超越 GPU V1 的 23.5 ms？（对照 `GATE_B B1/B2`）
2. **Batching / 任务调度**：能否超越现有 b≈1024 的 batch formation？（对照 batch sweep 曲线）
3. **I/O + GPU Pipeline**：能否突破 1.05× 端到端上限（预注册门槛 ≥1.20× 未达标），
   把 8.2× 计算收益转化为端到端加速？（对照 native mode1 = 2.22 s）

## 4. 主要证据文件

| 内容 | 路径 |
|---|---|
| 最终判定与 Q1–Q8 | `reports/FINAL_RESEARCH_ASSESSMENT.md` |
| GPU 验证主实验 | `reports/GATE_B_GPU_VERIFICATION.md` |
| CPU 瓶颈剖析 | `reports/GATE_A_CPU_PROFILE.md` |
| 正确性报告 | `reports/correctness_report.md` |
| 端到端汇总表 | `results/summary/e2e_summary.csv` |
| GPU 重放汇总表 | `results/summary/gpu_replay_summary.csv` |
| 原始逐次数据 | `results/raw/`（含 `/usr/bin/time` JSON） |
| 环境记录 | `ENVIRONMENT.md` |
| 代码语义审计 | `SOURCE_AUDIT.md` |
| 交接与阻塞项 | `HANDOFF_GPU.md` |
| 实际执行命令 | `command_manifest.tsv` |
| 全部产物校验 | `MANIFEST_SHA256.txt` |

## 5. 适用边界（不要外推的内容）

- **所有 GPU 口径均为 warm**：cold 含 CUDA init 3.3–4.6 s/进程，不可比较，也不代表服务化场景。
- **VRAM 常驻（29.6×）不可外推到论文规模**：100M×128d ≈ 51.2 GB > 单卡 32.6 GB（INFERRED，未实测）。
- **drop-in 1.05× 是负向证据且已否决 drop-in 集成路线**（E2E 被 IO 主导，91%）。
- 未测项：多卡（GPU1/2 驱动 OOM）、10M/100M 数据集（磁盘 99% 满硬阻塞）、cuVS/库基线
  （环境受限）、异构调度（SKIPPED_BY_GATE）。
- p3 数据集存在 ε²=0.2 的一–2 ulp FP 边界 tie，已单独定性，未计入其他数据集结论。

## 6. 消融基线用途与推荐后续组织

未来三组实验（各自只改一个因子，其余与 V1 完全一致）：

| 下一阶段 | 直接比较的 Baseline | 目标 |
|---|---|---|
| Kernel Optimization | GPU V1：23.5 ms | 降低 Verification 时间 |
| GPU Task Scheduling | GPU V1：b=1024 sweep 曲线 | 改进 Batch Formation 与任务排序 |
| I/O + GPU Pipeline | Native GPU V1：2.22 s | 降低端到端时间（先与优化后 CPU 对照分离收益来源） |

复现完全以 V1 为准：以上条件 + 仓库内 `command_manifest.tsv` 中的命令族、
`ENVIRONMENT.md` 中的驱动/CUDA 版本，以及本文件第 1 节的固定实验条件。
