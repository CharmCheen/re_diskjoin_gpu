# GPU Feasibility Study — DiskJoin on RTX 5090 (2026-11-02, amax)

本轮回答一个问题：**用 GPU 加速 DiskJoin 是否值得作为研究项目继续？**

**答案：MIXED / ENGINEERING-leaning —— GPU 验证收益真实（8.2x）但被 IO 瓶颈掩盖（drop-in E2E 仅 1.05x），唯一大收益路径（VRAM 常驻 29.6x）在论文目标规模下内存不可行。建议停止或重新立项。**

## 阅读顺序

1. `reports/FINAL_RESEARCH_ASSESSMENT.md` — 最终判定与 Q1–Q8 回答
2. `reports/GATE_B_GPU_VERIFICATION.md` — 核心实验（B0/B1/B2 + E2E 四模式）
3. `reports/GATE_A_CPU_PROFILE.md` — 瓶颈剖析与 Amdahl 上界
4. `reports/correctness_report.md` — 四实现逐位正确性
5. `ENVIRONMENT.md` / `SOURCE_AUDIT.md` — 环境与代码语义审计
6. `HANDOFF_GPU.md` — 交接与下一步
7. Gate C（SKIPPED_BY_GATE）/ Gate D（BLOCKED_ENV）报告

## 核心结果（bigann100k：n=100k, d=128, ε²=73984, 13,011 任务；5 reps 中位数）

| 配置 | 时间 | 加速比 |
|---|---|---|
| 原生 CPU 管线 16T (mode0) | 2.32 s | 1.00x |
| mode1：GPU 验证 + 原生 IO（warm） | 2.22 s | 1.05x |
| mode3：CPU DRAM 常驻（warm, 16T） | 0.158 s | 14.7x |
| mode2：VRAM 常驻 + 跳 IO（warm, 16T） | 0.0785 s | **29.6x** |
| 验证-only：CPU 16T / GPU resident | 193 ms / 23.5 ms | 8.2x |

- 预注册门槛（原生 E2E ≥1.20x）：**未达标**（mode1 = 1.05x）。
- 所有 GPU 口径必须用 warm（扣 CUDA init 3.3–4.6s/进程，驱动 Persistence Off 所致）。
- 正确性：输出 pair 多重集 native/CPU-replay/GPU/numpy 逐位一致；p3 仅 ε² 边界 1–2 ulp tie。

## 目录结构

```
ENVIRONMENT.md / SOURCE_AUDIT.md     环境与来源审计
reports/                             6 份 Gate/正确性/最终报告
results/summary/                     汇总 CSV（e2e_summary.csv 为核心表）
results/raw/                         原始数据（≥5 reps/配置）
traces/{p1,p3,bigann100k}/           冻结任务图 + 逐任务真值计数 + 缓存 trace
src/                                 CUDA/驱动实现（GATE_GPU=0/1/2/3 四模式）
configs/ runs/ build/ logs/          配置、冻结负载运行目录、构建、日志
command_manifest.tsv                 实际执行的命令族与状态
MANIFEST_SHA256.txt                  全部产物 SHA256
```

## 复现要点

- 编译：`/usr/local/cuda/bin/nvcc -arch=sm_120 -ccbin g++`（本机无 cmake、nvcc 不在 PATH）。
- 运行：`CUDA_VISIBLE_DEVICES=0`（GPU1/2 驱动故障不可用）。
- 受控协议：冻结任务图重放、CPU 侧 `numactl --cpunodebind=0 --membind=0`、≥5 reps 中位数、warm 双口径。
- 边界：全部产物 107MB（限额 512MB）；无 sudo、未改系统、未改 `reference/`（SHA `aa86151f`）与历史 Gate 资产。

## 证据等级

全部表格数字为 **MEASURED**（真实执行，日志在 `logs/`）；标注 INFERRED 的推断（如 100M 不可常驻、break-even）与 NOT_TESTED 项（多卡、cuVS 对比、10M+ 规模）在各报告中单独标明，未与实测混写。
