# GATE D — 多 GPU 扩展（BLOCKED，2026-11-02）

**状态：BLOCKED_ENV**（环境阻塞，未执行任何多卡实验，无伪造数据。）

## 阻塞证据（MEASURED）

- 主机有 3 × RTX 5090，但 **仅 GPU0 可用**：
  - `CUDA_VISIBLE_DEVICES=1 /tmp/gvutest2` 与 `CUDA_VISIBLE_DEVICES=2 /tmp/gvutest2`（最小 `cudaSetDevice`+`cudaMalloc` 探针）在 GPU1/GPU2 上直接返回 **`out of memory`**（32GB 空闲 VRAM 下），复现于 `logs/preflight.log` 与 ENVIRONMENT.md 记录；
  - 与 host NUMA 绑定无关（node0/node1 均复现）；无 sudo，无法重启驱动或改 Persistence Mode；
  - GPU0 一切正常（sm_120 smoke、全部本轮实验）。
- Runbook 同时规定"不要使用全部三张 GPU，无需执行任何三卡 Benchmark"，故即使 GPU1/2 可用，本轮也不做多卡扩展测量。

## 关联发现（MEASURED，影响未来多卡设计）

- 三卡互联均为 **SYS（跨 socket PCIe，无 NVLink）**（`nvidia-smi topo -m`）；
- GPU0 亲和 NUMA0，GPU1/2 亲和 NUMA1 ⇒ 数据划分需按 NUMA 分域，否则 H2D 带宽受损；
- CUDA 上下文创建 3.3–4.6s/进程 ⇒ 多卡多进程方案必须常驻化，否则 init 开销淹没收益。

## 未来开展本 Gate 的前提

1. 管理员修复 GPU1/2 的 `cudaSetDevice` OOM（驱动重载/ECC/持久化模式排查）；
2. 得到多卡 benchmark 的显式授权（Runbook 例外）；
3. 单卡上先解决 IO/常驻路径（见 GATE_B Decision），否则多卡只放大同一个被 IO 压死的管线。

## Decision

BLOCKED_ENV —— 本轮不执行，不计入任何性能结论。
