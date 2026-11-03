# ENVIRONMENT — amax (measured 2026-11-02)

所有条目为真实执行输出（`logs/preflight.log`、`nvidia-smi`、`/tmp/gvutest*` 探针）。

## 主机
| 项 | 值 |
|---|---|
| OS | Ubuntu 22.04（内核 6.8, x86_64） |
| CPU | 2 × Intel Xeon Gold 6526Y，64 逻辑核（32 物理） |
| RAM | 125 GiB（node0 ≈64G / node1 ≈64G） |
| 磁盘 | /dev/sda2 ext4 879G，**已用 99%，实验期间可用 ≈11 GiB** |
| GPU | 3 × NVIDIA GeForce RTX 5090（sm_120，170 SM，32.6 GiB/卡） |
| Driver | 570.153.02（CUDA 12.8 上限），**Persistence Mode: Off** |
| CUDA Toolkit | /usr/local/cuda-12.8（nvcc V12.8.61），**不在 PATH**，需 `-ccbin g++` |
| g++ | 11.4.0；cmake 本用户不可用（用 g++ 直接编译） |
| Python | 3.10 + numpy 2.2.6（`pip install --user numpy`，16.8MB，本轮新增的唯一系统级依赖） |
| OpenMP | g++ 4.5 支持 |

## GPU 拓扑（nvidia-smi topo -m，实测）
| GPU | NUMA | CPU Affinity | 互联 |
|---|---|---|---|
| 0 | 0 | 0-15,32-47 | 与 GPU1/2 均为 SYS（**无 NVLink**） |
| 1 | 1 | 16-31,48-63 | 与 GPU2 为 NODE |
| 2 | 1 | 16-31,48-63 | — |

## 关键环境发现（MEASURED）
1. **CUDA 上下文创建极慢**：空进程 `cudaSetDevice` = **4.62s**（GPU0，冷）；有常驻 CUDA 进程时 = **3.0–3.3s**；`numactl` 绑定无改善。根因未查明（驱动级）。`cudaMalloc`/`cudaMemcpy` 正常（51MB H2D ≈ 5ms）。对短生命周期进程这是压倒性开销；对常驻服务可摊销。
2. **GPU1/GPU2 无法使用**：`cudaSetDevice` 直接返回 `out of memory`（`/tmp/gvutest2` 探针，GPU1/GPU2 复现，与 host NUMA 绑定无关）。GPU0 正常。此为驱动/系统级问题，无 sudo 无法修复。→ Gate D（多卡）被环境阻塞。
3. 磁盘 99% 满（可用 11G）：本轮新增产物总量 107MB（< 512MB 目标）。

## 实验期间 GPU 使用纪律
- 全部实验 `CUDA_VISIBLE_DEVICES=0`；实验前 `nvidia-smi` 确认空载（一个自建的 cudahold 常驻探针进程除外，占 507MiB，用于驱动保温测量，实验结束后已终止）。
- 峰值 VRAM：51.6MB（bigann100k 全量常驻）<< 8GiB 限制。
- 每条命令带 `timeout`；无 sudo、无系统修改、无大规模下载。
