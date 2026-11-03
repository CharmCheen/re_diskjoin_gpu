# DiskJoin GPU Feasibility — amax 执行指令（2026-11-02）

## 角色与目标
你是运行在 `amax` 的 OpenCode 工程 Agent。当前用户授权你在**有明确资源上限**的条件下执行 GPU feasibility 实验，并交付可审核的结果。先检查本地仓库/工作树与已完成的 Gate，再运行实验；不要仅写计划或伪代码。严禁把微基准成绩宣称为原生 DiskJoin 端到端加速。

实验目标（依次）：
1. 验证 RTX 5090 上 CUDA Kernel 实际可运行，记录环境与 GPU–NUMA 拓扑。
2. 用现有 P1/P3 等已验证数据及最小必要合成任务，建立 CPU Verification 基线、Bucket Task Trace，并证明结果可复现。
3. 运行 GPU 单任务、GPU Batched 两种 Verification 实现，测量 H2D/Kernel/输出/CPU wall time，进行正确性对照。
4. 如果实现成本允许且不突破资源上限，运行固定候选任务、相同结果语义下的 CPU/GPU Native End-to-End 对照；否则明确标记 `MICROBENCH_ONLY`，不得判定端到端加速成立。
5. 生成完整实验报告及机器可读结果，并给出继续研究或停止的证据。

## 强制安全边界
- 当前根分区 99% 已用，用户可用空间约 9.6 GiB（执行时必须复测）。创建数据、缓存或编译产物前确认可用空间 >=7 GiB；实验过程中一旦低于 7 GiB，立即停止新任务。所有新增产物总量目标 <=512 MiB，绝对上限 1 GiB。**禁止**运行会重新生成大规模数据的脚本，尤其是 `datasets/create` 类脚本；禁止覆盖已有数据、清理其他用户文件或修改系统分区。
- 不使用 sudo；不升级驱动/系统 CUDA、不改全局 Python 环境；不安装大型 Torch/cuVS wheel。优先已有 Python/CUDA 库；有 `nvcc` 时可在隔离目录编译小型 CUDA 实验程序。如果没有任何支持 `sm_120` 的 GPU 后端，只做环境诊断并 `BLOCKED_ENV`，不要通过海量下载强行解决。
- 先使用 GPU 0（之前亲和 CPU Socket 0），**再次确认该卡在实验当下仍空载**。若被占用，选择另一张空闲卡并对应调整 NUMA 绑定；所有 GPU 程序必须显式指定 `CUDA_VISIBLE_DEVICES`。不可抢占或终止其他用户进程。
- 单卡实验显存用量 <=8 GiB；CPU 内存新增占用目标 <=8 GiB；每条测量命令设置 `timeout`，防止失控。不要使用全部三张 GPU。无需执行任何三卡 Benchmark。
- 初步单组实验尽量 <=300 秒；每个配置至少 3 次正式重复，优先使用 5 次。每阶段先 smoke，再扩大负载。在访问实际 SSD 前先确认不会生成超出限额的临时文件。
- 禁止 git reset/clean/stash、覆盖无关改动、修改 `reference/`、删除 frozen Gate 资产、提交/push。原有 Gate 0–4A 及 4B1 的状态须原样保留。将全部新代码/报告置于 `/home/qch/DiskJoin/diskjoin_repro_pack/workspace/gpu_feasibility_20261102/`，如路径实际不同以本地确认的项目根目录为准。
- 不因一项测试失败而修改实验定义/阈值来获取 PASS；记录失败和原因。

## Stage 0：目录与环境审计
1. 在项目根目录先执行 `pwd; git status --short; df -h /; nvidia-smi; nvidia-smi topo -m; numactl --hardware`（最后一项如未安装仅记录）。读取 `workspace/HANDOFF.md`, `workspace/GATE4A_REPORT.md`、已有 4B1 报告和可用测试脚本，确认已经完成的工作；禁止重复运行大规模前置实验。
2. 检测 `nvcc --version`、`gcc/g++ --version`、CMake、现有 Python 中的 `torch`、`cupy`、`cuvs` 是否安装。`nvidia-smi` 显示的 CUDA 版本是驱动支持上限，不代表 CUDA Toolkit 已装。RTX 5090 为 `sm_120`；GPU 可见不代表 GPU Kernel 能成功运行。运行真正的 GPU 向量计算 + synchronize smoke，失败则记录完整错误。
3. 记录 GPU index/UUID、GPU 与 NUMA 对应、CPU affinity、CPU threads、系统负载、PCIe link speed/width（能够只读查询时）、GPU free VRAM、磁盘可用量。为 CPU/GPU 比较选择本地 NUMA 节点；如使用 `numactl`，不得默认它已经生效，需验证实际亲和。
4. 在实验目录存 `environment.txt`, `preflight.log`, `command_manifest.tsv`。每条命令记录运行参数、返回码、耗时和日志路径。

## Stage 1：先审计正确性语义，再生成 Trace
1. 实际阅读本地 `DiskJoin.h`、距离函数、Gorder/Cache 和 Gate 4A 实验代码，确定具体输入、输出及可导出的任务层次，记录实现对应文件和函数名。官方公开实现中同桶有 `id2 < id1` 的条件，距离判断为严格 `< epsilon^2`，并通过采样计数评估 Recall；本地实现需现场核对，不要假定与官方一致。
2. 注意官方公开源码在 cross-bucket loop 中计算到 centroid 的距离但没有使用该值作为剪枝条件；不要凭论文描述假设已存在 per-pair pruning。区分 Bucket Pair Candidate Pruning、同桶 ID 去重和真正的向量对剪枝。
3. 生成轻量级 Task Trace（只记录 task id、bucket id、m/n/d、输入来源、同桶标记、potential comparisons、真实 verified comparisons、CPU count/hash、顺序），优先复用现存 Bucket 数据。不可导出海量向量对明细。采用固定随机种子，记录数据版本和 SHA256。
4. CPU reference：遵守原始候选集合、同桶 ID 去重、阈值严格比较与采样计数语义；增加可独立校验的小规模 all-pairs Oracle。执行 GPU 前必须有 CPU count/hash reference。
5. 选取 P1/P3 中实际出现的 Bucket，额外创建**内存内** 32/128/512/2048 大小、与本地维度一致的微基准输入；合成数据只能用于性能边界分析，不能替代真实数据的结果。

## Stage 2：GPU 对照实验
必须提供三个可区分的对照：
- B0：现有优化 CPU Verification（先本地 NUMA 16 物理核，再视情况做跨 Socket 32 核）。保留 AVX/编译选项，报告确实启用的优化路径。
- B1：最简单正确的单 Bucket Pair GPU 距离计算 + 阈值比较 + Count-only 输出；保留同桶去重条件与任务列表，不允许额外比较非候选 Bucket Pair。
- B2：多 Bucket Pair batching，把同一批任务提交给 GPU 以摊薄启动开销。必须报告 padding 和无效计算，防止合并后意外做新的 cross-product。
如果现有 cuVS 兼容环境可用，增加 B3：cuVS 的 squared L2 / L2Expanded 相应实现，注意输出距离矩阵的 O(mn) 内存成本。若没有 cuVS，不得把自写 Kernel 伪称为 cuVS Baseline。GPU 的 TF32/FP16 等近似模式在 correctness 阶段禁用，先统一 FP32；阈值附近的差异另外报告。

固定数据、阈值、任务集合与结果语义，分别测量：
- `T_cpu_wall`：从 CPU 输入已就绪开始，到结果可供主机读取的全程耗时；
- `T_h2d` / `T_kernel` / `T_d2h`：同步后分段测量，另记 GPU launch、可能的 compaction；
- `T_gpu_resident`：输入已在 VRAM 的 GPU 计算阶段；
- `T_gpu_e2e`：真实主机输入 + 所有必要数据传输 + GPU 计算 + 返回 Count 的墙钟时间；
- Verified pairs/s、Task/s、峰值 VRAM、CPU 线程数、NUMA pinning、Batch size、有效计算比例。
CUDA Events 用于 GPU 流内计时，CPU monotonic wall clock 用于完整调用。异步调用后必须同步，不能以 kernel launch 时间代替完成时间。

参数最小集合：
- Bucket size 32、128、512、2048（与真实 P1/P3 规模并列报告；若真实维度使 2048 超资源预算则减小），
- Batch size 1、8、32、128，
- 每个形状至少 3–5 次正式重复并记录 median/min/max、运行顺序与环境变动，
- GPU resident / Host pageable / Host pinned（三者条件允许时），
- 对于 Native DiskJoin：真实 Candidate Tasks + 一致的输出语义。若 native 改造成本高，先完成固定 Trace Replay，报告 `MICROBENCH_ONLY`。

一个输出为 Count-only 的 GPU 实现必须避免把整个距离矩阵拷回 CPU；可以在 GPU 做阈值判断和归约。完整 Pair Materialization 如需研究，应单列不同结果语义和成本，在当前磁盘约束下不执行大规模输出。

## Stage 3：Gate 判定
- Correctness：同一任务图、同一阈值及严格比较规则下，GPU 与 CPU 每任务计数应一致；数值舍入差异需要独立复核阈值边界，不可掩盖。采样计数的原有约定也必须一致；全量配对输出和采样 Recall 不可混淆。
- GPU break-even：画出 Bucket size 与 GPU/CPU 端到端 Verification speedup；画出 Batch size 与 Throughput。Kernel-only 快不构成端到端收益。
- Native E2E：只有真正集成 DiskJoin，计入 Bucketing、Orchestration、I/O 和输出成本，才能声称 Native end-to-end speedup。没有集成则标记 `NOT_TESTED`，不能称 PASS。
- 论文探索 Gate：预注册目标为至少一个代表性真实负载 GPU Native E2E >=1.20x，第二独立真实负载有正收益，且无实质结果退化。这个目标若当前环境无法完成，则判定 `INCONCLUSIVE`，保留 microbenchmark 发现。
- 本轮不要开展 GPU 调度算法的创新实验，除非 Gate 2 确认的价值已被完整测量。尤其不要修改 Gorder、Belady 和原有缓存以“追求更好结果”。

## 实验记录和交付
生成：
- `ENVIRONMENT.md`：环境、设备、可用存储、GPU/NUMA、库版本与所用后端；
- `SOURCE_AUDIT.md`：本地真实代码路径、运行语义、是否有逐对剪枝、输出语义、源码差异；
- `correctness.csv`：任务计数、配对 hash 或其他小规模 Oracle、阈值边界误差；
- `benchmark.csv`：运行 ID、方法、数据来源、bucket size、dim、batch size、CPU threads、GPU UUID、timing breakdown、throughput、VRAM、准确性状态；
- `GPU_FEASIBILITY_REPORT.md`：命令与环境、实验矩阵、关键表格/图、失败项、是否真正 Native E2E、证据与复现指令、Gate 判定；
- `HANDOFF_GPU.md`：下一步最小行动和所有未完成事项；
- `command_manifest.tsv`：实际执行命令及退出码。

报告必须把三类结果分开：`MEASURED`（真实执行）、`INFERRED`（分析估计）、`NOT_TESTED`。只有真实完成且正确性通过的项目才标 PASS。不要创造不存在的速度提升。运行结束检查 `df -h /`、`nvidia-smi`、新生成文件大小及 git status；确认没有改动冻结资产。最后用一页总结给用户：GPU 当前是否有效、在哪些规模下有效、是否值得进入联合调度 Gate、当前有哪些阻塞。
