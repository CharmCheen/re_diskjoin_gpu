# SOURCE_AUDIT — 本轮实际阅读的代码与确认的语义

对象：`reference/DiskJoin` @ 固定 SHA `aa86151f97527b1dcf9a032d523c1d67d892454a`（工作树 clean，本轮未改动）。
隔离工作副本：`src/DiskJoin_instr`（= Gate 4A 插桩副本，未改语义）与 `src/DiskJoin_gpu`（本轮 GPU 集成补丁，见下）。

## 1. 执行管线（DiskJoin.h::search，[源码已检查]）
1. **候选生成**：每桶 `knnSearchBaseLayer(i, K)`（HNSW，ef=1000），取 K 近邻桶中心距离/2 为 `dis_to_boundary`，按 arcos 角度累加 ≥ error_bound 截断 → `tasks[i]`（含自身；仅保留 `i ≤ neighbor` → 上三角任务图）。OpenMP 并行。
2. **Gorder 重排**：`order_gorder(shuffled_tasks, window)`，window = budget/4/length/K×2。
3. **DRAM 缓存循环**：按 slot 顺序为 `reordered_tasks[i]` 每个桶调 `cache.find`，miss 时 `posixDirectReadCluster`（**O_DIRECT** lseek+read+memcpy）+ `cache.push`。
4. **Verification**：对每个 slot：同桶（`id2 >= id1 continue`，即 id2<id1）+ 跨桶 k=1..（全 m×n），`dist_l2 < epsilon*epsilon` 严格比较；命中则 `count += (id1%100000==0)+(id2%100000==0)`（**采样计数语义**）。
5. 输出 `recall = count / gt`（**非全量 Recall**）。Count-only：原生不物化任何 pair。

## 2. 已确认语义细节（与本轮实现相关）
- `dist_l2 = utils::L2Sqr`（**平方** L2，float；USE_AVX2 → 8 宽 FMA 部分和 + 水平归约）。`epsilon = sqrt(config.radius)`，即 **radius 是 ε²**。
- `assignment[b]` 严格升序（writeClusters 排序；p1/p3/bigann100k 实测验证）⇒ `id2<id1` ⟺ 索引 `l<j`。
- **无 per-pair 剪枝**：跨桶内层循环前计算 vec1-质心距离（DiskJoin.h:142）但**不用于过滤**（gate5a/HANDOFF 结论本轮复核确认）；真实剪枝只有 error_bound 角度和的候选桶截断。
- 上游 `dist_comp` 是 **float reduction**：bigann100k 154M 次累加出现 ~±400 量化误差（154,113,552 vs uint64 精确和 154,113,486/154,113,584 波动）；P1 范围内 < 2^24 无误差。属上游报告口径，非语义差异。
- main.cpp 每次运行 `index.build()` 会**重写** config 指向的 cluster/metadata/hnsw 文件，且 OMP>1 时 HNSW 建图非确定（Gate 4A 已知）。**本轮事故记录**：初次 Gate A 时用全量 main 跑 sweep 覆盖了 gate4a/gate5a 冻结数据；已立即从 gate4b1/data 与 gate5a/frozen_r1 恢复（p1.clusters md5 复验 = 04042af9… 与覆盖前记录一致；bigann100k = frozen_r1 副本），并对全部 gate 数据文件 chmod 444。p3 恢复自 gate4b1 副本（覆盖前未记录 md5，属已知小风险）。此后所有原生对照均用**只读 search-only 驱动**（`src/DiskJoin_instr/expr/search_only.cpp`，跳过 build），冻结文件 md5 全程未再变化。

## 3. 本轮新增代码（全部在隔离副本/独立文件，未触碰 reference/）
| 文件 | 作用 |
|---|---|
| `src/common_replay.h` | 冻结 metadata/cluster/task-graph 加载器（语义注释如上） |
| `src/replay_cpu.cpp` | B0：固定任务图 CPU 验证 replay（复用上游 `utils::L2Sqr`），per-task 计数/采样输出，可选 pair dump |
| `src/replay_gpu.cu` | GPU trace replay（resident/naive-pageable/naive-pinned 三变体 + 合成 micro sweep），per-task 计数对照 |
| `src/gpu_verify.{h,cu}` | C-ABI GPU 验证模块（batched count kernel，严格 fp32、同桶 l<j、采样计数） |
| `src/DiskJoin_gpu/` | 插桩树副本 + `GATE_GPU` 环境变量门控：0=原生 / 1=GPU验证+原生IO / 2=GPU验证+全量VRAM常驻(跳IO) / 3=CPU验证+DRAM常驻(跳IO, 同口径对照)；含 cand/io/verify/init 分相计时 |
| `src/DiskJoin_instr/expr/search_only.cpp` | 只读 search 驱动 |
| `scripts/oracle_pairs.py` | 独立 numpy oracle（与 C++/CUDA 均不同的实现）：fp32/fp64、counts/pairs、任务子集 |

## 4. 与论文/前序 Gate 结论的一致性
- `add2choice` 2-choice 负载均衡 vs 论文"最近中心"（Gate 3 MISMATCH）本轮未再验证，沿用 Gate 3 结论。
- 采样计数 vs 全量 Recall 口径差异沿用 Gate 3 结论；本轮正确性对照统一采用：per-task 精确计数 + 全量 pair 多重集 + 采样计数三重口径。
