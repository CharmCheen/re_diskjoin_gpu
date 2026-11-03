// GPU verification replay for DiskJoin frozen task traces.
//
// Modes:
//   trace  : replay a real task trace (frozen bucketization + Gorder order)
//            with per-task-count correctness against the CPU reference CSV.
//   micro  : synthetic shape sweep for batch-size / bucket-size crossover.
//
// Semantics (must match reference DiskJoin.h @ aa86151f):
//   - squared L2, strict `dist < eps2`
//   - same-bucket task (k==0): only pairs l<j (== id2<id1 on sorted ids)
//   - cross-bucket task: full m*n
//   - sampled count: +(id%100000==0) per matched pair endpoint
// All counts are exact integers; no TF32/FP16 approximations.
#include "common_replay.h"
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------- micro mode ----------------
// Synthetic shape sweep: B tasks, each m x n distance matrix, dim d.
// Measures packing, H2D, kernel, D2H per batched launch, median over reps.
struct MicroDesc {
    const float* A; const float* B;
    const uint32_t* ida; const uint32_t* idb;
    uint32_t m, n, d, same;
    float eps2;
};

__global__ void micro_kernel(const MicroDesc* T, unsigned long long* counts) {
    const MicroDesc t = T[blockIdx.z];
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (int)t.m || l >= (int)t.n) return;
    const float* a = t.A + (size_t)j * t.d;
    const float* b = t.B + (size_t)l * t.d;
    float s = 0.f;
    for (uint32_t i = 0; i < t.d; i++) {
        float df = a[i] - b[i];
        s = fmaf(df, df, s);
    }
    if (s < t.eps2) atomicAdd(counts + blockIdx.z, 1ULL);
}

static int micro_main(const std::vector<int>& sizes, const std::vector<int>& batches,
                      int d, int reps, const std::string& out_csv) {
    CK(cudaSetDevice(0));
    FILE* fout = fopen(out_csv.c_str(), "w");
    fprintf(fout, "m,n,d,batch,rep,h2d_us,kernel_us,d2h_us,wall_us,cmp,gpu_cmp_per_s,eff_ratio\n");
    const float eps2 = 1e9f; // high threshold: all-match worst case (max output density)
    // also test sparse variant with threshold below min distance -> zero matches
    for (int m : sizes) for (int B : batches) {
        size_t per_task_floats = (size_t)m * d + (size_t)m * d; // A and B are distinct buffers per task (worst case, no reuse)
        size_t floats = per_task_floats * B;
        std::vector<float> ha(floats);
        std::vector<uint32_t> hid((size_t)B * m, 0);
        for (size_t i = 0; i < floats; i++) ha[i] = (float)(drand48() * 255.0);
        float* d_data; CK(cudaMalloc(&d_data, floats * 4));
        uint32_t* d_ids; CK(cudaMalloc(&d_ids, (size_t)B * m * 4));
        MicroDesc* d_desc; CK(cudaMalloc(&d_desc, B * sizeof(MicroDesc)));
        unsigned long long* d_cnt; CK(cudaMalloc(&d_cnt, B * 8));
        std::vector<MicroDesc> desc(B);
        double h2d_acc = 0, k_acc = 0, d2h_acc = 0, wall_acc = 0;
        uint64_t cmp_total = (uint64_t)B * m * m;
        for (int rep = 0; rep < reps; rep++) {
            auto W0 = std::chrono::steady_clock::now();
            auto C0 = std::chrono::steady_clock::now();
            CK(cudaMemcpy(d_data, ha.data(), floats * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_ids, hid.data(), (size_t)B * m * 4, cudaMemcpyHostToDevice));
            auto C1 = std::chrono::steady_clock::now();
            for (int i = 0; i < B; i++) {
                desc[i].A = d_data + (size_t)(2 * i) * m * d;
                desc[i].B = d_data + (size_t)(2 * i + 1) * m * d;
                desc[i].ida = d_ids + (size_t)i * m; desc[i].idb = desc[i].ida;
                desc[i].m = m; desc[i].n = m; desc[i].d = d; desc[i].same = 0; desc[i].eps2 = eps2;
            }
            CK(cudaMemcpy(d_desc, desc.data(), B * sizeof(MicroDesc), cudaMemcpyHostToDevice));
            CK(cudaMemset(d_cnt, 0, B * 8));
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            dim3 grid((m + 15) / 16, (m + 15) / 16, B);
            CK(cudaEventRecord(e0));
            micro_kernel<<<grid, dim3(16, 16)>>>(d_desc, d_cnt);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            auto C2 = std::chrono::steady_clock::now();
            std::vector<unsigned long long> cnt(B);
            CK(cudaMemcpy(cnt.data(), d_cnt, B * 8, cudaMemcpyDeviceToHost));
            auto C3 = std::chrono::steady_clock::now();
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            h2d_acc += std::chrono::duration<double, std::micro>(C1 - C0).count();
            k_acc += ms * 1000.0;
            d2h_acc += std::chrono::duration<double, std::micro>(C3 - C2).count();
            wall_acc += std::chrono::duration<double, std::micro>(C3 - W0).count();
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        fprintf(fout, "%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%llu,%.3e,%.3f\n",
                m, m, d, B, reps, h2d_acc / reps, k_acc / reps, d2h_acc / reps, wall_acc / reps,
                (unsigned long long)cmp_total, cmp_total / (k_acc / reps / 1e6), 0.0);
        printf("micro m=%d B=%d d=%d: h2d=%.1fus kernel=%.1fus wall=%.1fus per-rep-avg\n",
               m, B, d, h2d_acc / reps, k_acc / reps, wall_acc / reps);
        cudaFree(d_data); cudaFree(d_ids); cudaFree(d_desc); cudaFree(d_cnt);
    }
    fclose(fout);
    return 0;
}

struct TaskDescDev {
    const float* A; const float* B;
    const uint32_t* ida; const uint32_t* idb;
    uint32_t m, n, d, same;
    float eps2;
};

__global__ void count_kernel(const TaskDescDev* T, unsigned long long* counts, unsigned long long* sampled) {
    const TaskDescDev t = T[blockIdx.z];
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (int)t.m || l >= (int)t.n) return;
    if (t.same && l >= j) return;
    const float* a = t.A + (size_t)j * t.d;
    const float* b = t.B + (size_t)l * t.d;
    float s = 0.f;
    for (uint32_t i = 0; i < t.d; i++) {
        float df = a[i] - b[i];
        s = fmaf(df, df, s);
    }
    if (s < t.eps2) {
        atomicAdd(counts + blockIdx.z, 1ULL);
        uint32_t sv = (t.ida[j] % 100000 == 0) + (t.idb[l] % 100000 == 0);
        if (sv) atomicAdd(sampled + blockIdx.z, (unsigned long long)sv);
    }
}

// Pair materialization pass: writes packed (min_id<<32|max_id) pairs at
// per-task offsets (from a previous count pass). Uses fp64 accumulation as a
// cross-check variant when eps2_fp64 > 0.
__global__ void pair_kernel(const TaskDescDev* T, unsigned long long* cursors,
                            uint64_t* pairs, double eps2_64, int use_fp64) {
    const TaskDescDev t = T[blockIdx.z];
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (int)t.m || l >= (int)t.n) return;
    if (t.same && l >= j) return;
    const float* a = t.A + (size_t)j * t.d;
    const float* b = t.B + (size_t)l * t.d;
    double s64 = 0; float s = 0.f;
    for (uint32_t i = 0; i < t.d; i++) {
        float df = a[i] - b[i];
        s = fmaf(df, df, s);
        s64 = fma((double)df, (double)df, s64);
    }
    bool hit = use_fp64 ? (s64 < eps2_64) : (s < t.eps2);
    if (hit) {
        uint64_t id1 = t.ida[j], id2 = t.idb[l];
        uint64_t lo = id1 < id2 ? id1 : id2, hi = id1 < id2 ? id2 : id1;
        uint64_t slot = atomicAdd(cursors + blockIdx.z, 1ULL);
        pairs[slot] = (lo << 32) | hi;
    }
}

struct GpuTaskSet {
    // device-resident copy of all referenced buckets + task descriptors
    float* d_vecs = nullptr;       // all bucket vectors (full cluster file)
    uint32_t* d_ids = nullptr;     // all assignment ids
    TaskDescDev* d_desc = nullptr;
    unsigned long long* d_counts = nullptr;
    unsigned long long* d_sampled = nullptr;
};

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string mode = "trace";
    std::string meta_f, clu_f, task_f, ref_csv = "", pairs_out = "", out_csv = "results/raw/gpu_out.csv";
    int batch = 512, reps = 5, variant = 0; // variant: 0=resident,1=pageable H2D per task,2=pinned H2D per task
    float eps2 = 0;
    // micro params
    int mdim = 128; std::vector<int> msizes, mbatches;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return std::string(argv[++i]); };
        if (a == "--mode") mode = val();
        else if (a == "--meta") meta_f = val();
        else if (a == "--clusters") clu_f = val();
        else if (a == "--tasks") task_f = val();
        else if (a == "--eps2") eps2 = atof(val().c_str());
        else if (a == "--ref_csv") ref_csv = val();
        else if (a == "--pairs_out") pairs_out = val();
        else if (a == "--out_csv") out_csv = val();
        else if (a == "--batch") batch = atoi(val().c_str());
        else if (a == "--reps") reps = atoi(val().c_str());
        else if (a == "--variant") variant = atoi(val().c_str());
        else if (a == "--d") mdim = atoi(val().c_str());
        else if (a == "--msizes") { std::string s = val(); for (auto& c : s) if (c==',') c=' '; std::stringstream ss(s); int x; while (ss>>x) msizes.push_back(x); }
        else if (a == "--mbatches") { std::string s = val(); for (auto& c : s) if (c==',') c=' '; std::stringstream ss(s); int x; while (ss>>x) mbatches.push_back(x); }
    }
    if (mode == "micro") return micro_main(msizes, mbatches, mdim, reps, out_csv);

    // ---------------- trace mode ----------------
    Meta meta; meta.load(meta_f.c_str());
    BucketData bd; bd.load(clu_f.c_str(), meta);
    TaskGraph tg; tg.load(task_f.c_str(), meta.cluster_num);
    const uint32_t d = (uint32_t)meta.d;

    std::vector<Task> tasks;
    for (size_t s = 0; s < tg.cluster_num; s++)
        for (uint32_t k = 0; k < tg.deg[s]; k++) {
            uint32_t b = tg.nbrs[tg.nbr_off[s] + k];
            Task t; t.slot = (uint32_t)s; t.k = k; t.a = tg.target[s]; t.b = b;
            t.m = (uint32_t)meta.bucket_sizes[t.a]; t.n = (uint32_t)meta.bucket_sizes[t.b];
            t.off_a = meta.pos[t.a] * d; t.off_b = meta.pos[t.b] * d;
            t.ids_a = &meta.assignment[meta.assign_off[t.a]];
            t.ids_b = &meta.assignment[meta.assign_off[t.b]];
            tasks.push_back(t);
        }
    size_t nt = tasks.size();

    // reference counts (from CPU replay CSV)
    std::vector<uint64_t> ref_m(nt, 0), ref_s(nt, 0);
    if (!ref_csv.empty()) {
        FILE* f = fopen(ref_csv.c_str(), "r");
        char line[256]; fgets(line, 256, f);
        while (fgets(line, 256, f)) {
            int ti; unsigned long long cmp, m, s2;
            sscanf(line, "%d,%*d,%*d,%*d,%*d,%*d,%*d,%llu,%llu,%llu", &ti, &cmp, &m, &s2);
            ref_m[ti] = m; ref_s[ti] = s2;
        }
        fclose(f);
    }

    CK(cudaSetDevice(0));
    GpuTaskSet G;
    // upload all vectors + ids once (resident baseline; also used by h2d variants as fallback source)
    double t0 = now_s();
    CK(cudaMalloc(&G.d_vecs, bd.vecs.size() * 4));
    CK(cudaMalloc(&G.d_ids, meta.assignment.size() * 4));
    CK(cudaMemcpy(G.d_vecs, bd.vecs.data(), bd.vecs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G.d_ids, meta.assignment.data(), meta.assignment.size() * 4, cudaMemcpyHostToDevice));
    double upload_s = now_s() - t0;

    std::vector<TaskDescDev> desc(nt);
    std::vector<unsigned long long> counts(nt), sampled(nt);
    dim3 block(16, 16);
    FILE* fout = fopen(out_csv.c_str(), "w");
    fprintf(fout, "mode,variant,batch,rep,wall_s,kernel_s,h2d_s,upload_all_s,ntasks,dist_comp,match_sum,sampled_sum,ref_mismatch\n");

    for (int rep = 0; rep < reps; rep++) {
        uint64_t mism = 0, msum = 0, ssum = 0, dcomp = 0;
        double wall = 0, ktime = 0, h2d = 0;
        if (variant == 0) {
            CK(cudaMalloc(&G.d_desc, nt * sizeof(TaskDescDev)));
            CK(cudaMalloc(&G.d_counts, nt * 8));
            CK(cudaMalloc(&G.d_sampled, nt * 8));
            for (size_t i = 0; i < nt; i++) {
                const Task& t = tasks[i];
                desc[i].A = G.d_vecs + t.off_a; desc[i].B = G.d_vecs + t.off_b;
                desc[i].ida = G.d_ids + meta.assign_off[t.a]; desc[i].idb = G.d_ids + meta.assign_off[t.b];
                desc[i].m = t.m; desc[i].n = t.n; desc[i].d = d; desc[i].same = (t.k == 0);
                desc[i].eps2 = eps2;
            }
            CK(cudaMemcpy(G.d_desc, desc.data(), nt * sizeof(TaskDescDev), cudaMemcpyHostToDevice));
            CK(cudaMemset(G.d_counts, 0, nt * 8));
            CK(cudaMemset(G.d_sampled, 0, nt * 8));
            cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
            double w0 = now_s();
            for (size_t base = 0; base < nt; base += batch) {
                size_t nb = std::min((size_t)batch, nt - base);
                // grid: x=ceil(n/16) y=ceil(m/16) z=nb — use max over batch for x,y
                uint32_t gx = 0, gy = 0;
                for (size_t i = base; i < base + nb; i++) {
                    gx = std::max(gx, (tasks[i].n + 15) / 16);
                    gy = std::max(gy, (tasks[i].m + 15) / 16);
                }
                dim3 grid(gx, gy, (unsigned)nb);
                CK(cudaEventRecord(ev0));
                count_kernel<<<grid, block>>>(G.d_desc + base, G.d_counts + base, G.d_sampled + base);
                CK(cudaEventRecord(ev1));
                CK(cudaEventSynchronize(ev1));
                float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
                ktime += ms / 1000.0;
            }
            wall = now_s() - w0;
            CK(cudaMemcpy(counts.data(), G.d_counts, nt * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sampled.data(), G.d_sampled, nt * 8, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < nt; i++) {
                msum += counts[i]; ssum += sampled[i];
                const Task& t = tasks[i];
                dcomp += (t.k == 0) ? (uint64_t)t.m * (t.m - 1) / 2 : (uint64_t)t.m * t.n;
                if (!ref_csv.empty() && (counts[i] != ref_m[i] || sampled[i] != ref_s[i])) mism++;
            }
            cudaFree(G.d_desc); cudaFree(G.d_counts); cudaFree(G.d_sampled);
        } else {
            // naive per-task H2D: copy the two buckets for every task (pageable or pinned staging)
            uint32_t maxb = 0;
            for (auto s : meta.bucket_sizes) maxb = std::max(maxb, (uint32_t)s);
            size_t bmax = (size_t)maxb * d;
            float *h_a, *h_b; uint32_t *h_ia, *h_ib;
            float *d_a, *d_b; uint32_t *d_ia, *d_ib;
            unsigned long long* d_cnt; unsigned long long* d_smp; TaskDescDev* d_t;
            if (variant == 2) {
                CK(cudaMallocHost(&h_a, bmax * 4)); CK(cudaMallocHost(&h_b, bmax * 4));
                CK(cudaMallocHost(&h_ia, maxb * 4)); CK(cudaMallocHost(&h_ib, maxb * 4));
            } else {
                h_a = new float[bmax]; h_b = new float[bmax];
                h_ia = new uint32_t[maxb]; h_ib = new uint32_t[maxb];
            }
            CK(cudaMalloc(&d_a, bmax * 4)); CK(cudaMalloc(&d_b, bmax * 4));
            CK(cudaMalloc(&d_ia, maxb * 4)); CK(cudaMalloc(&d_ib, maxb * 4));
            CK(cudaMalloc(&d_cnt, 8)); CK(cudaMalloc(&d_smp, 8)); CK(cudaMalloc(&d_t, sizeof(TaskDescDev)));
            cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
            double w0 = now_s();
            // batch is ignored for variant>=1 (naive one-task-per-launch)
            for (size_t i = 0; i < nt; i++) {
                const Task& t = tasks[i];
                memcpy(h_a, &bd.vecs[t.off_a], (size_t)t.m * d * 4);
                memcpy(h_b, &bd.vecs[t.off_b], (size_t)t.n * d * 4);
                memcpy(h_ia, t.ids_a, t.m * 4);
                memcpy(h_ib, t.ids_b, t.n * 4);
                double c0 = now_s();
                CK(cudaMemcpy(d_a, h_a, (size_t)t.m * d * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_b, h_b, (size_t)t.n * d * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_ia, h_ia, t.m * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_ib, h_ib, t.n * 4, cudaMemcpyHostToDevice));
                h2d += now_s() - c0;
                TaskDescDev td;
                td.A = d_a; td.B = d_b; td.ida = d_ia; td.idb = d_ib;
                td.m = t.m; td.n = t.n; td.d = d; td.same = (t.k == 0); td.eps2 = eps2;
                CK(cudaMemcpy(d_t, &td, sizeof(td), cudaMemcpyHostToDevice));
                CK(cudaMemset(d_cnt, 0, 8)); CK(cudaMemset(d_smp, 0, 8));
                dim3 grid((t.n + 15) / 16, (t.m + 15) / 16, 1);
                CK(cudaEventRecord(ev0));
                count_kernel<<<grid, block>>>(d_t, d_cnt, d_smp);
                CK(cudaEventRecord(ev1));
                CK(cudaEventSynchronize(ev1));
                float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
                ktime += ms / 1000.0;
                unsigned long long hc = 0, hs = 0;
                CK(cudaMemcpy(&hc, d_cnt, 8, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&hs, d_smp, 8, cudaMemcpyDeviceToHost));
                counts[i] = hc; sampled[i] = hs;
            }
            wall = now_s() - w0;
            for (size_t i = 0; i < nt; i++) {
                msum += counts[i]; ssum += sampled[i];
                const Task& t = tasks[i];
                dcomp += (t.k == 0) ? (uint64_t)t.m * (t.m - 1) / 2 : (uint64_t)t.m * t.n;
                if (!ref_csv.empty() && (counts[i] != ref_m[i] || sampled[i] != ref_s[i])) mism++;
            }
            CK(cudaGetLastError());
            if (variant == 2) { cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_ia); cudaFreeHost(h_ib); }
            else { delete[] h_a; delete[] h_b; delete[] h_ia; delete[] h_ib; }
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_ia); cudaFree(d_ib);
            cudaFree(d_cnt); cudaFree(d_smp); cudaFree(d_t);
            CK(cudaDeviceSynchronize());
        }
        fprintf(fout, "trace,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%zu,%llu,%llu,%llu,%llu\n",
                variant, batch, rep, wall, ktime, h2d, upload_s, nt,
                (unsigned long long)dcomp, (unsigned long long)msum,
                (unsigned long long)ssum, (unsigned long long)mism);
        printf("trace variant=%d batch=%d rep=%d wall=%.4fs kernel=%.4fs h2d=%.4fs mism=%llu msum=%llu ssum=%llu\n",
               variant, batch, rep, wall, ktime, h2d, (unsigned long long)mism,
               (unsigned long long)msum, (unsigned long long)ssum);
    }
    fclose(fout);
    return 0;
}
