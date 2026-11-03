#include "gpu_verify.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "GVU CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

struct GvuDescDev {
    const float* A; const float* B;
    const uint32_t* ida; const uint32_t* idb;
    uint32_t m, n, d, same;
    float eps2;
};

__global__ void gvu_count_kernel(const GvuDescDev* T,
                                 unsigned long long* counts,
                                 unsigned long long* sampled) {
    const GvuDescDev t = T[blockIdx.z];
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

static float* g_vecs = nullptr;
static uint32_t* g_ids = nullptr;
static GvuDescDev* g_desc = nullptr;
static unsigned long long* g_cnt = nullptr;
static unsigned long long* g_smp = nullptr;
static unsigned long long* g_tot_smp = nullptr;
static unsigned long long* g_tot_cnt = nullptr;
static int g_max_tasks = 0;
static size_t g_peak = 0;

extern "C" int gvu_init(size_t vec_floats, const float* vecs_host,
                        size_t n_ids, const uint32_t* ids_host,
                        size_t max_tasks) {
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&g_vecs, vec_floats * 4));
    CK(cudaMalloc(&g_ids, n_ids * 4));
    CK(cudaMalloc(&g_desc, max_tasks * sizeof(GvuDescDev)));
    CK(cudaMalloc(&g_cnt, max_tasks * 8));
    CK(cudaMalloc(&g_smp, max_tasks * 8));
    CK(cudaMalloc(&g_tot_smp, 8));
    CK(cudaMalloc(&g_tot_cnt, 8));
    CK(cudaMemset(g_tot_smp, 0, 8));
    CK(cudaMemset(g_tot_cnt, 0, 8));
    CK(cudaMemcpy(g_vecs, vecs_host, vec_floats * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_ids, ids_host, n_ids * 4, cudaMemcpyHostToDevice));
    g_max_tasks = (int)max_tasks;
    g_peak = vec_floats * 4 + n_ids * 4 + max_tasks * (sizeof(GvuDescDev) + 16) + 16;
    return 0;
}

extern "C" void gvu_finalize() {
    if (g_vecs) cudaFree(g_vecs);
    if (g_ids) cudaFree(g_ids);
    if (g_desc) cudaFree(g_desc);
    if (g_cnt) cudaFree(g_cnt);
    if (g_smp) cudaFree(g_smp);
    if (g_tot_smp) cudaFree(g_tot_smp);
    if (g_tot_cnt) cudaFree(g_tot_cnt);
    g_vecs = nullptr; g_ids = nullptr; g_desc = nullptr;
    g_cnt = nullptr; g_smp = nullptr; g_tot_smp = nullptr; g_tot_cnt = nullptr;
}

extern "C" int gvu_run(const GvuTask* tasks_host, int ntasks, float eps2,
                       unsigned long long* batch_matches,
                       unsigned long long* batch_sampled) {
    if (ntasks == 0) { if (batch_matches) *batch_matches = 0; if (batch_sampled) *batch_sampled = 0; return 0; }
    if (ntasks > g_max_tasks) { fprintf(stderr, "gvu: batch too large\n"); return -1; }
    std::vector<GvuDescDev> desc(ntasks);
    uint32_t gx = 0, gy = 0;
    for (int i = 0; i < ntasks; i++) {
        const GvuTask& t = tasks_host[i];
        desc[i].A = t.A; desc[i].B = t.B; desc[i].ida = t.ida; desc[i].idb = t.idb;
        desc[i].m = t.m; desc[i].n = t.n; desc[i].d = t.d; desc[i].same = t.same;
        desc[i].eps2 = eps2;
        gx = gx > (t.n + 15) / 16 ? gx : (t.n + 15) / 16;
        gy = gy > (t.m + 15) / 16 ? gy : (t.m + 15) / 16;
    }
    CK(cudaMemcpy(g_desc, desc.data(), ntasks * sizeof(GvuDescDev), cudaMemcpyHostToDevice));
    CK(cudaMemset(g_cnt, 0, (size_t)ntasks * 8));
    CK(cudaMemset(g_smp, 0, (size_t)ntasks * 8));
    dim3 grid(gx, gy, (unsigned)ntasks);
    gvu_count_kernel<<<grid, dim3(16, 16)>>>(g_desc, g_cnt, g_smp);
    CK(cudaGetLastError());
    // accumulate into totals with a tiny kernel-less trick: copy back and
    // atomically add via a 1-thread kernel on pinned staging is overkill;
    // use cudaMemcpyAsYNC-safe approach: sum on host is fine (D2H of ntasks*8)
    if (batch_matches || batch_sampled) {
        std::vector<unsigned long long> hc(ntasks), hs(ntasks);
        CK(cudaMemcpy(hc.data(), g_cnt, (size_t)ntasks * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hs.data(), g_smp, (size_t)ntasks * 8, cudaMemcpyDeviceToHost));
        unsigned long long mc = 0, sc = 0;
        for (int i = 0; i < ntasks; i++) { mc += hc[i]; sc += hs[i]; }
        if (batch_matches) *batch_matches = mc;
        if (batch_sampled) *batch_sampled = sc;
    }
    return 0;
}

extern "C" unsigned long long gvu_total_sampled() {
    unsigned long long h = 0;
    if (g_tot_smp) cudaMemcpy(&h, g_tot_smp, 8, cudaMemcpyDeviceToHost);
    return h;
}
extern "C" unsigned long long gvu_total_matches() {
    unsigned long long h = 0;
    if (g_tot_cnt) cudaMemcpy(&h, g_tot_cnt, 8, cudaMemcpyDeviceToHost);
    return h;
}
extern "C" size_t gvu_peak_vram_bytes() { return g_peak; }

extern "C" const float* gvu_vecs_base() { return g_vecs; }
extern "C" const uint32_t* gvu_ids_base() { return g_ids; }
extern "C" int gvu_arena_upload(size_t float_offset, const float* src, size_t floats) {
    CK(cudaMemcpy(g_vecs + float_offset, src, floats * 4, cudaMemcpyHostToDevice));
    return 0;
}
extern "C" int gvu_sync() { CK(cudaDeviceSynchronize()); return 0; }
