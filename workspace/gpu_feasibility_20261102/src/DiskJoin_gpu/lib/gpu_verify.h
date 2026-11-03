// GPU verification module for native DiskJoin E2E integration (isolated copy).
// Exposes a C ABI so the C++ TU (compiled with g++) can link this nvcc-built
// object without exposing CUDA headers.
//
// Semantics identical to replay_gpu.cu count_kernel:
//   squared L2 fp32, strict < eps2, same-bucket l<j, sampled id%100000 count.
#pragma once
#include <cstddef>
#include <cstdint>

struct GvuTask {
    const float* A;        // device pointer, m*d floats
    const float* B;        // device pointer, n*d floats
    const uint32_t* ida;   // device pointer, m ids
    const uint32_t* idb;   // device pointer, n ids
    uint32_t m, n, d, same;
};

extern "C" {
// One-time device allocation of a bucket arena + id array.
int gvu_init(size_t vec_floats, const float* vecs_host,
             size_t n_ids, const uint32_t* ids_host,
             size_t max_tasks);
void gvu_finalize();

// Run a batch of tasks (host array), accumulating sampled count on device.
// Returns 0 on success. match_total/sample_total are optional host outputs
// (D2H of this batch's counts; combined with cumulative counters).
int gvu_run(const GvuTask* tasks_host, int ntasks, float eps2,
            unsigned long long* batch_matches, unsigned long long* batch_sampled);

// Cumulative counters since init.
unsigned long long gvu_total_sampled();
unsigned long long gvu_total_matches();
size_t gvu_peak_vram_bytes();

// Arena helpers for io_preserved mode (per-slot upload into device scratch).
const float* gvu_vecs_base();
const uint32_t* gvu_ids_base();
int gvu_arena_upload(size_t float_offset, const float* src, size_t floats);
int gvu_sync();
}
