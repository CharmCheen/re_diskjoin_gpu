// B0: controlled CPU verification replay on a frozen task graph.
// Replays exactly the verification work of native DiskJoin search()
// (same-bucket id2<id1 pairs + cross-bucket m*n pairs, strict eps^2,
// sampled-count semantics), on frozen bucketization + Gorder order.
//
// Deviation vs native inner loop (documented): the useless per-(j,k)
// vec-vs-centroid distance at DiskJoin.h:142 is NOT recomputed here; it
// contributes no output. Native E2E comparisons keep it.
//
// Outputs:
//   --tasks_out FILE : per-task CSV slot,k,a,b,m,n,comparisons,matches,sampled
//   --summary FILE   : one-line JSON summary
// Timing: monotonic wall around the verification loop only (data load excluded).
#include "common_replay.h"
#include "DiskJoin_instr/utils/dist_func.h"
#include <omp.h>
#include <chrono>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    std::string meta_f, clu_f, task_f, tasks_out = "", summary_f = "", pairs_out = "";
    int reps = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() { return std::string(argv[++i]); };
        if (a == "--meta") meta_f = val();
        else if (a == "--clusters") clu_f = val();
        else if (a == "--tasks") task_f = val();
        else if (a == "--tasks_out") tasks_out = val();
        else if (a == "--summary") summary_f = val();
        else if (a == "--pairs_out") pairs_out = val();
        else if (a == "--reps") reps = atoi(argv[++i]);
    }
    Meta meta; meta.load(meta_f.c_str());
    BucketData bd; bd.load(clu_f.c_str(), meta);
    TaskGraph tg; tg.load(task_f.c_str(), meta.cluster_num);
    const size_t d = meta.d;
    const float eps = sqrtf(73984.0f); // placeholder, overridden below by config radius? no: eps^2 passed by --eps2
    // eps^2 is workload-specific; accept explicitly:
    float eps2 = 0.f;
    for (int i = 1; i < argc - 1; i++)
        if (std::string(argv[i]) == "--eps2") eps2 = atof(argv[i + 1]);
    if (eps2 <= 0) { fprintf(stderr, "need --eps2\n"); return 1; }

    // Enumerate tasks once (execution order: slot-major, k-major).
    std::vector<Task> tasks;
    for (size_t s = 0; s < tg.cluster_num; s++) {
        for (uint32_t k = 0; k < tg.deg[s]; k++) {
            uint32_t b = tg.nbrs[tg.nbr_off[s] + k];
            Task t;
            t.slot = (uint32_t)s; t.k = k; t.a = tg.target[s]; t.b = b;
            t.m = (uint32_t)meta.bucket_sizes[t.a];
            t.n = (uint32_t)meta.bucket_sizes[t.b];
            t.off_a = meta.pos[t.a] * d; t.off_b = meta.pos[t.b] * d;
            t.ids_a = &meta.assignment[meta.assign_off[t.a]];
            t.ids_b = &meta.assignment[meta.assign_off[t.b]];
            tasks.push_back(t);
        }
    }

    std::vector<uint64_t> task_matches(tasks.size(), 0), task_sampled(tasks.size(), 0);
    std::vector<std::pair<uint64_t, uint64_t>> all_pairs;
    if (!pairs_out.empty()) all_pairs.reserve(200000);
    uint64_t total_dist = 0, total_sampled = 0;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        std::fill(task_matches.begin(), task_matches.end(), 0);
        std::fill(task_sampled.begin(), task_sampled.end(), 0);
        total_sampled = 0; total_dist = 0;
        if (!pairs_out.empty()) all_pairs.clear();
        auto t0 = std::chrono::steady_clock::now();
        for (size_t ti = 0; ti < tasks.size(); ti++) {
            const Task& t = tasks[ti];
            const float* A = bd.vecs.data() + t.off_a;
            const float* B = bd.vecs.data() + t.off_b;
            uint64_t matches = 0, sampled = 0;
            if (t.k == 0) { // same-bucket: strict id2<id1 == index l<j
                #pragma omp parallel for schedule(dynamic, 8) reduction(+:matches) reduction(+:sampled)
                for (long j = 0; j < (long)t.m; j++) {
                    uint64_t lo_m = 0, lo_s = 0;
                    for (size_t l = 0; l < (size_t)j; l++) {
                        float dist = utils::L2Sqr(A + j * d, B + l * d, &d);
                        if (dist < eps2) {
                            lo_m++;
                            if (t.ids_a[j] % 100000 == 0) lo_s++;
                            if (t.ids_b[l] % 100000 == 0) lo_s++;
                            if (!pairs_out.empty())
                                all_pairs.emplace_back(std::min(t.ids_a[j], t.ids_b[l]),
                                                       std::max(t.ids_a[j], t.ids_b[l]));
                        }
                    }
                    matches += lo_m; sampled += lo_s;
                }
            } else {
                #pragma omp parallel for schedule(dynamic, 8) reduction(+:matches) reduction(+:sampled)
                for (long j = 0; j < (long)t.m; j++) {
                    uint64_t lo_m = 0, lo_s = 0;
                    for (size_t l = 0; l < t.n; l++) {
                        float dist = utils::L2Sqr(A + j * d, B + l * d, &d);
                        if (dist < eps2) {
                            lo_m++;
                            if (t.ids_a[j] % 100000 == 0) lo_s++;
                            if (t.ids_b[l] % 100000 == 0) lo_s++;
                            if (!pairs_out.empty())
                                all_pairs.emplace_back(std::min(t.ids_a[j], t.ids_b[l]),
                                                       std::max(t.ids_a[j], t.ids_b[l]));
                        }
                    }
                    matches += lo_m; sampled += lo_s;
                }
            }
            task_matches[ti] = matches;
            task_sampled[ti] = sampled;
            total_sampled += sampled;
            total_dist += (t.k == 0) ? (uint64_t)t.m * (t.m - 1) / 2 : (uint64_t)t.m * t.n;
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        if (sec < best) best = sec;
        printf("rep%d verify_wall=%.6f dist_comp=%llu sampled=%llu\n",
               r, sec, (unsigned long long)total_dist, (unsigned long long)total_sampled);
    }
    if (!pairs_out.empty()) {
        std::sort(all_pairs.begin(), all_pairs.end());
        FILE* pf = fopen(pairs_out.c_str(), "w");
        for (auto& pr : all_pairs) fprintf(pf, "%llu %llu\n", (unsigned long long)pr.first, (unsigned long long)pr.second);
        fclose(pf);
    }
    if (!tasks_out.empty()) {
        FILE* f = fopen(tasks_out.c_str(), "w");
        fprintf(f, "task_id,slot,k,a,b,m,n,comparisons,matches,sampled\n");
        for (size_t ti = 0; ti < tasks.size(); ti++) {
            const Task& t = tasks[ti];
            uint64_t cmp = (t.k == 0) ? (uint64_t)t.m * (t.m - 1) / 2 : (uint64_t)t.m * t.n;
            fprintf(f, "%zu,%u,%u,%u,%u,%u,%u,%llu,%llu,%llu\n", ti, t.slot, t.k, t.a, t.b,
                    t.m, t.n, (unsigned long long)cmp,
                    (unsigned long long)task_matches[ti], (unsigned long long)task_sampled[ti]);
        }
        fclose(f);
    }
    if (!summary_f.empty()) {
        FILE* f = fopen(summary_f.c_str(), "w");
        fprintf(f, "{\"verify_wall_best\":%.6f,\"reps\":%d,\"threads\":%d,\"tasks\":%zu,\"dist_comp\":%llu,\"sampled\":%llu}\n",
                best, reps, omp_get_max_threads(), tasks.size(),
                (unsigned long long)total_dist, (unsigned long long)total_sampled);
        fclose(f);
    }
    return 0;
}
