// Shared loader for frozen DiskJoin artifacts: metadata (binary), cluster
// file (bucket-contiguous float32 vectors), and GATE4A task graph dump.
//
// Semantics mirror reference/DiskJoin/lib/{ClusterIO,DiskJoin}.h at pinned
// SHA aa86151f97527b1dcf9a032d523c1d67d892454a (see SOURCE_AUDIT notes):
//   - bucket b vectors start at float offset pos[b] = cumsum(bucket_sizes)*d
//   - assignment[b] is ascending-sorted global ids, matching file order
//   - verification: squared L2 (float), strict `< eps*eps`
//   - same-bucket pairs: id2 < id1 (== index l < j on sorted ids)
//   - sampled count: for each matched pair, +(id1%100000==0), +(id2%100000==0)
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cassert>

struct Meta {
    size_t n = 0, d = 0, cluster_num = 0, max_points = 0;
    std::vector<size_t> bucket_sizes;
    std::vector<size_t> pos;          // float offset of each bucket in cluster file
    std::vector<uint32_t> assignment; // concatenated per-bucket ascending ids
    std::vector<size_t> assign_off;   // offset per bucket into assignment
    std::vector<float> centroids;

    void load(const char* metafile) {
        FILE* f = fopen(metafile, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", metafile); exit(1); }
        auto rd = [&](void* p, size_t sz) { if (fread(p, 1, sz, f) != sz) { fprintf(stderr, "meta short read\n"); exit(1); } };
        rd(&n, 8); rd(&d, 8); rd(&cluster_num, 8); rd(&max_points, 8);
        bucket_sizes.resize(cluster_num); rd(bucket_sizes.data(), 8 * cluster_num);
        centroids.resize(cluster_num * d); rd(centroids.data(), 4 * cluster_num * d);
        std::vector<float> radii(cluster_num); rd(radii.data(), 4 * cluster_num);
        assignment.resize(n); assign_off.resize(cluster_num + 1);
        size_t cum = 0;
        for (size_t i = 0; i < cluster_num; i++) {
            assign_off[i] = cum;
            std::vector<size_t> tmp(bucket_sizes[i]);
            rd(tmp.data(), 8 * bucket_sizes[i]);
            for (size_t j = 0; j < bucket_sizes[i]; j++) assignment[cum + j] = (uint32_t)tmp[j];
            cum += bucket_sizes[i];
        }
        assign_off[cluster_num] = cum;
        assert(cum == n);
        pos.resize(cluster_num + 1);
        cum = 0;
        for (size_t i = 0; i < cluster_num; i++) { pos[i] = cum; cum += bucket_sizes[i]; }
        pos[cluster_num] = cum;
        fclose(f);
    }
};

struct TaskGraph {
    // One slot per execution step: slot i processes target order[i] against
    // tasks[i][0..deg-1] (tasks[i][0] == target == same-bucket task).
    size_t cluster_num = 0;
    std::vector<uint32_t> target;          // per slot
    std::vector<uint32_t> nbrs;            // flattened deg*cluster_num
    std::vector<uint32_t> deg;             // per slot
    std::vector<size_t> nbr_off;           // per slot offset into nbrs

    void load(const char* taskfile, size_t expect_cluster_num) {
        FILE* f = fopen(taskfile, "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", taskfile); exit(1); }
        std::vector<uint32_t> t, nb; std::vector<uint32_t> dg;
        uint32_t slot, tgt, d32;
        while (fscanf(f, "%u %u %u", &slot, &tgt, &d32) == 3) {
            t.push_back(tgt); dg.push_back(d32);
            for (uint32_t k = 0; k < d32; k++) {
                uint32_t x; if (fscanf(f, "%u", &x) != 1) { fprintf(stderr, "bad tasks file\n"); exit(1); }
                nb.push_back(x);
            }
        }
        fclose(f);
        cluster_num = t.size();
        assert(cluster_num == expect_cluster_num);
        target = t; deg = dg; nbrs = nb;
        nbr_off.resize(cluster_num + 1);
        size_t cum = 0;
        for (size_t i = 0; i < cluster_num; i++) { nbr_off[i] = cum; cum += dg[i]; }
        nbr_off[cluster_num] = cum;
    }
};

// Flat bucket storage: all vectors concatenated in bucket order (exactly the
// layout of the upstream cluster file).
struct BucketData {
    std::vector<float> vecs;
    void load(const char* clusterfile, const Meta& m) {
        size_t total = m.pos[m.cluster_num] * m.d;
        vecs.resize(total);
        FILE* f = fopen(clusterfile, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", clusterfile); exit(1); }
        if (fread(vecs.data(), 4, total, f) != total) { fprintf(stderr, "cluster short read\n"); exit(1); }
        fclose(f);
    }
};

struct Task { // verification task = one bucket pair
    uint32_t slot;      // execution slot (order position)
    uint32_t k;         // 0 = same-bucket, >=1 cross-bucket index
    uint32_t a, b;      // bucket ids (a = target)
    uint32_t m, n;      // sizes
    size_t off_a, off_b; // float offsets into BucketData::vecs
    const uint32_t* ids_a; const uint32_t* ids_b;
};
