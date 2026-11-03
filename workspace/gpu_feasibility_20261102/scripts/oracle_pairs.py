#!/usr/bin/env python3
# Independent numpy oracle for DiskJoin verification semantics.
# Modes: pairs (full multiset compare vs native GATE3 dump), counts (per task).
#
# bigann100k data is uint8-derived: every diff^2 product and partial sum is an
# integer < 2^24, so fp32 arithmetic here is EXACT and tie-free.
# P1 data is continuous: fp32 boundary ties possible; fp64 mode available.
import struct, sys
import numpy as np

def load(meta, clusters):
    f = open(meta, 'rb')
    n, d, c, mp = struct.unpack('4Q', f.read(32))
    bs = list(struct.unpack(f'{c}Q', f.read(8 * c)))
    cent = np.fromfile(f, dtype=np.float32, count=c * d).reshape(c, d)
    f.read(4 * c)
    assign = []
    for i in range(c):
        ids = np.fromfile(f, dtype=np.uint64, count=bs[i]).astype(np.int64)
        assign.append(ids)
    f.close()
    vecs = np.fromfile(clusters, dtype=np.float32)
    pos = np.zeros(c + 1, dtype=np.int64)
    pos[1:] = np.cumsum(bs) * d
    return n, d, c, bs, assign, vecs, pos

def dist_block(A, B, dtype):
    # returns squared distance matrix, chunked over rows of A
    m = len(A)
    out = np.empty((m, len(B)), dtype=dtype)
    CH = max(1, int(4e6) // max(1, len(B) * A.shape[1]))
    for i0 in range(0, m, CH):
        i1 = min(m, i0 + CH)
        df = A[i0:i1, None, :].astype(dtype) - B[None, :, :].astype(dtype)
        np.einsum('ijk,ijk->ij', df, df, out=out[i0:i1])
    return out

def run(meta, clusters, tasksf, eps2, mode, dumpfile, out, fp64=False, seed=0, subset=0):
    dtype = np.float64 if fp64 else np.float32
    n, d, c, bs, assign, vecs, pos = load(meta, clusters)
    lines = [list(map(int, l.split())) for l in open(tasksf)]
    tasks = []
    for li, (slot, tgt, deg, *nbrs) in enumerate(lines):
        for k in range(deg):
            tasks.append((slot, k, tgt, nbrs[k]))
    if subset:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tasks), size=min(subset, len(tasks)), replace=False)
        tasks = [tasks[i] for i in sorted(idx)]
    counts = []
    pairs = []
    for (slot, k, tgt, nbr) in tasks:
        A = vecs[pos[tgt]:pos[tgt] + bs[tgt] * d].reshape(-1, d)
        same = (k == 0)
        B = A if same else vecs[pos[nbr]:pos[nbr] + bs[nbr] * d].reshape(-1, d)
        m = len(A)
        if same:
            S = dist_block(A, A, dtype)
            iu = np.tril_indices(m, -1)
            s = S[iu]
            hits = np.nonzero(s < eps2)[0]
            counts.append(len(hits))
            if mode == 'pairs':
                for t in hits:
                    j, l = iu[0][t], iu[1][t]
                    a, b = int(assign[tgt][j]), int(assign[tgt][l])
                    pairs.append((min(a, b), max(a, b)))
        else:
            S = dist_block(A, B, dtype)
            jj, ll = np.nonzero(S < eps2)
            counts.append(len(jj))
            if mode == 'pairs':
                for j, l in zip(jj, ll):
                    a, b = int(assign[tgt][j]), int(assign[nbr][l])
                    pairs.append((min(a, b), max(a, b)))
    print(f"tasks={len(tasks)} oracle_matches={sum(counts)}")
    if mode == 'counts':
        with open(out, 'w') as f:
            for (slot, k, tgt, nbr), cnt in zip(tasks, counts):
                f.write(f"{slot},{k},{tgt},{nbr},{cnt}\n")
    else:
        pairs.sort()
        print("oracle pairs:", len(pairs))
        native = sorted(tuple(map(int, l.split())) for l in open(dumpfile))
        print("native pairs:", len(native))
        if pairs == native:
            print("PAIR MULTISET vs NATIVE DUMP: EXACT MATCH")
        else:
            sp, sn = set(map(tuple, pairs)), set(native)
            print("MISMATCH: oracle-only", len(sp - sn), "native-only", len(sn - sp))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--eps2', type=float, required=True)
    ap.add_argument('--mode', choices=['pairs', 'counts'], default='pairs')
    ap.add_argument('--dump', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--fp64', action='store_true')
    ap.add_argument('--subset', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run(a.meta, a.clusters, a.tasks, a.eps2, a.mode, a.dump, a.out, a.fp64, a.seed, a.subset)
