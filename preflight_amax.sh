#!/usr/bin/env bash
# Read-only environment probe for GPU feasibility. Writes only a bounded log.
set -Eeuo pipefail
ROOT="${1:-/home/qch/DiskJoin/diskjoin_repro_pack}"
GPU_INDEX="${2:-0}"
if [[ ! -d "$ROOT" ]]; then
  echo "Project root not found: $ROOT" >&2
  exit 2
fi
free_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
min_kib=$((7 * 1024 * 1024))
if (( free_kib < min_kib )); then
  echo "STOP: free disk ${free_kib} KiB < 7 GiB safety floor" >&2
  exit 3
fi
OUT="$ROOT/workspace/gpu_feasibility_20261102"
mkdir -p "$OUT"
LOG="$OUT/preflight.log"
exec > >(tee "$LOG") 2>&1
probe() { echo; echo "==> $*"; "$@" || echo "WARN: command exited $?"; }
echo "PRE-FLIGHT $(date -Is) root=$ROOT"
probe date -Is
probe df -h / "$ROOT"
probe uname -a
probe cat /etc/os-release
probe nvidia-smi
probe nvidia-smi topo -m
probe nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
probe lscpu
if command -v numactl >/dev/null 2>&1; then probe numactl --hardware; else echo "numactl not installed"; fi
if command -v nvcc >/dev/null 2>&1; then probe nvcc --version; else echo "nvcc not installed"; fi
if command -v g++ >/dev/null 2>&1; then probe g++ --version; fi
if command -v cmake >/dev/null 2>&1; then probe cmake --version; fi
(
  cd "$ROOT"
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo; echo "==> git status --short"
    git status --short
  fi
)
GPU_SAFE=0
if command -v nvidia-smi >/dev/null 2>&1; then
  state="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ "$state" =~ ^[[:space:]]*([0-9]+),[[:space:]]*([0-9]+) ]]; then
    mem_mb="${BASH_REMATCH[1]}"; util="${BASH_REMATCH[2]}"
    if (( mem_mb < 1024 && util < 10 )); then GPU_SAFE=1; else echo "GPU $GPU_INDEX appears busy (${mem_mb} MiB, ${util}%); CUDA smoke skipped"; fi
  else echo "GPU $GPU_INDEX state unavailable; CUDA smoke skipped"; fi
fi
if (( GPU_SAFE == 1 )) && command -v python3 >/dev/null 2>&1; then
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python3 - <<'PY'
import importlib.util
for pkg in ('torch','cupy','cuvs'):
    try:
        print(pkg, 'available' if importlib.util.find_spec(pkg) is not None else 'not installed')
    except Exception as exc:
        print(pkg, 'probe error', repr(exc))
try:
    import torch
    print('torch version:', torch.__version__, 'torch CUDA build:', torch.version.cuda)
    print('torch.cuda.is_available():', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('capability:', torch.cuda.get_device_capability(0), 'device:', torch.cuda.get_device_name(0))
        with torch.cuda.device(0):
            x = torch.ones((8,), device='cuda', dtype=torch.float32)
            y = (x + 1).sum()
            torch.cuda.synchronize()
            print('REAL_TORCH_CUDA_SMOKE: PASS', y.item())
except Exception as exc:
    print('REAL_TORCH_CUDA_SMOKE: FAIL_OR_UNAVAILABLE', type(exc).__name__, str(exc)[:600])
PY
fi
probe df -h / "$ROOT"
echo "Preflight finished. Review $LOG for warnings and device selection."
