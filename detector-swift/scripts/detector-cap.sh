#!/usr/bin/env bash
# detector-cap.sh
#
# Apply a memory and OOM-score cap to the Swift detector container on a
# WendyOS device. Runs on the device itself (execute via `ssh root@<device>`).
#
# Why this exists:
#   wendy-agent has no way to express resource limits in wendy.json — the
#   generated OCI spec has no linux.resources.memory field. On an 8GB Jetson
#   Orin Nano, a runaway Swift process (software H.264 decode, debug build)
#   consumed 11GB of unified memory, and because containerd-shim runs with
#   oom_score_adj=-998 (inherited from containerd.service's -999), the kernel
#   killed networking first. This script caps the detector so the kernel kills
#   the detector process instead of everything else.
#
# Cgroup v2 notes:
#   - memory.max      — hard memory ceiling (kills the process)
#   - memory.swap.max — swap ceiling; without this the kernel will page out to
#                       zram before OOM-killing, defeating memory.max
#   - oom_score_adj   — raised from -998 (inherited from shim) to 300 so that
#                       user workloads are killed before system daemons
#
# The cgroup is recreated on every `ctr task start`, so this script must be
# re-run after each deploy until wendy-agent gains a resources block upstream.

set -euo pipefail

CONTAINER="${CONTAINER:-detector}"
LIMIT_BYTES="${LIMIT_BYTES:-$((5 * 1024 * 1024 * 1024))}"  # 5 GiB default
OOM_SCORE="${OOM_SCORE:-300}"

CG="/sys/fs/cgroup/system.slice/${CONTAINER}"

if [[ ! -d "$CG" ]]; then
    echo "detector-cap: cgroup $CG does not exist — is the container running?" >&2
    echo "              (ctr -n default tasks ls  →  look for '${CONTAINER}')" >&2
    exit 1
fi

# Confirm this is cgroup v2 (the files below only exist on v2).
if [[ ! -f "$CG/memory.max" || ! -f "$CG/memory.swap.max" ]]; then
    echo "detector-cap: $CG is not a cgroup v2 memory controller" >&2
    exit 1
fi

echo "detector-cap: capping '${CONTAINER}' at $((LIMIT_BYTES / 1024 / 1024)) MiB (mem + swap)"
echo "$LIMIT_BYTES" > "$CG/memory.max"
echo "$LIMIT_BYTES" > "$CG/memory.swap.max"

# Raise oom_score_adj for every process currently in the cgroup.
# cgroup.procs is refreshed every read; Swift spawns worker threads, but
# oom_score_adj applies per-task-group so a single write to the main PID is
# usually enough. We iterate to be safe.
raised=0
while read -r pid; do
    [[ -z "$pid" ]] && continue
    if [[ -w "/proc/$pid/oom_score_adj" ]]; then
        echo "$OOM_SCORE" > "/proc/$pid/oom_score_adj" || true
        raised=$((raised + 1))
    fi
done < "$CG/cgroup.procs"

echo "detector-cap: raised oom_score_adj to ${OOM_SCORE} for ${raised} process(es)"

# Print the resulting state so the operator can verify.
echo
echo "=== post-cap state ==="
printf 'memory.max       = %s\n' "$(cat "$CG/memory.max")"
printf 'memory.swap.max  = %s\n' "$(cat "$CG/memory.swap.max")"
printf 'memory.current   = %s\n' "$(cat "$CG/memory.current")"
head -1 "$CG/cgroup.procs" | while read -r pid; do
    printf 'pid %-7s oom_score_adj = %s\n' "$pid" "$(cat "/proc/$pid/oom_score_adj" 2>/dev/null || echo '?')"
done
