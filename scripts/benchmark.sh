#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — Python vs Swift DeepStream detector head-to-head benchmark
#
# Runs each detector sequentially against the same RTSP camera stream for a
# fixed window, captures metrics, and emits a markdown comparison report.
#
# Usage:
#   ./scripts/benchmark.sh [OPTIONS]
#
# Options:
#   --duration N        Measurement window per detector in seconds (default: 300)
#   --warmup N          Warm-up period in seconds, excluded from stats (default: 30)
#   --detector WHICH    Which detector(s) to run: python | swift | both (default: both)
#   --concurrent        Run both detectors simultaneously (requires --detector both).
#                       Python uses stream2 (640x360), Swift uses stream1 (1920x1080).
#                       Both remain running after benchmark; data written to separate CSVs.
#   --camera URL        RTSP camera URL (default: rtsp://192.168.68.69:554/stream1)
#   --device IP         Jetson device IP (default: 10.42.0.2)
#   --report PATH       Output markdown report path (default: docs/benchmark-python-vs-swift.md)
#   --help              Show this help message
#
# Requirements:
#   - wendy CLI in PATH
#   - ssh access to root@<device> without password
#   - curl in PATH
#   - awk, bc, python3 (or python) for numeric aggregation
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DURATION=300
WARMUP=30
DETECTOR="both"
CONCURRENT=false
CAMERA_URL="rtsp://192.168.68.69:554/stream1"
DEVICE_IP="10.42.0.2"
SWIFT_METRICS_PORT=9090
PYTHON_METRICS_PORT=9092
GPU_STATS_PORT=9091
SAMPLE_INTERVAL=15   # seconds between VmRSS / nvmap samples

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_PATH="${REPO_ROOT}/docs/benchmark-python-vs-swift.md"
DATA_DIR="${REPO_ROOT}/docs/benchmark-data"

SWIFT_DIR="${REPO_ROOT}/detector-swift"
PYTHON_DIR="${REPO_ROOT}/detector"

# Wendy container app names (as reported by `wendy device apps list`)
SWIFT_APP_NAME="detector-swift"
# Python detector uses a distinct appId (sh.wendy.examples.deepstream-detector-python)
# which wendy maps to the container name "deepstream-detector-python"
PYTHON_APP_NAME="deepstream-detector-python"

# Container image names on device (from `ctr -n default images ls`)
SWIFT_IMAGE_NAME="detector-swift"
PYTHON_IMAGE_NAME="deepstream-detector-python"

# SSH shorthand — StrictHostKeyChecking disabled for lab device
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@${DEVICE_IP}"

# Wifi reconnect command for Jetson
WIFI_RECONNECT_CMD="nmcli con up 'badgers den'"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
print_help() {
    sed -n '/^# Usage:/,/^# =====/p' "$0" | grep '^#' | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)   DURATION="$2";  shift 2 ;;
        --warmup)     WARMUP="$2";    shift 2 ;;
        --detector)   DETECTOR="$2";  shift 2 ;;
        --concurrent) CONCURRENT=true; shift 1 ;;
        --camera)     CAMERA_URL="$2"; shift 2 ;;
        --device)     DEVICE_IP="$2"; SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@${DEVICE_IP}"; shift 2 ;;
        --report)     REPORT_PATH="$2"; shift 2 ;;
        --help|-h)    print_help; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TOTAL_WINDOW=$(( DURATION + WARMUP ))

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

# ---------------------------------------------------------------------------
# Ensure data directory exists
# ---------------------------------------------------------------------------
mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# Run-metadata
# ---------------------------------------------------------------------------
RUN_DATE="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
log "Benchmark start — ${RUN_DATE}"
log "Duration per detector: ${DURATION}s  |  Warmup: ${WARMUP}s"
log "Detector(s): ${DETECTOR}  |  Concurrent: ${CONCURRENT}  |  Camera: ${CAMERA_URL}"

# ---------------------------------------------------------------------------
# Utility: recover WiFi before each SSH-intensive run
# ---------------------------------------------------------------------------
recover_wifi() {
    log "Ensuring WiFi connectivity on device..."
    $SSH "${WIFI_RECONNECT_CMD}" &>/dev/null || true
    sleep 2
}

# ---------------------------------------------------------------------------
# Utility: wait for /metrics endpoint to become healthy
# ---------------------------------------------------------------------------
wait_for_metrics() {
    local label="$1"
    local port="$2"
    local max_wait="${3:-120}"
    local elapsed=0
    log "Waiting for ${label} /metrics on port ${port} to become available..."
    while ! curl -sf "http://${DEVICE_IP}:${port}/metrics" -o /dev/null; do
        sleep 5
        elapsed=$(( elapsed + 5 ))
        if (( elapsed >= max_wait )); then
            warn "${label}: /metrics not available after ${max_wait}s"
            return 1
        fi
    done
    log "${label} /metrics is responding (${elapsed}s)"
    return 0
}

# ---------------------------------------------------------------------------
# Utility: fetch a single Prometheus gauge/counter value by metric name
#          Handles label matchers with optional grep pattern.
#          Returns the last matching numeric value or "N/A"
# ---------------------------------------------------------------------------
fetch_metric() {
    local endpoint="$1"   # full URL
    local metric="$2"     # metric name (exact prefix before { or space)
    local raw result
    raw=$(curl -sf "${endpoint}" 2>/dev/null || true)
    # Filter comment lines, then match the metric name; return last match
    result=$(echo "${raw}" \
        | grep -v '^#' \
        | grep "^${metric}" \
        | awk '{print $NF}' \
        | tail -1)
    echo "${result}" | tr -d '\n'
}

# Sum all label variants of a counter (e.g. all class_ values)
fetch_metric_sum() {
    local endpoint="$1"
    local metric="$2"
    local raw result
    raw=$(curl -sf "${endpoint}" 2>/dev/null || true)
    result=$(echo "${raw}" \
        | grep -v '^#' \
        | grep "^${metric}" \
        | awk '{sum += $NF} END {print sum+0}')
    # Collapse any accidental multi-line output to a single value
    echo "${result}" | tr -d '\n' | awk '{print $1+0}'
}

# Return count of a histogram (_count suffix)
fetch_histogram_count() {
    local endpoint="$1"
    local metric_base="$2"
    fetch_metric "${endpoint}" "${metric_base}_count"
}

fetch_histogram_sum() {
    local endpoint="$1"
    local metric_base="$2"
    fetch_metric "${endpoint}" "${metric_base}_sum"
}

# ---------------------------------------------------------------------------
# Utility: sample VmRSS from device
# ---------------------------------------------------------------------------
sample_vmrss() {
    local pgrep_pattern="$1"
    $SSH "cat /proc/\$(pgrep -f '${pgrep_pattern}' | head -1)/status 2>/dev/null | grep VmRSS" \
        2>/dev/null | awk '{print $2}' || echo "0"
}

# ---------------------------------------------------------------------------
# Utility: sample nvmap iovmm — sum all user-client total sizes (in kB)
# File format: lines starting with "user" have the per-process total at field 4 (e.g. "224484K")
# We sum all user lines and strip the trailing K.
# ---------------------------------------------------------------------------
sample_nvmap() {
    $SSH "awk '/^user/{gsub(/K/,\"\",\$4); sum+=\$4} END{print sum+0}' /sys/kernel/debug/nvmap/iovmm/allocations 2>/dev/null" \
        2>/dev/null || echo "0"
}

# ---------------------------------------------------------------------------
# Utility: compute min/max/mean/last from a space-separated list of numbers
# ---------------------------------------------------------------------------
compute_stats() {
    local values=("$@")
    if (( ${#values[@]} == 0 )); then
        echo "N/A N/A N/A N/A"
        return
    fi
    # Use python3 for arithmetic (awk can lose precision; bc lacks arrays)
    python3 - "${values[@]}" <<'PYEOF'
import sys
vals = [float(x) for x in sys.argv[1:] if x and x != '0']
if not vals:
    print("N/A N/A N/A N/A")
else:
    print(f"{min(vals):.1f} {max(vals):.1f} {sum(vals)/len(vals):.1f} {vals[-1]:.1f}")
PYEOF
}

# ---------------------------------------------------------------------------
# Utility: record initial container state so we can restore it
# ---------------------------------------------------------------------------
snapshot_container_state() {
    log "Snapshotting current container state on device..."
    INITIAL_SWIFT_RUNNING=false
    INITIAL_PYTHON_RUNNING=false

    # Check via wendy device apps list (uses container names)
    local apps_list
    apps_list=$(wendy device apps list --device "${DEVICE_IP}" 2>/dev/null || echo "")

    if echo "${apps_list}" | grep -q "\"${SWIFT_APP_NAME}\""; then
        INITIAL_SWIFT_RUNNING=true
        log "Swift detector was initially RUNNING"
    fi
    if echo "${apps_list}" | grep -q "\"${PYTHON_APP_NAME}\""; then
        INITIAL_PYTHON_RUNNING=true
        log "Python detector was initially RUNNING"
    fi
}

# ---------------------------------------------------------------------------
# Utility: restore container state to what it was before the benchmark
# In concurrent mode: leave Swift running (it's the reference baseline)
# ---------------------------------------------------------------------------
restore_container_state() {
    log "Restoring initial container state..."
    if [[ "${INITIAL_SWIFT_RUNNING}" == true ]] || [[ "${CONCURRENT}" == true ]]; then
        log "Ensuring Swift detector is running..."
        wendy device apps start "${SWIFT_APP_NAME}" --device "${DEVICE_IP}" 2>/dev/null || \
            (cd "${SWIFT_DIR}" && wendy run -y --detach --device "${DEVICE_IP}" 2>&1 | tail -5) || \
            warn "Failed to restore Swift detector — manual intervention needed"
    fi
    if [[ "${INITIAL_PYTHON_RUNNING}" == false && "${CONCURRENT}" == false ]]; then
        log "Python detector was not running before benchmark; stopping it."
        wendy device apps stop "${PYTHON_APP_NAME}" --device "${DEVICE_IP}" 2>/dev/null || true
    fi
    log "State restoration complete."
}

# ---------------------------------------------------------------------------
# Utility: stop a detector by container app name (best-effort, no exit on failure)
# ---------------------------------------------------------------------------
stop_detector() {
    local app_name="$1"
    local label="$2"
    local port="${3:-${SWIFT_METRICS_PORT}}"
    log "Stopping ${label} (${app_name})..."
    wendy device apps stop "${app_name}" --device "${DEVICE_IP}" 2>/dev/null || true
    # Wait for port to clear
    local waited=0
    while curl -sf "http://${DEVICE_IP}:${port}/metrics" -o /dev/null 2>/dev/null; do
        sleep 3
        waited=$(( waited + 3 ))
        if (( waited >= 30 )); then
            warn "${label} port still open after 30s — proceeding anyway"
            break
        fi
    done
    log "${label} stopped (port clear after ${waited}s)"
}

# ---------------------------------------------------------------------------
# Utility: get container image size from device ctr
# ctr images ls columns: REF  TYPE  DIGEST  SIZE  UNIT  PLATFORMS  LABELS
# SIZE is field 4, unit (MiB/GiB) is field 5.
# ---------------------------------------------------------------------------
get_image_size() {
    local grep_term="$1"
    $SSH "ctr -n default images ls 2>/dev/null | grep '^${grep_term}:'" \
        2>/dev/null | awk '{print $4, $5}' | tail -1 || echo "N/A"
}

# ---------------------------------------------------------------------------
# Utility: get device info (Jetson model, JetPack, RAM)
# ---------------------------------------------------------------------------
get_device_info() {
    JETSON_MODEL=$($SSH "cat /proc/device-tree/model 2>/dev/null | tr -d '\0'" 2>/dev/null || echo "Unknown")
    JETPACK_VER=$($SSH "cat /etc/nv_tegra_release 2>/dev/null | head -1" 2>/dev/null || echo "Unknown")
    TOTAL_RAM_MB=$($SSH "awk '/MemTotal/{printf \"%.0f\", \$2/1024}' /proc/meminfo 2>/dev/null" 2>/dev/null || echo "Unknown")
    log "Device: ${JETSON_MODEL} | JetPack: ${JETPACK_VER} | RAM: ${TOTAL_RAM_MB} MB"
}

# ---------------------------------------------------------------------------
# Core: run_detector_benchmark <name> <pgrep_pattern> <image_grep> <metrics_port>
#   Runs the benchmark window, samples metrics, writes CSV.
#   Sets result variables: RESULT_<NAME>_*
# ---------------------------------------------------------------------------
run_detector_benchmark() {
    local name="$1"            # "python" or "swift"
    local pgrep_pattern="$2"   # pattern for pgrep on device (e.g. "python" or "/app/Detector")
    local image_grep="$3"      # grep term for ctr images (e.g. "detector" or "detector-swift")
    local metrics_port="${4:-${SWIFT_METRICS_PORT}}"

    local csv_path="${DATA_DIR}/${name}.csv"
    local metrics_url="http://${DEVICE_IP}:${metrics_port}/metrics"
    local gpu_url="http://${DEVICE_IP}:${GPU_STATS_PORT}/metrics"

    log "============================================================"
    log "Starting ${name} detector benchmark (warmup ${WARMUP}s + measure ${DURATION}s)"
    log "Metrics URL: ${metrics_url}"
    log "============================================================"

    recover_wifi

    # Record image size before measurements (match exact image name prefix)
    local image_size
    image_size=$(get_image_size "${image_grep}")
    log "Container image size (${name}): ${image_size}"

    # Wait for /metrics to come up
    local metrics_ok=true
    if ! wait_for_metrics "${name}" "${metrics_port}"; then
        metrics_ok=false
    fi

    # --- Time-to-first-detection ---
    local ttfd="N/A"
    if [[ "${metrics_ok}" == true ]]; then
        log "Measuring time-to-first-detection for ${name}..."
        local t0=$SECONDS
        local ttfd_found=false
        for _ in $(seq 1 60); do
            local det_count
            det_count=$(fetch_metric_sum "${metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
            if [[ -n "${det_count}" ]] && (( $(echo "${det_count} > 0" | bc -l 2>/dev/null || echo 0) )); then
                ttfd=$(( SECONDS - t0 ))
                log "${name} first detection at ${ttfd}s"
                ttfd_found=true
                break
            fi
            sleep 2
        done
        if [[ "${ttfd_found}" == false ]]; then
            warn "${name}: No detections observed in first 120s"
            ttfd="timeout"
        fi
    fi

    # --- Warm-up phase ---
    log "Warm-up phase: ${WARMUP}s..."
    sleep "${WARMUP}"

    # --- Baseline counters (for delta calculations) ---
    local baseline_detections baseline_frames baseline_lat_sum baseline_lat_count
    baseline_detections=$(fetch_metric_sum "${metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
    baseline_frames=$(fetch_metric_sum "${metrics_url}" "deepstream_frames_processed_total" 2>/dev/null || echo "0")
    baseline_lat_sum=$(fetch_histogram_sum "${metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")
    baseline_lat_count=$(fetch_histogram_count "${metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")

    log "Baselines — detections: ${baseline_detections}, frames: ${baseline_frames}"

    # --- Measurement loop ---
    local vmrss_samples=()
    local nvmap_samples=()
    local gpu_util_samples=()
    local gpu_power_samples=()
    local gpu_ram_samples=()
    local fps_samples=()

    # Write CSV header
    echo "timestamp,fps,vmrss_kb,nvmap_kb,gpu_util_pct,gpu_power_w,gpu_ram_mb" > "${csv_path}"

    log "Measurement window: ${DURATION}s (sampling every ${SAMPLE_INTERVAL}s)..."
    local elapsed=0
    while (( elapsed < DURATION )); do
        local ts
        ts=$(date '+%Y-%m-%dT%H:%M:%SZ')

        local fps vmrss nvmap gpu_util gpu_power gpu_ram
        fps=$(fetch_metric "${metrics_url}" "deepstream_fps" 2>/dev/null || echo "0")
        vmrss=$(sample_vmrss "${pgrep_pattern}")
        nvmap=$(sample_nvmap)
        gpu_util=$(fetch_metric "${gpu_url}" "jetson_gpu_utilization_percent" 2>/dev/null || echo "0")
        # Power: sum all rails (VDD_CPU_GPU_CV + VDD_SOC)
        gpu_power=$(fetch_metric_sum "${gpu_url}" "jetson_power_watts" 2>/dev/null || echo "0")
        gpu_ram=$(fetch_metric "${gpu_url}" "jetson_ram_used_mb" 2>/dev/null || echo "0")

        # Accumulate (skip 0 values from dead endpoints)
        [[ -n "${fps}"      && "${fps}"      != "0" ]] && fps_samples+=("${fps}")
        [[ -n "${vmrss}"    && "${vmrss}"    != "0" ]] && vmrss_samples+=("${vmrss}")
        [[ -n "${nvmap}"    && "${nvmap}"    != "0" ]] && nvmap_samples+=("${nvmap}")
        [[ -n "${gpu_util}" && "${gpu_util}" != "0" ]] && gpu_util_samples+=("${gpu_util}")
        [[ -n "${gpu_power}"                        ]] && gpu_power_samples+=("${gpu_power}")
        [[ -n "${gpu_ram}"  && "${gpu_ram}"  != "0" ]] && gpu_ram_samples+=("${gpu_ram}")

        echo "${ts},${fps},${vmrss},${nvmap},${gpu_util},${gpu_power},${gpu_ram}" >> "${csv_path}"

        log "  [+${elapsed}s] fps=${fps} vmrss=${vmrss}kB nvmap=${nvmap}kB gpu=${gpu_util}% pwr=${gpu_power}W ram=${gpu_ram}MB"

        sleep "${SAMPLE_INTERVAL}"
        elapsed=$(( elapsed + SAMPLE_INTERVAL ))
    done

    # --- End-of-window counters ---
    local final_detections final_frames final_lat_sum final_lat_count
    final_detections=$(fetch_metric_sum "${metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
    final_frames=$(fetch_metric_sum "${metrics_url}" "deepstream_frames_processed_total" 2>/dev/null || echo "0")
    final_lat_sum=$(fetch_histogram_sum "${metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")
    final_lat_count=$(fetch_histogram_count "${metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")

    # Delta detections/frames over measurement window
    local window_detections window_frames
    window_detections=$(echo "${final_detections} - ${baseline_detections}" | bc 2>/dev/null || echo "0")
    window_frames=$(echo "${final_frames} - ${baseline_frames}" | bc 2>/dev/null || echo "0")

    # Mean pipeline latency = delta_sum / delta_count
    local pipeline_latency_ms="N/A"
    local delta_lat_count
    delta_lat_count=$(echo "${final_lat_count} - ${baseline_lat_count}" | bc 2>/dev/null || echo "0")
    if (( $(echo "${delta_lat_count} > 0" | bc -l 2>/dev/null || echo 0) )); then
        local delta_lat_sum
        delta_lat_sum=$(echo "${final_lat_sum} - ${baseline_lat_sum}" | bc 2>/dev/null || echo "0")
        pipeline_latency_ms=$(echo "scale=2; ${delta_lat_sum} / ${delta_lat_count}" | bc 2>/dev/null || echo "N/A")
    fi

    # Compute aggregated stats
    local fps_stats vmrss_stats nvmap_stats gpu_util_stats gpu_power_stats gpu_ram_stats
    fps_stats=$(compute_stats "${fps_samples[@]+"${fps_samples[@]}"}")
    vmrss_stats=$(compute_stats "${vmrss_samples[@]+"${vmrss_samples[@]}"}")
    nvmap_stats=$(compute_stats "${nvmap_samples[@]+"${nvmap_samples[@]}"}")
    gpu_util_stats=$(compute_stats "${gpu_util_samples[@]+"${gpu_util_samples[@]}"}")
    gpu_power_stats=$(compute_stats "${gpu_power_samples[@]+"${gpu_power_samples[@]}"}")
    gpu_ram_stats=$(compute_stats "${gpu_ram_samples[@]+"${gpu_ram_samples[@]}"}")

    # Parse stats: "min max mean last"
    read -r fps_min fps_max fps_mean fps_last       <<< "${fps_stats}"
    read -r vmrss_min vmrss_max vmrss_mean vmrss_last <<< "${vmrss_stats}"
    read -r nvmap_min nvmap_max nvmap_mean nvmap_last  <<< "${nvmap_stats}"
    read -r gpu_util_min gpu_util_max gpu_util_mean _  <<< "${gpu_util_stats}"
    read -r gpu_pwr_min gpu_pwr_max gpu_pwr_mean _     <<< "${gpu_power_stats}"
    read -r gpu_ram_min gpu_ram_max gpu_ram_mean _      <<< "${gpu_ram_stats}"

    # Detect memory leak trend in VmRSS (simple: last - first > 10%)
    local vmrss_trend="flat"
    if [[ "${#vmrss_samples[@]}" -ge 4 ]]; then
        local vmrss_first="${vmrss_samples[0]}"
        local vmrss_latest="${vmrss_samples[-1]}"
        local drift
        drift=$(echo "scale=1; (${vmrss_latest} - ${vmrss_first}) * 100 / (${vmrss_first} + 1)" | bc 2>/dev/null || echo "0")
        if (( $(echo "${drift} > 10" | bc -l 2>/dev/null || echo 0) )); then
            vmrss_trend="RISING (+${drift}%)"
        elif (( $(echo "${drift} < -5" | bc -l 2>/dev/null || echo 0) )); then
            vmrss_trend="falling (${drift}%)"
        fi
    fi

    # Export results as environment variables for report generation
    local uname="${name^^}"
    eval "RESULT_${uname}_FPS_MEAN='${fps_mean}'"
    eval "RESULT_${uname}_FPS_MIN='${fps_min}'"
    eval "RESULT_${uname}_FPS_MAX='${fps_max}'"
    eval "RESULT_${uname}_DETECTIONS='${window_detections}'"
    eval "RESULT_${uname}_FRAMES='${window_frames}'"
    eval "RESULT_${uname}_LATENCY_MS='${pipeline_latency_ms}'"
    eval "RESULT_${uname}_VMRSS_MIN='${vmrss_min}'"
    eval "RESULT_${uname}_VMRSS_MAX='${vmrss_max}'"
    eval "RESULT_${uname}_VMRSS_MEAN='${vmrss_mean}'"
    eval "RESULT_${uname}_VMRSS_LAST='${vmrss_last}'"
    eval "RESULT_${uname}_VMRSS_TREND='${vmrss_trend}'"
    eval "RESULT_${uname}_NVMAP_MIN='${nvmap_min}'"
    eval "RESULT_${uname}_NVMAP_MAX='${nvmap_max}'"
    eval "RESULT_${uname}_NVMAP_MEAN='${nvmap_mean}'"
    eval "RESULT_${uname}_NVMAP_LAST='${nvmap_last}'"
    eval "RESULT_${uname}_GPU_UTIL_MEAN='${gpu_util_mean}'"
    eval "RESULT_${uname}_GPU_POWER_MEAN='${gpu_pwr_mean}'"
    eval "RESULT_${uname}_GPU_RAM_MEAN='${gpu_ram_mean}'"
    eval "RESULT_${uname}_IMAGE_SIZE='${image_size}'"
    eval "RESULT_${uname}_TTFD='${ttfd}'"
    eval "RESULT_${uname}_OK='true'"
    eval "RESULT_${uname}_CSV='${csv_path}'"

    log "${name} benchmark complete. Detections in window: ${window_detections} | Mean FPS: ${fps_mean}"
}

# ---------------------------------------------------------------------------
# Core: run_concurrent_benchmark
#   Both detectors running simultaneously.
#   Python on stream2 (port 9092), Swift on stream1 (port 9090).
#   Samples both at each interval, writes two CSVs.
#   nvmap is shared — one sample per tick represents the combined footprint.
#   GPU util and power are also shared — sampled once per tick.
#   Sets RESULT_CONCURRENT_* variables for report generation.
# ---------------------------------------------------------------------------
run_concurrent_benchmark() {
    local py_csv="${DATA_DIR}/concurrent-python.csv"
    local sw_csv="${DATA_DIR}/concurrent-swift.csv"
    local py_metrics_url="http://${DEVICE_IP}:${PYTHON_METRICS_PORT}/metrics"
    local sw_metrics_url="http://${DEVICE_IP}:${SWIFT_METRICS_PORT}/metrics"
    local gpu_url="http://${DEVICE_IP}:${GPU_STATS_PORT}/metrics"

    log "============================================================"
    log "CONCURRENT benchmark: Python (port ${PYTHON_METRICS_PORT}) + Swift (port ${SWIFT_METRICS_PORT})"
    log "Python stream: stream2 (640x360) | Swift stream: stream1 (1920x1080)"
    log "Warmup: ${WARMUP}s | Measure: ${DURATION}s"
    log "============================================================"

    recover_wifi

    local py_image_size sw_image_size
    py_image_size=$(get_image_size "${PYTHON_IMAGE_NAME}")
    sw_image_size=$(get_image_size "${SWIFT_IMAGE_NAME}")
    log "Python image size: ${py_image_size}"
    log "Swift image size: ${sw_image_size}"

    # Wait for both /metrics endpoints
    local py_metrics_ok=true sw_metrics_ok=true
    wait_for_metrics "Python (concurrent)" "${PYTHON_METRICS_PORT}" 300 || py_metrics_ok=false
    wait_for_metrics "Swift (concurrent)"  "${SWIFT_METRICS_PORT}"  120 || sw_metrics_ok=false

    if [[ "${py_metrics_ok}" == false && "${sw_metrics_ok}" == false ]]; then
        warn "Neither detector metrics endpoint is responding — aborting concurrent benchmark"
        RESULT_CONCURRENT_OK="false"
        RESULT_CONCURRENT_REASON="Both metrics endpoints unreachable"
        return 1
    fi

    # Time-to-first-detection for both (in parallel — use background subshells)
    local py_ttfd="N/A" sw_ttfd="N/A"
    local t0=$SECONDS

    if [[ "${py_metrics_ok}" == true ]]; then
        for _ in $(seq 1 60); do
            local det
            det=$(fetch_metric_sum "${py_metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
            if [[ -n "${det}" ]] && (( $(echo "${det} > 0" | bc -l 2>/dev/null || echo 0) )); then
                py_ttfd=$(( SECONDS - t0 ))
                log "Python (concurrent) first detection at ${py_ttfd}s"
                break
            fi
            sleep 2
        done
    fi

    if [[ "${sw_metrics_ok}" == true ]]; then
        local t1=$SECONDS
        for _ in $(seq 1 30); do
            local det
            det=$(fetch_metric_sum "${sw_metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
            if [[ -n "${det}" ]] && (( $(echo "${det} > 0" | bc -l 2>/dev/null || echo 0) )); then
                sw_ttfd=$(( SECONDS - t1 ))
                log "Swift (concurrent) first detection at ${sw_ttfd}s"
                break
            fi
            sleep 2
        done
    fi

    # Warm-up phase
    log "Concurrent warm-up: ${WARMUP}s..."
    sleep "${WARMUP}"

    # Baselines
    local py_base_det py_base_frames py_base_lat_sum py_base_lat_count
    local sw_base_det sw_base_frames sw_base_lat_sum sw_base_lat_count
    py_base_det=$(fetch_metric_sum "${py_metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
    py_base_frames=$(fetch_metric_sum "${py_metrics_url}" "deepstream_frames_processed_total" 2>/dev/null || echo "0")
    py_base_lat_sum=$(fetch_histogram_sum "${py_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")
    py_base_lat_count=$(fetch_histogram_count "${py_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")

    sw_base_det=$(fetch_metric_sum "${sw_metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
    sw_base_frames=$(fetch_metric_sum "${sw_metrics_url}" "deepstream_frames_processed_total" 2>/dev/null || echo "0")
    sw_base_lat_sum=$(fetch_histogram_sum "${sw_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")
    sw_base_lat_count=$(fetch_histogram_count "${sw_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")

    log "Concurrent baselines — py_det: ${py_base_det}, sw_det: ${sw_base_det}"

    # Sample arrays — Python
    local py_fps_samples=() py_vmrss_samples=() py_nvmap_samples=()
    # Sample arrays — Swift
    local sw_fps_samples=() sw_vmrss_samples=()
    # Shared (nvmap, GPU) — sampled once per tick
    local shared_nvmap_samples=() shared_gpu_util_samples=() shared_gpu_power_samples=() shared_gpu_ram_samples=()

    # Write CSV headers
    echo "timestamp,fps,vmrss_kb,nvmap_kb,gpu_util_pct,gpu_power_w,gpu_ram_mb" > "${py_csv}"
    echo "timestamp,fps,vmrss_kb,nvmap_kb,gpu_util_pct,gpu_power_w,gpu_ram_mb" > "${sw_csv}"

    log "Concurrent measurement window: ${DURATION}s (sampling every ${SAMPLE_INTERVAL}s)..."
    local elapsed=0
    while (( elapsed < DURATION )); do
        local ts
        ts=$(date '+%Y-%m-%dT%H:%M:%SZ')

        # Per-detector metrics
        local py_fps sw_fps py_vmrss sw_vmrss
        py_fps=$(fetch_metric "${py_metrics_url}" "deepstream_fps" 2>/dev/null || echo "0")
        sw_fps=$(fetch_metric "${sw_metrics_url}" "deepstream_fps" 2>/dev/null || echo "0")
        py_vmrss=$(sample_vmrss "detector.py")
        sw_vmrss=$(sample_vmrss "/app/Detector")

        # Shared metrics (one sample each)
        local nvmap gpu_util gpu_power gpu_ram
        nvmap=$(sample_nvmap)
        gpu_util=$(fetch_metric "${gpu_url}" "jetson_gpu_utilization_percent" 2>/dev/null || echo "0")
        gpu_power=$(fetch_metric_sum "${gpu_url}" "jetson_power_watts" 2>/dev/null || echo "0")
        gpu_ram=$(fetch_metric "${gpu_url}" "jetson_ram_used_mb" 2>/dev/null || echo "0")

        # Accumulate
        [[ -n "${py_fps}"   && "${py_fps}"   != "0" ]] && py_fps_samples+=("${py_fps}")
        [[ -n "${sw_fps}"   && "${sw_fps}"   != "0" ]] && sw_fps_samples+=("${sw_fps}")
        [[ -n "${py_vmrss}" && "${py_vmrss}" != "0" ]] && py_vmrss_samples+=("${py_vmrss}")
        [[ -n "${sw_vmrss}" && "${sw_vmrss}" != "0" ]] && sw_vmrss_samples+=("${sw_vmrss}")
        [[ -n "${nvmap}"    && "${nvmap}"    != "0" ]] && shared_nvmap_samples+=("${nvmap}")
        [[ -n "${gpu_util}" && "${gpu_util}" != "0" ]] && shared_gpu_util_samples+=("${gpu_util}")
        [[ -n "${gpu_power}"                        ]] && shared_gpu_power_samples+=("${gpu_power}")
        [[ -n "${gpu_ram}"  && "${gpu_ram}"  != "0" ]] && shared_gpu_ram_samples+=("${gpu_ram}")

        # Write CSVs (nvmap/gpu cols shared — same value in both CSVs)
        echo "${ts},${py_fps},${py_vmrss},${nvmap},${gpu_util},${gpu_power},${gpu_ram}" >> "${py_csv}"
        echo "${ts},${sw_fps},${sw_vmrss},${nvmap},${gpu_util},${gpu_power},${gpu_ram}" >> "${sw_csv}"

        log "  [+${elapsed}s] py_fps=${py_fps} sw_fps=${sw_fps} py_vmrss=${py_vmrss}kB sw_vmrss=${sw_vmrss}kB nvmap=${nvmap}kB gpu=${gpu_util}% pwr=${gpu_power}W"

        sleep "${SAMPLE_INTERVAL}"
        elapsed=$(( elapsed + SAMPLE_INTERVAL ))
    done

    # End-of-window counters
    local py_final_det py_final_frames py_final_lat_sum py_final_lat_count
    local sw_final_det sw_final_frames sw_final_lat_sum sw_final_lat_count
    py_final_det=$(fetch_metric_sum "${py_metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
    py_final_frames=$(fetch_metric_sum "${py_metrics_url}" "deepstream_frames_processed_total" 2>/dev/null || echo "0")
    py_final_lat_sum=$(fetch_histogram_sum "${py_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")
    py_final_lat_count=$(fetch_histogram_count "${py_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")

    sw_final_det=$(fetch_metric_sum "${sw_metrics_url}" "deepstream_detections_total" 2>/dev/null || echo "0")
    sw_final_frames=$(fetch_metric_sum "${sw_metrics_url}" "deepstream_frames_processed_total" 2>/dev/null || echo "0")
    sw_final_lat_sum=$(fetch_histogram_sum "${sw_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")
    sw_final_lat_count=$(fetch_histogram_count "${sw_metrics_url}" "deepstream_total_latency_ms" 2>/dev/null || echo "0")

    # Window deltas
    local py_window_det py_window_frames sw_window_det sw_window_frames
    py_window_det=$(echo "${py_final_det} - ${py_base_det}" | bc 2>/dev/null || echo "0")
    py_window_frames=$(echo "${py_final_frames} - ${py_base_frames}" | bc 2>/dev/null || echo "0")
    sw_window_det=$(echo "${sw_final_det} - ${sw_base_det}" | bc 2>/dev/null || echo "0")
    sw_window_frames=$(echo "${sw_final_frames} - ${sw_base_frames}" | bc 2>/dev/null || echo "0")

    # Pipeline latencies
    local py_lat_ms="N/A" sw_lat_ms="N/A"
    local py_delta_lat_count sw_delta_lat_count
    py_delta_lat_count=$(echo "${py_final_lat_count} - ${py_base_lat_count}" | bc 2>/dev/null || echo "0")
    sw_delta_lat_count=$(echo "${sw_final_lat_count} - ${sw_base_lat_count}" | bc 2>/dev/null || echo "0")
    if (( $(echo "${py_delta_lat_count} > 0" | bc -l 2>/dev/null || echo 0) )); then
        local py_delta_lat_sum
        py_delta_lat_sum=$(echo "${py_final_lat_sum} - ${py_base_lat_sum}" | bc 2>/dev/null || echo "0")
        py_lat_ms=$(echo "scale=2; ${py_delta_lat_sum} / ${py_delta_lat_count}" | bc 2>/dev/null || echo "N/A")
    fi
    if (( $(echo "${sw_delta_lat_count} > 0" | bc -l 2>/dev/null || echo 0) )); then
        local sw_delta_lat_sum
        sw_delta_lat_sum=$(echo "${sw_final_lat_sum} - ${sw_base_lat_sum}" | bc 2>/dev/null || echo "0")
        sw_lat_ms=$(echo "scale=2; ${sw_delta_lat_sum} / ${sw_delta_lat_count}" | bc 2>/dev/null || echo "N/A")
    fi

    # Aggregate stats
    local py_fps_stats py_vmrss_stats sw_fps_stats sw_vmrss_stats
    local nvmap_stats gpu_util_stats gpu_power_stats gpu_ram_stats
    py_fps_stats=$(compute_stats "${py_fps_samples[@]+"${py_fps_samples[@]}"}")
    py_vmrss_stats=$(compute_stats "${py_vmrss_samples[@]+"${py_vmrss_samples[@]}"}")
    sw_fps_stats=$(compute_stats "${sw_fps_samples[@]+"${sw_fps_samples[@]}"}")
    sw_vmrss_stats=$(compute_stats "${sw_vmrss_samples[@]+"${sw_vmrss_samples[@]}"}")
    nvmap_stats=$(compute_stats "${shared_nvmap_samples[@]+"${shared_nvmap_samples[@]}"}")
    gpu_util_stats=$(compute_stats "${shared_gpu_util_samples[@]+"${shared_gpu_util_samples[@]}"}")
    gpu_power_stats=$(compute_stats "${shared_gpu_power_samples[@]+"${shared_gpu_power_samples[@]}"}")
    gpu_ram_stats=$(compute_stats "${shared_gpu_ram_samples[@]+"${shared_gpu_ram_samples[@]}"}")

    read -r py_fps_min py_fps_max py_fps_mean _ <<< "${py_fps_stats}"
    read -r py_vmrss_min py_vmrss_max py_vmrss_mean py_vmrss_last <<< "${py_vmrss_stats}"
    read -r sw_fps_min sw_fps_max sw_fps_mean _ <<< "${sw_fps_stats}"
    read -r sw_vmrss_min sw_vmrss_max sw_vmrss_mean sw_vmrss_last <<< "${sw_vmrss_stats}"
    read -r nvmap_min nvmap_max nvmap_mean nvmap_last <<< "${nvmap_stats}"
    read -r gpu_util_min gpu_util_max gpu_util_mean _ <<< "${gpu_util_stats}"
    read -r gpu_pwr_min gpu_pwr_max gpu_pwr_mean _ <<< "${gpu_power_stats}"
    read -r gpu_ram_min gpu_ram_max gpu_ram_mean _ <<< "${gpu_ram_stats}"

    # nvmap drift check (critical — if rising, Python may have old leak pattern)
    local nvmap_trend="flat"
    if [[ "${#shared_nvmap_samples[@]}" -ge 4 ]]; then
        local nvmap_first="${shared_nvmap_samples[0]}"
        local nvmap_latest="${shared_nvmap_samples[-1]}"
        local nvmap_drift
        nvmap_drift=$(echo "scale=1; (${nvmap_latest} - ${nvmap_first}) * 100 / (${nvmap_first} + 1)" | bc 2>/dev/null || echo "0")
        if (( $(echo "${nvmap_drift} > 5" | bc -l 2>/dev/null || echo 0) )); then
            nvmap_trend="RISING (+${nvmap_drift}%) — potential leak in one or both detectors"
        elif (( $(echo "${nvmap_drift} < -2" | bc -l 2>/dev/null || echo 0) )); then
            nvmap_trend="falling (${nvmap_drift}%)"
        fi
    fi

    # VmRSS trend for each
    local py_vmrss_trend="flat" sw_vmrss_trend="flat"
    if [[ "${#py_vmrss_samples[@]}" -ge 4 ]]; then
        local drift
        drift=$(echo "scale=1; (${py_vmrss_samples[-1]} - ${py_vmrss_samples[0]}) * 100 / (${py_vmrss_samples[0]} + 1)" | bc 2>/dev/null || echo "0")
        (( $(echo "${drift} > 10" | bc -l 2>/dev/null || echo 0) )) && py_vmrss_trend="RISING (+${drift}%)"
    fi
    if [[ "${#sw_vmrss_samples[@]}" -ge 4 ]]; then
        local drift
        drift=$(echo "scale=1; (${sw_vmrss_samples[-1]} - ${sw_vmrss_samples[0]}) * 100 / (${sw_vmrss_samples[0]} + 1)" | bc 2>/dev/null || echo "0")
        (( $(echo "${drift} > 10" | bc -l 2>/dev/null || echo 0) )) && sw_vmrss_trend="RISING (+${drift}%)"
    fi

    # Export concurrent result vars
    RESULT_CONCURRENT_OK="true"
    RESULT_CONCURRENT_PY_FPS_MEAN="${py_fps_mean}"
    RESULT_CONCURRENT_PY_FPS_MIN="${py_fps_min}"
    RESULT_CONCURRENT_PY_FPS_MAX="${py_fps_max}"
    RESULT_CONCURRENT_PY_DET="${py_window_det}"
    RESULT_CONCURRENT_PY_LAT="${py_lat_ms}"
    RESULT_CONCURRENT_PY_VMRSS_MEAN="${py_vmrss_mean}"
    RESULT_CONCURRENT_PY_VMRSS_TREND="${py_vmrss_trend}"
    RESULT_CONCURRENT_PY_TTFD="${py_ttfd}"
    RESULT_CONCURRENT_PY_IMAGE="${py_image_size}"
    RESULT_CONCURRENT_PY_CSV="${py_csv}"

    RESULT_CONCURRENT_SW_FPS_MEAN="${sw_fps_mean}"
    RESULT_CONCURRENT_SW_FPS_MIN="${sw_fps_min}"
    RESULT_CONCURRENT_SW_FPS_MAX="${sw_fps_max}"
    RESULT_CONCURRENT_SW_DET="${sw_window_det}"
    RESULT_CONCURRENT_SW_LAT="${sw_lat_ms}"
    RESULT_CONCURRENT_SW_VMRSS_MEAN="${sw_vmrss_mean}"
    RESULT_CONCURRENT_SW_VMRSS_TREND="${sw_vmrss_trend}"
    RESULT_CONCURRENT_SW_TTFD="${sw_ttfd}"
    RESULT_CONCURRENT_SW_IMAGE="${sw_image_size}"
    RESULT_CONCURRENT_SW_CSV="${sw_csv}"

    RESULT_CONCURRENT_NVMAP_MIN="${nvmap_min}"
    RESULT_CONCURRENT_NVMAP_MAX="${nvmap_max}"
    RESULT_CONCURRENT_NVMAP_MEAN="${nvmap_mean}"
    RESULT_CONCURRENT_NVMAP_LAST="${nvmap_last}"
    RESULT_CONCURRENT_NVMAP_TREND="${nvmap_trend}"
    RESULT_CONCURRENT_GPU_UTIL="${gpu_util_mean}"
    RESULT_CONCURRENT_GPU_POWER="${gpu_pwr_mean}"
    RESULT_CONCURRENT_GPU_RAM="${gpu_ram_mean}"

    log "Concurrent benchmark complete."
    log "  Python: det=${py_window_det} fps=${py_fps_mean}  Swift: det=${sw_window_det} fps=${sw_fps_mean}"
    log "  nvmap trend: ${nvmap_trend}"
}

# ---------------------------------------------------------------------------
# Utility: mark a detector as NOT RUN with a reason
# ---------------------------------------------------------------------------
mark_not_run() {
    local name="$1"
    local reason="$2"
    local uname="${name^^}"
    eval "RESULT_${uname}_OK='false'"
    eval "RESULT_${uname}_REASON='${reason}'"
    warn "${name} detector NOT RUN: ${reason}"
}

# ---------------------------------------------------------------------------
# Utility: determine winner for a metric (lower or higher)
# ---------------------------------------------------------------------------
winner_higher() {
    local py="$1" sw="$2"
    if [[ "${py}" == "N/A" || "${sw}" == "N/A" ]]; then echo "N/A"; return; fi
    if (( $(echo "${sw} > ${py}" | bc -l 2>/dev/null || echo 0) )); then echo "Swift"
    elif (( $(echo "${py} > ${sw}" | bc -l 2>/dev/null || echo 0) )); then echo "Python"
    else echo "Tie"; fi
}

winner_lower() {
    local py="$1" sw="$2"
    if [[ "${py}" == "N/A" || "${sw}" == "N/A" ]]; then echo "N/A"; return; fi
    if (( $(echo "${sw} < ${py}" | bc -l 2>/dev/null || echo 0) )); then echo "Swift"
    elif (( $(echo "${py} < ${sw}" | bc -l 2>/dev/null || echo 0) )); then echo "Python"
    else echo "Tie"; fi
}

# ---------------------------------------------------------------------------
# Utility: compute detection parity note
# ---------------------------------------------------------------------------
detection_parity_note() {
    local py="$1" sw="$2"
    if [[ "${py}" == "N/A" || "${sw}" == "N/A" || "${py}" == "0" || "${sw}" == "0" ]]; then
        echo "Cannot compare (one or both detectors returned no data)"
        return
    fi
    local diff pct model_note
    diff=$(echo "${sw} - ${py}" | bc 2>/dev/null || echo "0")
    # pct = |diff| / max(py,sw) * 100
    local denom
    denom=$(echo "${py} ${sw}" | awk '{print ($1>$2)?$1:$2}')
    pct=$(echo "scale=1; if (${diff} < 0) -${diff}*100/${denom} else ${diff}*100/${denom}" | bc 2>/dev/null || echo "N/A")
    # Different models (yolo11n vs yolo26n) so flag that
    model_note="Note: Python uses yolo11n, Swift uses yolo26n — models differ, so detection parity is informational only."
    echo "${pct}% difference (Python=${py}, Swift=${sw}). ${model_note}"
}

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
generate_report() {
    log "Generating markdown report at ${REPORT_PATH}..."

    local py_ok="${RESULT_PYTHON_OK:-false}"
    local sw_ok="${RESULT_SWIFT_OK:-false}"
    local concurrent_ok="${RESULT_CONCURRENT_OK:-false}"

    # Pull values (default to N/A if not set)
    local py_fps_mean="${RESULT_PYTHON_FPS_MEAN:-N/A}"
    local py_fps_min="${RESULT_PYTHON_FPS_MIN:-N/A}"
    local py_fps_max="${RESULT_PYTHON_FPS_MAX:-N/A}"
    local py_det="${RESULT_PYTHON_DETECTIONS:-N/A}"
    local py_lat="${RESULT_PYTHON_LATENCY_MS:-N/A}"
    local py_vmrss_min="${RESULT_PYTHON_VMRSS_MIN:-N/A}"
    local py_vmrss_max="${RESULT_PYTHON_VMRSS_MAX:-N/A}"
    local py_vmrss_mean="${RESULT_PYTHON_VMRSS_MEAN:-N/A}"
    local py_vmrss_last="${RESULT_PYTHON_VMRSS_LAST:-N/A}"
    local py_vmrss_trend="${RESULT_PYTHON_VMRSS_TREND:-N/A}"
    local py_nvmap_min="${RESULT_PYTHON_NVMAP_MIN:-N/A}"
    local py_nvmap_max="${RESULT_PYTHON_NVMAP_MAX:-N/A}"
    local py_nvmap_mean="${RESULT_PYTHON_NVMAP_MEAN:-N/A}"
    local py_nvmap_last="${RESULT_PYTHON_NVMAP_LAST:-N/A}"
    local py_gpu_util="${RESULT_PYTHON_GPU_UTIL_MEAN:-N/A}"
    local py_gpu_pwr="${RESULT_PYTHON_GPU_POWER_MEAN:-N/A}"
    local py_gpu_ram="${RESULT_PYTHON_GPU_RAM_MEAN:-N/A}"
    local py_imgsize="${RESULT_PYTHON_IMAGE_SIZE:-N/A}"
    local py_ttfd="${RESULT_PYTHON_TTFD:-N/A}"
    local py_reason="${RESULT_PYTHON_REASON:-}"
    local py_csv="${RESULT_PYTHON_CSV:-N/A}"

    local sw_fps_mean="${RESULT_SWIFT_FPS_MEAN:-N/A}"
    local sw_fps_min="${RESULT_SWIFT_FPS_MIN:-N/A}"
    local sw_fps_max="${RESULT_SWIFT_FPS_MAX:-N/A}"
    local sw_det="${RESULT_SWIFT_DETECTIONS:-N/A}"
    local sw_lat="${RESULT_SWIFT_LATENCY_MS:-N/A}"
    local sw_vmrss_min="${RESULT_SWIFT_VMRSS_MIN:-N/A}"
    local sw_vmrss_max="${RESULT_SWIFT_VMRSS_MAX:-N/A}"
    local sw_vmrss_mean="${RESULT_SWIFT_VMRSS_MEAN:-N/A}"
    local sw_vmrss_last="${RESULT_SWIFT_VMRSS_LAST:-N/A}"
    local sw_vmrss_trend="${RESULT_SWIFT_VMRSS_TREND:-N/A}"
    local sw_nvmap_min="${RESULT_SWIFT_NVMAP_MIN:-N/A}"
    local sw_nvmap_max="${RESULT_SWIFT_NVMAP_MAX:-N/A}"
    local sw_nvmap_mean="${RESULT_SWIFT_NVMAP_MEAN:-N/A}"
    local sw_nvmap_last="${RESULT_SWIFT_NVMAP_LAST:-N/A}"
    local sw_gpu_util="${RESULT_SWIFT_GPU_UTIL_MEAN:-N/A}"
    local sw_gpu_pwr="${RESULT_SWIFT_GPU_POWER_MEAN:-N/A}"
    local sw_gpu_ram="${RESULT_SWIFT_GPU_RAM_MEAN:-N/A}"
    local sw_imgsize="${RESULT_SWIFT_IMAGE_SIZE:-N/A}"
    local sw_ttfd="${RESULT_SWIFT_TTFD:-N/A}"
    local sw_csv="${RESULT_SWIFT_CSV:-N/A}"

    # Concurrent results
    local con_py_fps="${RESULT_CONCURRENT_PY_FPS_MEAN:-N/A}"
    local con_py_fps_min="${RESULT_CONCURRENT_PY_FPS_MIN:-N/A}"
    local con_py_fps_max="${RESULT_CONCURRENT_PY_FPS_MAX:-N/A}"
    local con_py_det="${RESULT_CONCURRENT_PY_DET:-N/A}"
    local con_py_lat="${RESULT_CONCURRENT_PY_LAT:-N/A}"
    local con_py_vmrss="${RESULT_CONCURRENT_PY_VMRSS_MEAN:-N/A}"
    local con_py_vmrss_trend="${RESULT_CONCURRENT_PY_VMRSS_TREND:-N/A}"
    local con_py_ttfd="${RESULT_CONCURRENT_PY_TTFD:-N/A}"
    local con_py_img="${RESULT_CONCURRENT_PY_IMAGE:-N/A}"
    local con_py_csv="${RESULT_CONCURRENT_PY_CSV:-N/A}"

    local con_sw_fps="${RESULT_CONCURRENT_SW_FPS_MEAN:-N/A}"
    local con_sw_fps_min="${RESULT_CONCURRENT_SW_FPS_MIN:-N/A}"
    local con_sw_fps_max="${RESULT_CONCURRENT_SW_FPS_MAX:-N/A}"
    local con_sw_det="${RESULT_CONCURRENT_SW_DET:-N/A}"
    local con_sw_lat="${RESULT_CONCURRENT_SW_LAT:-N/A}"
    local con_sw_vmrss="${RESULT_CONCURRENT_SW_VMRSS_MEAN:-N/A}"
    local con_sw_vmrss_trend="${RESULT_CONCURRENT_SW_VMRSS_TREND:-N/A}"
    local con_sw_ttfd="${RESULT_CONCURRENT_SW_TTFD:-N/A}"
    local con_sw_img="${RESULT_CONCURRENT_SW_IMAGE:-N/A}"
    local con_sw_csv="${RESULT_CONCURRENT_SW_CSV:-N/A}"

    local con_nvmap_min="${RESULT_CONCURRENT_NVMAP_MIN:-N/A}"
    local con_nvmap_max="${RESULT_CONCURRENT_NVMAP_MAX:-N/A}"
    local con_nvmap_mean="${RESULT_CONCURRENT_NVMAP_MEAN:-N/A}"
    local con_nvmap_last="${RESULT_CONCURRENT_NVMAP_LAST:-N/A}"
    local con_nvmap_trend="${RESULT_CONCURRENT_NVMAP_TREND:-N/A}"
    local con_gpu_util="${RESULT_CONCURRENT_GPU_UTIL:-N/A}"
    local con_gpu_power="${RESULT_CONCURRENT_GPU_POWER:-N/A}"
    local con_gpu_ram="${RESULT_CONCURRENT_GPU_RAM:-N/A}"
    local con_reason="${RESULT_CONCURRENT_REASON:-}"

    # Compute winners (sequential runs)
    local fps_winner; fps_winner=$(winner_higher "${py_fps_mean}" "${sw_fps_mean}")
    local ttfd_winner; ttfd_winner=$(winner_lower "${py_ttfd}" "${sw_ttfd}")
    local lat_winner; lat_winner=$(winner_lower "${py_lat}" "${sw_lat}")
    local vmrss_winner; vmrss_winner=$(winner_lower "${py_vmrss_mean}" "${sw_vmrss_mean}")
    local nvmap_winner; nvmap_winner=$(winner_lower "${py_nvmap_mean}" "${sw_nvmap_mean}")
    local gpu_util_winner; gpu_util_winner=$(winner_lower "${py_gpu_util}" "${sw_gpu_util}")
    local gpu_pwr_winner; gpu_pwr_winner=$(winner_lower "${py_gpu_pwr}" "${sw_gpu_pwr}")
    local gpu_ram_winner; gpu_ram_winner=$(winner_lower "${py_gpu_ram}" "${sw_gpu_ram}")

    local parity_note
    parity_note=$(detection_parity_note "${py_det}" "${sw_det}")

    # Python detector status note — only show if python was attempted or scheduled
    local py_status_note=""
    if [[ "${py_ok}" == "false" && "${py_reason}" != "not scheduled" ]]; then
        py_status_note="

> **Python detector not runnable at benchmark time.**
> Reason: ${py_reason}
> All Python columns below show N/A.
"
    elif [[ "${py_reason}" == "not scheduled" && "${DETECTOR}" != "python" && "${DETECTOR}" != "both" ]]; then
        py_status_note="

> **Python detector not included in this run** (use \`--detector both\` or \`--detector python\` to include it).
> All Python columns show N/A.
"
    fi

    # Concurrent section
    local concurrent_section=""
    if [[ "${concurrent_ok}" == "true" ]]; then
        concurrent_section="

---

## Run 2 — Concurrent (Python sub-stream + Swift main stream)

> **IMPORTANT CAVEAT:** Python runs on stream2 (640x360 sub-stream); Swift runs on
> stream1 (1920x1080 main stream). This is **not a pixel-for-pixel race**. The sub-stream
> delivers lower per-frame inference cost to the Python detector. The goal of this run
> is to measure coexistence — whether both can run simultaneously without resource
> starvation, port conflicts, or nvmap leaks — not to declare a winner on raw FPS.

**Run date:** ${RUN_DATE}
**Python stream:** stream2 (640x360), port ${PYTHON_METRICS_PORT}
**Swift stream:** stream1 (1920x1080), port ${SWIFT_METRICS_PORT}
**Measurement window:** ${DURATION}s (warmup: ${WARMUP}s excluded)
**Both detectors left running after benchmark** (Swift is reference baseline; detach mode)

### Concurrent Throughput

| Metric | Python (stream2) | Swift (stream1) | Notes |
|---|---|---|---|
| FPS (mean) | ${con_py_fps} | ${con_sw_fps} | See caveat: different source resolutions |
| FPS (min) | ${con_py_fps_min} | ${con_sw_fps_min} | |
| FPS (max) | ${con_py_fps_max} | ${con_sw_fps_max} | |
| Total detections (window) | ${con_py_det} | ${con_sw_det} | Different models + streams — not comparable |
| Time to first detection | ${con_py_ttfd}s | ${con_sw_ttfd}s | Python includes TRT engine build if first run |
| Mean pipeline latency | ${con_py_lat} ms | ${con_sw_lat} ms | |

### Concurrent Per-Process Memory

| Stat | Python VmRSS (kB) | Swift VmRSS (kB) | Notes |
|---|---|---|---|
| mean | ${con_py_vmrss} | ${con_sw_vmrss} | |
| Trend | ${con_py_vmrss_trend} | ${con_sw_vmrss_trend} | >10% rise = potential leak |

### Concurrent nvmap iovmm — SHARED (kB, critical leak canary)

> nvmap is a system-wide counter. Both processes share the same GPU buffer pool.
> A rising trend here while both run indicates a leak in one or both detectors.

| Stat | Value | Notes |
|---|---|---|
| min | ${con_nvmap_min} | |
| max | ${con_nvmap_max} | |
| mean | ${con_nvmap_mean} | |
| last | ${con_nvmap_last} | |
| Trend | ${con_nvmap_trend} | FLAT = healthy; RISING = investigate |

### Concurrent GPU & Power (shared, mean over window)

| Metric | Value | Notes |
|---|---|---|
| GPU utilization | ${con_gpu_util}% | Combined load of both detectors |
| Total power draw | ${con_gpu_power} W | All rails; higher than single-detector baseline |
| System RAM used | ${con_gpu_ram} MB | Jetson unified RAM |

### Container Images

| Detector | Image | Size |
|---|---|---|
| Python | \`${PYTHON_IMAGE_NAME}:latest\` | ${con_py_img} |
| Swift | \`${SWIFT_IMAGE_NAME}:latest\` | ${con_sw_img} |

### Concurrent Raw Data

| Detector | CSV |
|---|---|
| Python (concurrent) | \`${con_py_csv}\` |
| Swift (concurrent) | \`${con_sw_csv}\` |

Columns: \`timestamp, fps, vmrss_kb, nvmap_kb, gpu_util_pct, gpu_power_w, gpu_ram_mb\`
"
    elif [[ "${concurrent_ok}" == "false" && -n "${con_reason}" ]]; then
        concurrent_section="

---

## Run 2 — Concurrent Run (attempted, not completed)

> **Concurrent benchmark did not complete.**
> Reason: ${con_reason}
> See Run 1 (Swift only) results above for the baseline reference.
"
    fi

    cat > "${REPORT_PATH}" << MDEOF
# Python vs Swift DeepStream Detector Benchmark

## Run 1 — Swift Only (main stream, sequential baseline)

> This section preserves the original single-detector findings from Run 1 (Swift only).
> Swift ran on stream1 (1920x1080). Python was not yet deployed at this time.

**Run date:** ${RUN_DATE}
**Camera URL:** \`${CAMERA_URL}\`
**Measurement window:** ${DURATION}s per detector (warmup: ${WARMUP}s excluded from stats)
**Device:** ${DEVICE_IP}
${py_status_note}

### Device Info

| Property | Value |
|---|---|
| Jetson model | ${JETSON_MODEL:-Unknown} |
| JetPack version | ${JETPACK_VER:-Unknown} |
| Total RAM | ${TOTAL_RAM_MB:-Unknown} MB |

### Model Configuration

| Property | Python | Swift |
|---|---|---|
| Model file | yolo11n.onnx | yolo26n.onnx |
| Engine | model_b2_gpu0_fp16.engine (batch=2) | yolo26n_b2_fp16.engine (batch=1) |
| Classes | 80 | 80 |
| Input resolution | 640x640 | 640x640 |
| NMS | Custom YOLO lib (cluster-mode=2) | NMS-free end-to-end (cluster-mode=4) |
| Confidence threshold | 0.4 | 0.4 |

> **Warning:** The two detectors use different model architectures (YOLOv11n vs YOLOv2.6n).
> Detection count parity is informational only.

### Throughput

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| FPS (mean) | ${py_fps_mean} | ${sw_fps_mean} | ${fps_winner} | Steady-state from \`deepstream_fps\` gauge |
| FPS (min) | ${py_fps_min} | ${sw_fps_min} | — | |
| FPS (max) | ${py_fps_max} | ${sw_fps_max} | — | |
| Total detections (window) | ${py_det} | ${sw_det} | — | ${parity_note} |

### Cold-Start

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| Time to first detection | ${py_ttfd}s | ${sw_ttfd}s | ${ttfd_winner} | From task start → first non-zero \`deepstream_detections_total\` |

### Pipeline Latency

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| Mean pipeline latency | ${py_lat} ms | ${sw_lat} ms | ${lat_winner} | \`deepstream_total_latency_ms_sum / _count\` over measurement window |

### Memory — VmRSS (kB, sampled every ${SAMPLE_INTERVAL}s)

| Stat | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| min | ${py_vmrss_min} | ${sw_vmrss_min} | — | |
| max | ${py_vmrss_max} | ${sw_vmrss_max} | — | |
| mean | ${py_vmrss_mean} | ${sw_vmrss_mean} | ${vmrss_winner} | Lower = better |
| last | ${py_vmrss_last} | ${sw_vmrss_last} | — | |
| Trend | ${py_vmrss_trend} | ${sw_vmrss_trend} | — | >10% rise = potential leak |

### Memory — nvmap iovmm (kB, sampled every ${SAMPLE_INTERVAL}s)

> GPU unified memory allocations — the primary leak canary on Jetson.

| Stat | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| min | ${py_nvmap_min} | ${sw_nvmap_min} | — | |
| max | ${py_nvmap_max} | ${sw_nvmap_max} | — | |
| mean | ${py_nvmap_mean} | ${sw_nvmap_mean} | ${nvmap_winner} | Lower = better |
| last | ${py_nvmap_last} | ${sw_nvmap_last} | — | |

### GPU & Power (mean over measurement window)

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| GPU utilization | ${py_gpu_util}% | ${sw_gpu_util}% | ${gpu_util_winner} | \`jetson_gpu_utilization_percent\` |
| Total power draw | ${py_gpu_pwr} W | ${sw_gpu_pwr} W | ${gpu_pwr_winner} | Sum of all rails from \`jetson_power_watts\` |
| System RAM used | ${py_gpu_ram} MB | ${sw_gpu_ram} MB | ${gpu_ram_winner} | \`jetson_ram_used_mb\` (Jetson unified RAM) |

### Container Image

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| Image size (ctr) | ${py_imgsize} | ${sw_imgsize} | — | Compressed, from \`ctr -n default images ls\` |

### Raw Data

| Detector | CSV |
|---|---|
| Python | \`${py_csv}\` |
| Swift | \`${sw_csv}\` |

Columns: \`timestamp, fps, vmrss_kb, nvmap_kb, gpu_util_pct, gpu_power_w, gpu_ram_mb\`
${concurrent_section}

---

## What This Measures — And What It Doesn't

### What it measures
- **Steady-state throughput** (FPS) under real RTSP load from a live camera.
- **Per-process RSS** as a proxy for CPU-side memory growth. A rising trend
  over the 5-minute window is an early leak indicator but not conclusive.
- **nvmap iovmm allocations** — Jetson-specific GPU/iommu mappings. A
  monotonically increasing value here strongly suggests a GStreamer or
  DeepStream buffer leak.
- **End-to-end pipeline latency** from the histogram exposed by each detector.
  Includes RTSP jitter, decode, pre-process, inference, and post-process.
- **Power draw** across all voltage rails — useful for real-world deployment
  budgeting.
- **Concurrent coexistence** (Run 2): both detectors running simultaneously,
  with shared nvmap monitored as the primary leak canary.

### What it does NOT measure
- **Detection accuracy / mAP.** FPS and detection count are throughput proxies,
  not quality metrics. The two detectors use *different model architectures*
  (YOLOv11n vs YOLOv2.6n), so count parity is informational.
- **Long-term memory stability.** 5 minutes is too short to catch slow leaks
  (e.g., one that grows at 1 MB/min would only be 300 MB — within normal
  variance). Re-run with \`--duration 3600\` for a meaningful leak test.
- **Multi-stream scalability.** Each detector is configured for a single RTSP
  stream in this benchmark.
- **RTSP session variability.** The camera at \`${CAMERA_URL}\` delivers
  packet loss or WiFi jitter that affects both detectors, but not identically.
- **TensorRT engine build time.** If the Python engine is not cached in the
  persist volume, the first run triggers an 8+ minute build.
- **VLM sidecar interaction.** VLM is disabled in this benchmark.

## Follow-up Runs

1. **Extended leak test** — \`--duration 3600 --warmup 60 --concurrent\` to observe
   nvmap and VmRSS over 1 hour with both detectors running. A 5-minute window
   is too short to catch slow leaks.
2. **Model parity** — export the same ONNX model into both detectors and re-run;
   removes architecture variable from detection count comparison.
3. **Multi-stream** — update \`streams.json\` with 2–4 streams and benchmark
   FPS degradation curve per detector.
4. **Engine build time** — add a separate phase that forces engine rebuild
   (\`--force-rebuild\`) and records wall-clock build time.
5. **VLM co-location** — enable VLM sidecar and observe power, RAM, and FPS
   delta with the Python detector.

MDEOF

    log "Report written to ${REPORT_PATH}"
}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

# Record what was running before we touch anything
snapshot_container_state

# Gather device info
get_device_info

# Trap: restore state on exit (including signals)
trap restore_container_state EXIT INT TERM

# Initialize result variables
RESULT_PYTHON_OK="false"
RESULT_PYTHON_REASON="not scheduled"
RESULT_SWIFT_OK="false"
RESULT_SWIFT_REASON="not scheduled"
RESULT_CONCURRENT_OK="false"

# ---------------------------------------------------------------------------
# Concurrent mode: both detectors running simultaneously
# ---------------------------------------------------------------------------
if [[ "${CONCURRENT}" == true && "${DETECTOR}" == "both" ]]; then
    log "========== CONCURRENT MODE =========="
    log "Swift stays on port ${SWIFT_METRICS_PORT} (stream1)"
    log "Python runs on port ${PYTHON_METRICS_PORT} (stream2)"

    recover_wifi

    # Ensure Swift is running (it's the reference baseline)
    log "Ensuring Swift detector is running..."
    if ! wendy device apps start "${SWIFT_APP_NAME}" --device "${DEVICE_IP}" 2>/dev/null; then
        (cd "${SWIFT_DIR}" && wendy run -y --detach --device "${DEVICE_IP}" 2>&1 | tail -5) || \
            warn "Swift start returned non-zero (may already be running)"
    fi
    sleep 10

    # Deploy Python detector if not already running
    log "Checking Python detector (${PYTHON_APP_NAME}) on device..."
    local_py_image=$($SSH "ctr -n default images ls 2>/dev/null | grep '^${PYTHON_IMAGE_NAME}:'" 2>/dev/null || echo "")
    if [[ -z "${local_py_image}" ]]; then
        log "Python image not found on device — deploying via wendy run..."
        log "NOTE: First deploy will build TRT engine (~8-10 min). The persist volume"
        log "      (/app/engines) caches it for subsequent runs."
        DEPLOY_OUT=$(cd "${PYTHON_DIR}" && timeout 600 wendy run -y --detach --device "${DEVICE_IP}" 2>&1) || DEPLOY_OUT="timeout or error"
        if echo "${DEPLOY_OUT}" | grep -qi "error\|failed\|cannot"; then
            RESULT_CONCURRENT_OK="false"
            RESULT_CONCURRENT_REASON="Python deploy failed: $(echo "${DEPLOY_OUT}" | tail -3 | tr '\n' ' ')"
            log "Python deploy log: ${DEPLOY_OUT}"
        else
            log "Python deployed. Waiting for startup (TRT build may take up to 10 min)..."
            sleep 30
        fi
    else
        log "Python image found: ${local_py_image}"
        # Start the container
        if ! wendy device apps start "${PYTHON_APP_NAME}" --device "${DEVICE_IP}" 2>/dev/null; then
            (cd "${PYTHON_DIR}" && wendy run -y --detach --device "${DEVICE_IP}" 2>&1 | tail -5) || \
                warn "Python start returned non-zero (may already be running)"
        fi
        sleep 15
    fi

    if [[ "${RESULT_CONCURRENT_OK}" != "false" ]]; then
        run_concurrent_benchmark
    fi

elif [[ "${CONCURRENT}" == true && "${DETECTOR}" != "both" ]]; then
    warn "--concurrent requires --detector both; ignoring --concurrent flag"
    CONCURRENT=false
fi

# ---------------------------------------------------------------------------
# Sequential Python detector run
# ---------------------------------------------------------------------------
if [[ "${CONCURRENT}" == false && ("${DETECTOR}" == "python" || "${DETECTOR}" == "both") ]]; then
    log "---------- PYTHON DETECTOR PHASE (sequential) ----------"

    # Stop Swift detector first (camera only allows one session per stream URL)
    stop_detector "${SWIFT_APP_NAME}" "Swift detector" "${SWIFT_METRICS_PORT}"

    log "Checking Python detector image availability..."
    recover_wifi

    PY_IMAGE_CHECK=$($SSH "ctr -n default images ls 2>/dev/null | grep '^${PYTHON_IMAGE_NAME}:'" 2>/dev/null || echo "")
    if [[ -z "${PY_IMAGE_CHECK}" ]]; then
        log "Python image '${PYTHON_IMAGE_NAME}:latest' not found on device — attempting fresh deploy..."
        DEPLOY_LOG=$(cd "${PYTHON_DIR}" && timeout 600 wendy run -y --detach --device "${DEVICE_IP}" 2>&1) || DEPLOY_LOG="timeout or error"
        if echo "${DEPLOY_LOG}" | grep -qi "error\|failed\|cannot\|notFound"; then
            mark_not_run "python" "Image not on device and deploy failed: $(echo "${DEPLOY_LOG}" | tail -3 | tr '\n' ' ')"
        else
            log "Python deploy initiated. Waiting for container to start..."
            sleep 20
        fi
    else
        log "Python detector image found on device: ${PY_IMAGE_CHECK}"
        if ! wendy device apps start "${PYTHON_APP_NAME}" --device "${DEVICE_IP}" 2>/dev/null; then
            (cd "${PYTHON_DIR}" && wendy run -y --detach --device "${DEVICE_IP}" 2>&1 | tail -5) || \
                warn "Python start returned non-zero (may already be running)"
        fi
        sleep 15
    fi

    if [[ "${RESULT_PYTHON_REASON}" == "not scheduled" ]]; then
        if run_detector_benchmark "python" "detector.py" "${PYTHON_IMAGE_NAME}" "${PYTHON_METRICS_PORT}"; then
            :
        else
            mark_not_run "python" "Benchmark run function returned error"
        fi
    fi

    stop_detector "${PYTHON_APP_NAME}" "Python detector (post-benchmark)" "${PYTHON_METRICS_PORT}"
fi

# ---------------------------------------------------------------------------
# Sequential Swift detector run
# ---------------------------------------------------------------------------
if [[ "${CONCURRENT}" == false && ("${DETECTOR}" == "swift" || "${DETECTOR}" == "both") ]]; then
    log "---------- SWIFT DETECTOR PHASE (sequential) ----------"

    recover_wifi

    log "Starting Swift detector..."
    if ! wendy device apps start "${SWIFT_APP_NAME}" --device "${DEVICE_IP}" 2>/dev/null; then
        (cd "${SWIFT_DIR}" && wendy run -y --detach --device "${DEVICE_IP}" 2>&1 | tail -5) || \
            warn "Swift start returned non-zero (may already be running)"
    fi
    sleep 15

    if run_detector_benchmark "swift" "/app/Detector" "${SWIFT_IMAGE_NAME}" "${SWIFT_METRICS_PORT}"; then
        :
    else
        mark_not_run "swift" "Benchmark run function returned error"
    fi
fi

# ---------------------------------------------------------------------------
# Generate the report (runs before EXIT trap triggers restore)
# ---------------------------------------------------------------------------
generate_report

log "============================================================"
log "Benchmark complete."
log "Report: ${REPORT_PATH}"
log "Data:   ${DATA_DIR}/"
log "============================================================"
