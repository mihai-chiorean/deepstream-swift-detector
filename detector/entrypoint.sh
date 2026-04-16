#!/bin/bash
# Set environment variables for DeepStream before starting Python

# Set GST_PLUGIN_PATH to point to the actual plugin location mounted by CDI
export GST_PLUGIN_PATH="/usr/lib/gstreamer-1.0/deepstream"
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream-7.1/lib:/usr/lib/gstreamer-1.0/deepstream:/usr/lib/aarch64-linux-gnu:/usr/lib:/usr/local/cuda-12.6/lib"
export GST_DEBUG="1"
export EGL_PLATFORM="device"
export CUDA_VER="12.6"

# Metrics port — override with METRICS_PORT env var (default 9092 to avoid
# conflict with Swift detector on 9090)
export METRICS_PORT="${METRICS_PORT:-9092}"

# Stream mux dimensions — sub-stream is 640x360
export STREAMMUX_WIDTH="${STREAMMUX_WIDTH:-640}"
export STREAMMUX_HEIGHT="${STREAMMUX_HEIGHT:-360}"

# Clear GStreamer plugin cache to force re-scan
rm -rf ~/.cache/gstreamer-1.0/ 2>/dev/null

# Start the Python application
exec /opt/venv/bin/python /app/detector.py
