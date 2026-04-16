# Slim runtime Dockerfile for DeepStream Python YOLO detector
# No builder stage: uses pre-compiled libnvdsinfer_custom_impl_Yolo.so (arm64)
# and the pre-existing yolo11n.onnx from the build context.
#
# TRT engine is built on first run and cached in the persist volume (/app/engines)
# nvinfer_config.txt points model-engine-file at /app/engines/model_b2_gpu0_fp16.engine
#
# Force rebuild: 2026-04-15-v2
FROM ubuntu:24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies:
# - Build tools for compiling Python extensions
# - Python 3.10 and development headers (required for pyds)
# - GObject/GStreamer development libraries (for building PyGObject)
# - GStreamer GI typelibs for introspection (runtime introspection)
# - GStreamer runtime plugins (for RTSP, codecs, etc.)
# - libjsoncpp25 (required by DeepStream nvmultiurisrcbin plugin)
#
# Note: DeepStream libraries and CUDA libraries are mounted from the host via CDI (/etc/cdi/nvidia.yaml):
# - /opt/nvidia/deepstream/deepstream-7.1/lib -> DeepStream libraries
# - /usr/lib/gstreamer-1.0/deepstream -> DeepStream GStreamer plugins
# - /usr/local/cuda-12.6/lib -> CUDA runtime libraries
# Add deadsnakes PPA manually (avoids network issues with add-apt-repository)
RUN apt-get update && apt-get install -y \
    build-essential \
    gnupg \
    ca-certificates \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu noble main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libjsoncpp25 \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.10 virtual environment WITHOUT system packages
# We'll install PyGObject directly in the venv to avoid version mismatches
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyGObject 3.46.0 and pycairo in venv (required for GStreamer bindings)
# Using 3.46.0 specifically as it's the last version that works well with girepository-1.0
RUN pip install --no-cache-dir \
    pycairo>=1.20.0 \
    PyGObject==3.46.0

# Install Python packages in venv
RUN pip install --no-cache-dir \
    flask>=2.0.0 \
    prometheus-client>=0.16.0 \
    'numpy>=1.21.0,<2.0' \
    Pillow>=9.0.0 \
    requests>=2.28.0 \
    opencv-python-headless>=4.5.0 \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-http

# Install DeepStream Python bindings (pyds)
RUN pip install --no-cache-dir \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# Set up working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/engines /app/lib

# Copy pre-compiled custom YOLO library (arm64, built from DeepStream-Yolo for DS 7.1 + CUDA 12.6)
COPY libnvdsinfer_custom_impl_Yolo.so /app/lib/

# Custom YOLO26 bbox parser — shared with the Swift detector so both run the
# same engine and parser for a fair comparison. See detector-swift/scripts/
# for how this .so is cross-compiled on the host.
COPY libnvdsparsebbox_yolo26.so /app/lib/

# Copy YOLO ONNX model (yolo11n exported with static batch=2 for DeepStream-Yolo)
COPY yolo11n.onnx /app/

# Pre-built YOLO26n TensorRT engine (same file the Swift detector uses).
COPY yolo26n_b2_fp16.engine /app/yolo26n_b2_fp16.engine

# Copy labels
COPY labels.txt /app/

# Copy application files
COPY detector.py /app/
COPY vlm_client.py /app/
COPY nvinfer_config.txt /app/
COPY tracker_config.yml /app/
COPY streams.json /app/
COPY entrypoint.sh /app/

# Make scripts executable
RUN chmod +x /app/detector.py /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GST_DEBUG=1
# EGL configuration for headless GPU operation (no display server required)
ENV EGL_PLATFORM=device

# DeepStream library and plugin paths (mounted from host via CDI)
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-7.1/lib:/usr/lib/gstreamer-1.0/deepstream:/usr/local/cuda-12.6/lib:/usr/lib/aarch64-linux-gnu:/usr/lib
ENV GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0/deepstream:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/deepstream

# Metrics port -- 9092 to avoid conflict with Swift detector on 9090
ENV METRICS_PORT=9092
ENV STREAMMUX_WIDTH=1920
ENV STREAMMUX_HEIGHT=1080

# Expose Prometheus metrics port
EXPOSE 9092

# Run detector via entrypoint script that sets environment variables
CMD ["/app/entrypoint.sh"]
