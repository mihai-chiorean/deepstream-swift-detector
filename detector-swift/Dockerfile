# Minimal runtime Dockerfile.
#
# The Swift Detector binary is cross-compiled on the host using the
# WendyOS SDK:
#   swift build --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 \
#               --product Detector -c release
#
# This Dockerfile only assembles the runtime image from the resulting
# aarch64 binary. DeepStream, CUDA, GStreamer NVMM plugins, AND the
# Swift runtime libs are all CDI-injected by wendy at runtime (the
# `gpu` entitlement in wendy.json triggers nvidia-ctk CDI injection,
# and a systemd-service-provided csv adds the Swift runtime libs).

# swift-detector-base at 192.168.68.70:5000 is unreachable from this dev
# host's buildx (WiFi route broken), so we use plain ubuntu:24.04 and COPY
# the aarch64 Swift runtime libs (staged from the WendyOS SDK on the host).
FROM --platform=linux/arm64 ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    kmod \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    && rm -rf /var/lib/apt/lists/*

# libcairo2 + transitive deps (libpixman, libfontconfig, libfreetype, libpng,
# libxcb, libXrender, libX11, libXext, etc.) staged from the Linux_for_Tegra
# aarch64 rootfs. Copying raw .so files is dramatically faster than apt-get
# under qemu-binfmt emulation (12+ minutes for 2 packages vs seconds here).
# Required by /usr/lib/gstreamer-1.0/deepstream/libnvdsgst_osd.so — which
# provides the nvdsosd element that draws detection bboxes on the MJPEG
# branch.
COPY cairo-stage/ /usr/lib/aarch64-linux-gnu/
RUN ldconfig

WORKDIR /app

# Pre-built aarch64 Swift binary.
COPY .build/aarch64-unknown-linux-gnu/release/Detector /app/Detector

# Swift runtime libs (aarch64) from the WendyOS SDK.
COPY swift-runtime/ /usr/lib/swift/linux/

# Config, labels, and pre-built TensorRT engine baked into image.
COPY labels.txt streams.json nvinfer_config.txt tracker_config.yml /app/
COPY yolo26n_b2_fp16.engine /app/yolo26n_b2_fp16.engine

# Custom YOLO26 bbox parser .so, cross-compiled on the host via
# swiftly's clang against the WendyOS SDK + DeepStream headers from
# ~/jetson/Linux_for_Tegra/.../deepstream-7.1/sources/includes/.
RUN mkdir -p /app/lib
COPY libnvdsparsebbox_yolo26.so /app/lib/libnvdsparsebbox_yolo26.so

# libnvdsparsebbox_yolo26.so is mounted via wendy.json --resources at
# /app/lib/ at runtime (built separately on the Jetson).

# LD_LIBRARY_PATH entries:
#   /app/lib                       — custom parser .so (libnvdsparsebbox_yolo26.so)
#   /usr/lib/swift/linux           — Swift runtime (libswiftCore etc.)
#   /usr/lib                       — CDI-injected TensorRT, CUDA, GStreamer core
#   /usr/local/cuda-12.6/lib       — CUDA device libs
#   /usr/lib/aarch64-linux-gnu     — Ubuntu system libs (libc, libstdc++ etc.)
#   /usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu
#                                  — libgstnvexifmeta.so (dependency of libgstnvjpeg.so).
#                                    Without this path, GStreamer's registry scan
#                                    fails to dlopen libgstnvjpeg.so and blacklists
#                                    nvjpegenc, breaking the MJPEG sidecar pipeline.
ENV LD_LIBRARY_PATH=/app/lib:/usr/lib/swift/linux:/usr/lib:/usr/local/cuda-12.6/lib:/usr/lib/aarch64-linux-gnu:/usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu

# Enable DeepStream latency measurement APIs.
#
# NVDS_ENABLE_LATENCY_MEASUREMENT=1 causes nvds_measure_buffer_latency() to
# populate NvDsFrameLatencyInfo (end-to-end frame latency from decoder input).
#
# NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 causes each pipeline component
# (nvv4l2decoder, nvstreammux, nvinfer, nvtracker, etc.) to attach an
# NvDsMetaCompLatency user-meta entry to the batch with its own
# in_system_timestamp / out_system_timestamp. The C shim reads these in
# wendy_extract_frame_timing() and maps them to our three Prometheus buckets:
#   deepstream_decode_latency_ms     — nvv4l2decoder
#   deepstream_preprocess_latency_ms — nvstreammux + nvinfer
#   deepstream_postprocess_latency_ms— nvtracker
ENV NVDS_ENABLE_LATENCY_MEASUREMENT=1
ENV NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1

EXPOSE 9090

CMD ["/app/Detector"]
