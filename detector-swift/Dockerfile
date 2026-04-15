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
    libcairo2 \
    libjsoncpp25 \
    && rm -rf /var/lib/apt/lists/*

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

EXPOSE 9090

CMD ["/app/Detector"]
