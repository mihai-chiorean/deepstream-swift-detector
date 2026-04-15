# Build the YOLO26 custom bbox parser as a DeepStream-loadable shared object.
#
# The DeepStream SDK headers are typically at
#   /opt/nvidia/deepstream/deepstream/sources/includes
# on a *dev* install. On our runtime target (WendyOS / JetPack 6 Orin Nano)
# only the libs are installed — the headers must be provided at build time.
#
# Two build modes:
#
#   1. On-device (if you've installed deepstream-6.x-dev / 7.x-dev on a
#      separate Jetson to compile):
#        make
#
#   2. Cross-compile in a container that has the DeepStream SDK
#      (dustynv/l4t-deepstream:r36.4.0 or similar):
#        docker run --rm -v "$PWD:/src" -w /src dustynv/l4t-deepstream:r36.4.0 make
#
# Output: libnvdsparsebbox_yolo26.so
#
# This .so is loaded by nvinfer at runtime via the `custom-lib-path` key in
# nvinfer_config.txt. It must match the target device's DeepStream ABI
# (DS 7.1 on L4T r36 in our case).

DEEPSTREAM_INCLUDES ?= /opt/nvidia/deepstream/deepstream/sources/includes
CUDA_VER            ?= 12.6

CFLAGS   := -Wall -Wextra -O2 -fPIC -std=c++14 -I$(DEEPSTREAM_INCLUDES)
LDFLAGS  := -shared -Wl,--no-undefined-version
TARGET   := libnvdsparsebbox_yolo26.so

SRCS := nvdsparsebbox_yolo26.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean
all: $(TARGET)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -f $(OBJS) $(TARGET)
