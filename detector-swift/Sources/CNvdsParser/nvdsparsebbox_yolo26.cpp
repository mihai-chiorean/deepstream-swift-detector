// nvdsparsebbox_yolo26.cpp
//
// Custom DeepStream nvinfer bbox parser for YOLO26 (Ultralytics end-to-end detector).
//
// Why custom: YOLO26 uses a one-to-one detection head that emits a fixed
// [1, 300, 6] tensor — 300 detection slots, each [x1, y1, x2, y2, confidence,
// class_id] in letterboxed-input-pixel space. No NMS, no anchors, no clustering
// required. The stock DeepStream-Yolo parsers assume the YOLOv3/v4/v5/v7/v8
// layout (anchor-based or anchor-free with class-score vectors).
//
// Outputs are already in letterboxed-input coordinate space (0..640). nvinfer
// scales back to source-frame pixels for us when `maintain-aspect-ratio=1`.
//
// Build:
//   g++ -std=c++14 -Wall -Werror -shared -fPIC -O2 \
//     -I/opt/nvidia/deepstream/deepstream/sources/includes \
//     nvdsparsebbox_yolo26.cpp -o libnvdsparsebbox_yolo26.so
//
// Reference:
//   https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html
//   /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer_customparser/nvdsinfer_custombboxparser.cpp

#include <vector>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo26(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

// YOLO26 end-to-end output is a single tensor, shape [1, 300, 6].
static constexpr int kNumSlots = 300;
static constexpr int kValuesPerSlot = 6;

extern "C" bool NvDsInferParseYolo26(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        return false;
    }

    // YOLO26's ONNX export produces a single output named "output0".
    // We expect exactly one FP32 output layer.
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    if (layer.dataType != FLOAT) {
        return false;
    }

    const float* out = static_cast<const float*>(layer.buffer);
    if (out == nullptr) {
        return false;
    }

    // Sanity: layer should hold at least kNumSlots * kValuesPerSlot floats.
    size_t expected = static_cast<size_t>(kNumSlots) * kValuesPerSlot;
    size_t actual = 1;
    for (unsigned int i = 0; i < layer.inferDims.numDims; ++i) {
        actual *= layer.inferDims.d[i];
    }
    if (actual < expected) {
        return false;
    }

    objectList.reserve(kNumSlots);

    for (int i = 0; i < kNumSlots; ++i) {
        const float* slot = out + i * kValuesPerSlot;

        const float x1   = slot[0];
        const float y1   = slot[1];
        const float x2   = slot[2];
        const float y2   = slot[3];
        const float conf = slot[4];
        const int   cls  = static_cast<int>(slot[5]);

        // Confidence gate. Use per-class threshold if provided, else the
        // cluster (preCluster) threshold as a single cut-off.
        float threshold = detectionParams.perClassPreclusterThreshold.size() > static_cast<size_t>(cls)
            ? detectionParams.perClassPreclusterThreshold[cls]
            : 0.0f;
        if (conf < threshold) {
            continue;
        }

        // Skip out-of-range class IDs (pad slots sometimes emit class=-1).
        if (cls < 0 || cls >= static_cast<int>(detectionParams.numClassesConfigured)) {
            continue;
        }

        // YOLO26 may emit padding slots with invalid coordinates (e.g. zeros).
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        NvDsInferParseObjectInfo obj{};
        obj.classId = static_cast<unsigned int>(cls);
        obj.detectionConfidence = conf;
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;

        objectList.push_back(obj);
    }

    return true;
}

// nvdsinfer dlsym lookup expects this symbol.
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26);
