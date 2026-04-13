// YOLOPostprocessor.swift
// Parses YOLO26n output tensors — NMS-free end-to-end detection.
//
// YOLO26 output tensor shape: [1, 300, 6]
//   - 300 detection slots (zero-padded beyond actual detections)
//   - Per slot: [x1, y1, x2, y2, confidence, class_id]
//   - Coordinates are absolute pixels in model input space (0–640)
//   - No NMS required — the model's one-to-one head resolves duplicates
//
// This replaces the YOLO11 postprocessor which required:
//   - Transposed [1, 84, 8400] decode
//   - Sigmoid on class scores
//   - Per-class NMS with IoU threshold
//
// Migration note: YOLO26 removes DFL and NMS entirely. The ONNX graph
// outputs final detections directly. Export with:
//   yolo export model=yolo26n.pt format=onnx

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - Detection

/// A single object detection in model coordinate space (0–640).
struct Detection: Sendable {
    /// Top-left corner x coordinate (pixels).
    var x: Float
    /// Top-left corner y coordinate (pixels).
    var y: Float
    /// Bounding box width (pixels).
    var width: Float
    /// Bounding box height (pixels).
    var height: Float
    /// COCO class index (0-based).
    var classId: Int
    /// Detection confidence score (0–1).
    var confidence: Float
    /// Human-readable class name from labels.txt.
    var label: String
    /// Optional tracker-assigned identity; populated downstream.
    var trackId: Int?
}

// MARK: - YOLOPostprocessor

/// Parses raw YOLO26 output into filtered detections.
///
/// YOLO26's one-to-one detection head outputs exactly 300 slots per image.
/// Unused slots are zero-padded. The only postprocessing required is a
/// confidence threshold filter and coordinate format conversion.
struct YOLOPostprocessor: Sendable {
    let confidenceThreshold: Float
    let labels: [String]

    /// Number of detection slots in the YOLO26 output (fixed by the model).
    static let maxDetections = 300

    /// Number of values per detection: [x1, y1, x2, y2, confidence, class_id].
    static let valuesPerDetection = 6

    // MARK: Initialisation

    /// Creates a postprocessor by loading class labels from a text file.
    ///
    /// - Parameter labelsPath: Path to a labels.txt with one class name per line
    ///   (80 lines for COCO). Empty lines are skipped.
    init(
        labelsPath: String,
        confidenceThreshold: Float = 0.4
    ) {
        self.confidenceThreshold = confidenceThreshold

        // Load labels, tolerating a trailing newline and Windows-style \r\n.
        let raw = (try? String(contentsOfFile: labelsPath, encoding: .utf8)) ?? ""
        self.labels = raw.split(separator: "\n").map { line in
            var s = String(line)
            while s.last == "\r" { s.removeLast() }
            return s
        }
    }

    // MARK: Processing

    /// Parses the raw YOLO26 output buffer and returns confidence-filtered detections.
    ///
    /// - Parameters:
    ///   - output: Pointer to the flat Float tensor. Expected layout:
    ///     `[batchSize, 300, 6]` in row-major order. Only batch index 0 is processed.
    ///   - batchSize: Number of images in the batch (default 1).
    /// - Returns: Detections passing the confidence threshold, sorted by
    ///   confidence descending. Coordinates are converted from xyxy to xywh format.
    func process(
        output: UnsafeBufferPointer<Float>,
        batchSize: Int = 1
    ) -> [Detection] {
        let slotsPerImage = Self.maxDetections * Self.valuesPerDetection  // 1800
        guard output.count >= slotsPerImage else { return [] }

        let base = output.baseAddress!

        var detections: [Detection] = []
        detections.reserveCapacity(64)

        for slot in 0 ..< Self.maxDetections {
            let offset = slot * Self.valuesPerDetection

            let confidence = base[offset + 4]
            guard confidence >= confidenceThreshold else { continue }

            // YOLO26 outputs xyxy (top-left, bottom-right) absolute pixel coords.
            let x1 = base[offset + 0]
            let y1 = base[offset + 1]
            let x2 = base[offset + 2]
            let y2 = base[offset + 3]

            // Convert to xywh (top-left + size) for compatibility with the
            // tracker and renderer which expect this format.
            let w = x2 - x1
            let h = y2 - y1
            guard w > 0, h > 0 else { continue }

            let classId = Int(base[offset + 5])
            let label = classId < labels.count ? labels[classId] : "\(classId)"

            detections.append(Detection(
                x: x1,
                y: y1,
                width: w,
                height: h,
                classId: classId,
                confidence: confidence,
                label: label
            ))
        }

        // Sort by confidence descending for consistent output ordering.
        detections.sort { $0.confidence > $1.confidence }

        return detections
    }
}
