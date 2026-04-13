// DetectorEngine.swift
// High-level YOLO26n detection interface backed by a TensorRT engine.
//
// Engine configuration:
//   - Pre-built engine: yolo26n_b2_fp16.engine  (FP16, batch size 2)
//   - ONNX fallback:    yolo26n.onnx
//   - Input shape:      [batch, 3, 640, 640]  Float32
//   - Output shape:     [batch, 300, 6]       Float32  (YOLO26 NMS-free)
//
// YOLO26 output format: 300 detection slots, each [x1, y1, x2, y2, conf, class_id].
// No NMS or anchor decoding required — the model outputs final detections.
//
// The actor serialises all engine state access so that concurrent callers
// cannot corrupt the TensorRT execution context. CPU-heavy work (preprocessing,
// postprocessing) runs inside the actor calls; GPU work is launched from the
// context's enqueue methods and waits for completion before returning.

import TensorRT
import Logging

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - DetectorError

/// Errors raised during engine lifecycle or inference.
enum DetectorError: Error, Sendable {
    /// Neither enginePath nor onnxPath was provided, or neither file existed.
    case noModelSource(String)
    /// The TensorRT runtime could not deserialise or build an engine.
    case engineLoadFailed(String)
    /// Engine returned an output tensor of unexpected size.
    case unexpectedOutputSize(got: Int, expected: Int)
    /// Inference failed at the TensorRT layer.
    case inferenceFailed(String)
    /// An empty batch was submitted to `detectBatch`.
    case emptyBatch
}

// MARK: - DetectorEngine

/// Actor that owns a TensorRT engine and execution context and exposes a
/// high-level detection API.
///
/// All TensorRT state is confined to this actor. Preprocessing and
/// postprocessing are stateless and cheap to call; the actor's serial executor
/// prevents concurrent use of the execution context, which is not thread-safe.
actor DetectorEngine {

    // MARK: Stored properties

    /// Loaded TensorRT engine (owns the plan memory).
    let engine: Engine

    /// Execution context bound to `engine`.
    ///
    /// `ExecutionContext` is not thread-safe; actor isolation guarantees that
    /// only one call drives it at a time.
    let context: ExecutionContext

    /// Stateless YOLO preprocessor (letterbox resize + CHW conversion).
    let preprocessor: YOLOPreprocessor

    /// Stateless YOLO postprocessor (anchor decode + per-class NMS).
    let postprocessor: YOLOPostprocessor

    /// Side length of the square model input (640 for YOLO11n).
    let modelSize: Int

    // MARK: Private properties

    private let logger: Logging.Logger

    /// Maximum batch size the engine was built for.
    private let maxBatchSize: Int

    // MARK: Init

    /// Load or build a TensorRT engine and prepare it for inference.
    ///
    /// - Parameters:
    ///   - enginePath: Path to a serialised `.engine` file. Loaded first when the
    ///     file exists.
    ///   - onnxPath: Path to an ONNX model used to build an engine when
    ///     `enginePath` is absent or points to a non-existent file. The engine is
    ///     built with FP16 precision.
    ///   - labelsPath: Path to a text file with one COCO class name per line
    ///     (80 lines for standard YOLO11n).
    ///
    /// - Throws: `DetectorError` when neither source produces a valid engine, or
    ///   when engine introspection/warmup fails.
    init(
        enginePath: String?,
        engineCachePath: String? = nil,
        onnxPath: String?,
        labelsPath: String
    ) async throws {
        self.logger = Logger(label: "DetectorEngine")
        self.modelSize = 640
        self.preprocessor = YOLOPreprocessor()

        // ------------------------------------------------------------------
        // 1. Load or build the TensorRT engine
        //
        // Resolution order:
        //   a) `enginePath`      — pre-built engine shipped in the image
        //   b) `engineCachePath` — engine saved to the persist volume after a
        //                         previous ONNX rebuild on this specific device
        //   c) `onnxPath`        — rebuild from ONNX (5-8 min on Orin Nano),
        //                         then write the result to engineCachePath for
        //                         next time.
        // ------------------------------------------------------------------

        let runtime = TensorRTRuntime()
        let loadedEngine: Engine

        // Pick the first existing pre-built engine file from (enginePath, engineCachePath).
        let resolvedEnginePath: String? = {
            if let ep = enginePath, FileManager.default.fileExists(atPath: ep) { return ep }
            if let cp = engineCachePath, FileManager.default.fileExists(atPath: cp) { return cp }
            return nil
        }()

        if let ep = resolvedEnginePath {
            // Preferred path: deserialise a pre-built plan (fast startup).
            logger.info("Loading pre-built TensorRT engine", metadata: [
                "path": "\(ep)",
            ])
            do {
                let engineData = try Data(contentsOf: URL(fileURLWithPath: ep))
                loadedEngine = try runtime.deserializeEngine(from: engineData)
            } catch {
                throw DetectorError.engineLoadFailed(
                    "Failed to deserialise engine at \(ep): \(error)"
                )
            }
        } else if let op = onnxPath {
            // Fallback: build from ONNX with FP16 precision (slow, ~minutes).
            logger.warning(
                "Engine file not found; building from ONNX (this may take several minutes)",
                metadata: ["onnxPath": "\(op)"]
            )
            let onnxURL = URL(fileURLWithPath: op)
            // TODO: Verify exact EngineBuildOptions initialiser against the
            //       installed tensorrt-swift version. The type and field names
            //       here follow the documented API surface.
            let options = EngineBuildOptions(precision: [.fp16])
            do {
                loadedEngine = try runtime.buildEngine(onnxURL: onnxURL, options: options)
            } catch {
                throw DetectorError.engineLoadFailed(
                    "Failed to build engine from ONNX at \(op): \(error)"
                )
            }
            // Persist the built engine so subsequent starts skip the ~8-minute
            // ONNX-to-TensorRT conversion. Prefer the persist-volume cache
            // path (writable) over enginePath (which may point into /app
            // that's mounted read-only from the image).
            let saveURL: URL
            if let cp = engineCachePath {
                saveURL = URL(fileURLWithPath: cp)
                // Ensure the parent directory exists.
                let parent = saveURL.deletingLastPathComponent()
                try? FileManager.default.createDirectory(
                    at: parent, withIntermediateDirectories: true
                )
            } else if let ep = enginePath {
                saveURL = URL(fileURLWithPath: ep)
            } else {
                saveURL = URL(fileURLWithPath: op).deletingPathExtension()
                    .appendingPathExtension("engine")
            }
            do {
                try loadedEngine.save(to: saveURL)
                logger.info("TensorRT engine saved", metadata: ["path": "\(saveURL.path)"])
            } catch {
                // Non-fatal: the engine works in memory even if saving fails
                // (e.g. read-only filesystem).
                logger.warning(
                    "Could not save engine to disk (will rebuild next start)",
                    metadata: ["path": "\(saveURL.path)", "error": "\(error)"]
                )
            }
        } else {
            throw DetectorError.noModelSource(
                "Provide enginePath (preferred) or onnxPath to load YOLO11n."
            )
        }

        self.engine = loadedEngine

        // ------------------------------------------------------------------
        // 2. Create an execution context
        // ------------------------------------------------------------------

        // TODO: Verify the exact method name against the installed
        //       tensorrt-swift version; it may be makeExecutionContext() or
        //       createExecutionContext().
        self.context = try engine.makeExecutionContext()

        // ------------------------------------------------------------------
        // 3. Determine max batch size from the engine
        // ------------------------------------------------------------------

        // TODO: Replace with the actual API to query the engine's max batch
        //       size. tensorrt-swift may expose this via engine.maxBatchSize,
        //       engine.profile, or a binding descriptor. Default to 2 to match
        //       the pre-built model_b2_gpu0_fp16.engine.
        self.maxBatchSize = 2

        // ------------------------------------------------------------------
        // 4. Build the postprocessor (loads labels from disk)
        // ------------------------------------------------------------------

        self.postprocessor = YOLOPostprocessor(labelsPath: labelsPath)

        logger.info("DetectorEngine ready", metadata: [
            "modelSize": "\(self.modelSize)",
            "maxBatchSize": "\(self.maxBatchSize)",
            "labels": "\(self.postprocessor.labels.count)",
        ])

        // ------------------------------------------------------------------
        // 5. Warm-up: a few dummy inferences to prime GPU kernels
        // ------------------------------------------------------------------

        try await warmup()
    }

    // MARK: - Public API

    /// Run YOLO11n inference on a single video frame.
    ///
    /// The call is fully serialised through the actor's executor, so concurrent
    /// calls will queue naturally.
    ///
    /// - Parameter frame: An RGB24 frame decoded from an RTSP stream.
    /// - Returns: Detections in original-image coordinate space (normalised 0–1).
    /// - Throws: `DetectorError` on inference failure.
    func detect(frame: Frame) async throws -> [Detection] {
        // Preprocess: letterbox → normalise → CHW float tensor.
        let (inputData, letterbox) = frame.data.withUnsafeBufferPointer { ptr in
            YOLOPreprocessor.preprocess(ptr, width: frame.width, height: frame.height)
        }

        // YOLO26 output layout: [1, 300, 6] — 300 detection slots, each
        // containing [x1, y1, x2, y2, confidence, class_id].
        let outputCount = YOLOPostprocessor.maxDetections * YOLOPostprocessor.valuesPerDetection
        var outputBuffer = [Float](repeating: 0.0, count: outputCount)

        // Run inference on the GPU via the execution context.
        try await context.enqueueF32(
            inputName:  "images",
            input:      inputData,
            outputName: "output0",
            output:     &outputBuffer
        )

        // YOLO26: no anchor decode or NMS — just confidence filter.
        var detections = outputBuffer.withUnsafeBufferPointer { ptr in
            postprocessor.process(output: ptr, batchSize: 1)
        }

        // Remap from 640x640 model space back to original image coordinates.
        YOLOPreprocessor.remapBoxes(&detections, letterbox: letterbox)

        return detections
    }

    /// Run YOLO11n inference on a batch of frames in a single GPU call.
    ///
    /// This is more efficient than calling `detect(frame:)` repeatedly when
    /// multiple camera streams need to be processed in lock-step. The batch
    /// size matches the engine's compile-time maximum (2 for the default
    /// `model_b2_gpu0_fp16.engine`).
    ///
    /// - Parameter frames: 1 to `maxBatchSize` frames. When more frames are
    ///   provided than `maxBatchSize`, the excess frames are ignored.
    ///   TODO: Add chunked execution for stream counts larger than maxBatchSize.
    /// - Returns: One `[Detection]` array per input frame, in the same order.
    /// - Throws: `DetectorError.emptyBatch` when `frames` is empty, or
    ///   `DetectorError.inferenceFailed` on GPU errors.
    func detectBatch(frames: [Frame]) async throws -> [[Detection]] {
        guard !frames.isEmpty else {
            throw DetectorError.emptyBatch
        }

        // Cap at the engine's maximum batch size.
        // TODO: For stream counts > maxBatchSize, split into chunks and run
        //       multiple inference calls, then concatenate the results.
        let batch = Array(frames.prefix(maxBatchSize))
        let batchSize = batch.count

        // ------------------------------------------------------------------
        // 1. Preprocess all frames and concatenate into a single batch tensor.
        // ------------------------------------------------------------------

        // Each preprocessed image is a flat CHW float array of size
        // 3 * modelSize * modelSize. The batch tensor layout is
        // [batchSize, 3, 640, 640] in row-major (C-contiguous) order.
        let imagePlaneSize = 3 * modelSize * modelSize
        var batchInput  = [Float](repeating: 0.0, count: batchSize * imagePlaneSize)
        var letterboxes = [LetterboxInfo]()
        letterboxes.reserveCapacity(batchSize)

        for (i, frame) in batch.enumerated() {
            let (imageData, lb) = frame.data.withUnsafeBufferPointer { ptr in
                YOLOPreprocessor.preprocess(ptr, width: frame.width, height: frame.height)
            }
            letterboxes.append(lb)

            // Write this image's CHW data into its contiguous slot in the
            // batch tensor.
            let dstOffset = i * imagePlaneSize
            batchInput.withUnsafeMutableBufferPointer { dst in
                imageData.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: dstOffset)
                        .update(from: src.baseAddress!, count: imagePlaneSize)
                }
            }
        }

        // ------------------------------------------------------------------
        // 2. Single batched inference call.
        // ------------------------------------------------------------------

        // YOLO26 output layout: [batchSize, 300, 6].
        let elementsPerImage = YOLOPostprocessor.maxDetections * YOLOPostprocessor.valuesPerDetection
        var batchOutput      = [Float](repeating: 0.0, count: batchSize * elementsPerImage)

        try await context.enqueueF32(
            inputName:  "images",
            input:      batchInput,
            outputName: "output0",
            output:     &batchOutput
        )

        // ------------------------------------------------------------------
        // 3. Split output tensor and postprocess per frame.
        // ------------------------------------------------------------------

        var results = [[Detection]]()
        results.reserveCapacity(batchSize)

        for i in 0 ..< batchSize {
            let offset      = i * elementsPerImage
            var detections  = batchOutput.withUnsafeBufferPointer { fullBuf -> [Detection] in
                let slice = UnsafeBufferPointer(
                    start: fullBuf.baseAddress!.advanced(by: offset),
                    count: elementsPerImage
                )
                return postprocessor.process(output: slice, batchSize: 1)
            }

            YOLOPreprocessor.remapBoxes(&detections, letterbox: letterboxes[i])
            results.append(detections)
        }

        return results
    }

    // MARK: - Private helpers

    /// Run a small number of dummy inferences to amortise GPU kernel
    /// compilation and CUDA stream creation before live traffic arrives.
    ///
    /// Uses a gray-filled frame (pixel value 114, the standard YOLO padding
    /// colour) at the model's native 640x640 resolution.
    private func warmup() async throws {
        logger.info("Warming up TensorRT engine (3 iterations)")

        let dummyData  = [UInt8](repeating: 114, count: modelSize * modelSize * 3)
        let dummyFrame = Frame(data: dummyData, width: modelSize, height: modelSize)

        let warmupIterations = 3
        for iteration in 1 ... warmupIterations {
            _ = try await detect(frame: dummyFrame)
            logger.debug("Warmup iteration \(iteration)/\(warmupIterations) complete")
        }

        logger.info("Warmup complete")
    }
}
