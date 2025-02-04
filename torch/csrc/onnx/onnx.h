#pragma once

namespace torch::onnx {

enum class OperatorExportTypes {
  ONNX, // Strict ONNX export
  ONNX_ATEN, // ONNX With ATen op everywhere
  ONNX_ATEN_FALLBACK, // ONNX export with ATen fallback
  ONNX_FALLTHROUGH, // Export supported ONNX ops. Pass through unsupported ops.
};

enum class TrainingMode {
  EVAL, // Inference mode
  PRESERVE, // Preserve model state (eval/training)
  TRAINING, // Training mode
};

constexpr auto kOnnxNodeNameAttribute = "onnx_name";

} // namespace torch::onnx
