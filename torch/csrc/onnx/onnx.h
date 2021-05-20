#pragma once

namespace torch { namespace onnx {

enum class OperatorExportTypes {
  ONNX, // Strict ONNX export
  ONNX_ATEN, // ONNX With ATen op everywhere
  ONNX_ATEN_FALLBACK, // ONNX export with ATen fallback
  RAW, // Raw export (no ONNX)
  ONNX_FALLTHROUGH, // Export supported ONNX ops. Pass through unsupported ops.
};

enum class TrainingMode {
  EVAL, // Inference mode
  PRESERVE, // Preserve model state (eval/training)
  TRAINING, // Training mode
};

// we pin IR version to version 6 (12/11/2019) instead of using
// onnx::IR_VERSION. with this change, the test_operators.py will be more
// stable. only bump it when it's necessary
static const size_t IR_VERSION = 6;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static const char* PRODUCER_VERSION = "1.9";
}} // namespace torch::onnx
