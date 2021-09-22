#pragma once

namespace torch {
namespace onnx {

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

// We pin IR version instead of using onnx::IR_VERSION so that the
// test_operators.py will be more stable. Only bump it when
// necessary.
static const size_t IR_VERSION = 7;
static const char* PRODUCER_VERSION = "1.10";
} // namespace onnx
} // namespace torch
