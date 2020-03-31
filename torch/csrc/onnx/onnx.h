#pragma once

namespace torch { namespace onnx {

enum class OperatorExportTypes {
  ONNX, // Strict ONNX export
  ONNX_ATEN, // ONNX With ATen op everywhere
  ONNX_ATEN_FALLBACK, // ONNX export with ATen fallback
  RAW, // Raw export (no ONNX)
};

// we pin IR version to version 6 (12/11/2019) instead of using
// onnx::IR_VERSION. with this change, the test_operators.py will be more
// stable. only bump it when it's necessary
static const size_t IR_VERSION = 6;
static const char* PRODUCER_VERSION = "1.5";
}} // namespace torch::onnx
