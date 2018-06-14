#pragma once

namespace torch { namespace onnx {

enum class OperatorExportTypes {
  ONNX, // Strict ONNX export
  ONNX_ATEN, // ONNX With ATen op everywhere
  ONNX_ATEN_FALLBACK, // ONNX export with ATen fallback
  RAW, // Raw export (no ONNX)
};
}} // namespace torch::onnx
