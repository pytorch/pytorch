#pragma once

// @generated from /home/bowbao/pytorch/tools/onnx/templates/rules.h.in

namespace torch {
namespace onnx {
namespace diagnostic {

enum class Rule : uint32_t {
  /**
   * @brief ONNX shape inference is missing for node.
   */
  ONNXShapeInferenceIsMissingForNode,

  /**
   * @brief Missing symbolic function for custom PyTorch operator, cannot translate node to ONNX.
   */
  MissingCustomSymbolicFunction,

  /**
   * @brief Missing symbolic function for standard PyTorch operator, cannot translate node to ONNX.
   */
  MissingStandardSymbolicFunction,

  /**
   * @brief Operator is supported in newer opset version.
   */
  OperatorSupportedInNewerOpsetVersion,
};

static constexpr const char* const RuleNames [] = {
  "ONNXShapeInferenceIsMissingForNode",
  "MissingCustomSymbolicFunction",
  "MissingStandardSymbolicFunction",
  "OperatorSupportedInNewerOpsetVersion",
};

} // namespace diagnostic
} // namespace onnx
} // namespace torch
