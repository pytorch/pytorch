#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <c10/util/Exception.h>

#include <limits>

namespace torch::nativert {
namespace {

bool contains(std::string_view value, std::string_view needle) {
  return value.find(needle) != std::string_view::npos;
}

bool isPointerParam(std::string_view param_type) {
  return contains(param_type, "*") || contains(param_type, "pointer");
}

bool isBoolParam(std::string_view param_type) {
  return param_type == "i1" || contains(param_type, "bool");
}

bool isInt32Param(std::string_view param_type) {
  return param_type == "i32" || param_type == "u32" ||
      contains(param_type, "int32") || contains(param_type, "uint32");
}

bool isFloat64Param(std::string_view param_type) {
  return param_type == "fp64" || param_type == "f64" ||
      contains(param_type, "float64") || contains(param_type, "double");
}

} // namespace

void* KernelInputs::store_scalar_arg(
    const c10::IValue& value,
    std::string_view param_type) {
  TORCH_CHECK(
      !isPointerParam(param_type),
      "Expected Tensor for Triton pointer parameter type ",
      param_type,
      " but got ",
      value.tagKind());

  if (value.isBool()) {
    const bool scalar = value.toBool();
    if (isInt32Param(param_type)) {
      scalar_values_.emplace_back(static_cast<int32_t>(scalar));
      return &std::get<int32_t>(scalar_values_.back());
    }
    scalar_values_.emplace_back(scalar);
    return &std::get<bool>(scalar_values_.back());
  }

  if (value.isInt()) {
    const auto scalar = value.toInt();
    if (isBoolParam(param_type)) {
      TORCH_CHECK(
          scalar == 0 || scalar == 1,
          "Cannot pass integer value ",
          scalar,
          " as Triton bool parameter");
      scalar_values_.emplace_back(static_cast<bool>(scalar));
      return &std::get<bool>(scalar_values_.back());
    }
    if (isInt32Param(param_type)) {
      TORCH_CHECK(
          scalar >= std::numeric_limits<int32_t>::min() &&
              scalar <= std::numeric_limits<int32_t>::max(),
          "Triton i32 parameter value out of range: ",
          scalar);
      scalar_values_.emplace_back(static_cast<int32_t>(scalar));
      return &std::get<int32_t>(scalar_values_.back());
    }
    scalar_values_.emplace_back(static_cast<int64_t>(scalar));
    return &std::get<int64_t>(scalar_values_.back());
  }

  if (value.isDouble()) {
    const auto scalar = value.toDouble();
    if (isFloat64Param(param_type)) {
      scalar_values_.emplace_back(static_cast<double>(scalar));
      return &std::get<double>(scalar_values_.back());
    }
    scalar_values_.emplace_back(static_cast<float>(scalar));
    return &std::get<float>(scalar_values_.back());
  }

  TORCH_CHECK(
      false,
      "Unsupported Triton scalar parameter IValue kind: ",
      value.tagKind());
}

void LaunchParams::parseCommonAttributes(const Node* node) {
  for (const auto& attr : node->attributes()) {
    std::vector<int64_t> grid;
    if (set_from_variant<std::vector<int64_t>>(grid, "grid", attr)) {
      TORCH_CHECK(grid.size() == 3, "grid must be a 3D vector");
      grid_dims = GridDims(
          static_cast<int>(grid[0]),
          static_cast<int>(grid[1]),
          static_cast<int>(grid[2]));
    }
  }
}

std::unique_ptr<LaunchParams> TritonKernelManager::createLaunchParams(
    const Node* node) const {
  auto params = std::make_unique<LaunchParams>();
  params->parseCommonAttributes(node);
  return params;
}

} // namespace torch::nativert
