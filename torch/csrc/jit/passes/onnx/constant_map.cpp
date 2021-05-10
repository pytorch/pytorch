#include <torch/csrc/jit/passes/onnx/constant_map.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <iostream>
#include <sstream>
#include <string>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

// Meyer’s Singleton for C++ 14
ConstantValueMap& ConstantValueMap::getInstance() {
  static ConstantValueMap s;
  return s;
}

void ConstantValueMap::SetRank(
    const std::string& tensorName,
    size_t rankValue) {
  ConstantValueMap::getInstance().rankMap.emplace(tensorName, rankValue);
}

bool ConstantValueMap::HasRank(const std::string& tensorName) {
  return ConstantValueMap::getInstance().rankMap.find(tensorName) !=
      ConstantValueMap::getInstance().rankMap.end();
}

c10::optional<size_t> ConstantValueMap::GetRank(const std::string& tensorName) {
  if (!HasRank(tensorName)) {
    return c10::nullopt;
  }
  return ConstantValueMap::getInstance().rankMap[tensorName];
}

void ConstantValueMap::SetShape(
    const std::string& tensorName,
    const c10::SymbolicShape& shapeValue) {
  ConstantValueMap::getInstance().shapeMap.emplace(tensorName, shapeValue);
}

bool ConstantValueMap::HasShape(const std::string& tensorName) {
  return ConstantValueMap::getInstance().shapeMap.find(tensorName) !=
      ConstantValueMap::getInstance().shapeMap.end();
}

c10::optional<c10::SymbolicShape> ConstantValueMap::GetShape(
    const std::string& tensorName) {
  if (!HasShape(tensorName)) {
    return c10::nullopt;
  }
  return ConstantValueMap::getInstance().shapeMap[tensorName];
}

void ConstantValueMap::SetValue(
    const std::string& tensorName,
    const at::Tensor& value) {
  ConstantValueMap::getInstance().tensorValueMap.emplace(tensorName, value);
}

bool ConstantValueMap::HasValue(const std::string& tensorName) {
  return ConstantValueMap::getInstance().tensorValueMap.find(tensorName) !=
      ConstantValueMap::getInstance().tensorValueMap.end();
}

c10::optional<at::Tensor> ConstantValueMap::GetValue(
    const std::string& tensorName) {
  if (!HasValue(tensorName)) {
    return c10::nullopt;
  }
  return ConstantValueMap::getInstance().tensorValueMap[tensorName];
}

std::vector<int64_t> ConstantValueMap::GetCompleteShapeInto1DInt64Vector(
    const c10::SymbolicShape& shape) {
  TORCH_INTERNAL_ASSERT(shape.isComplete());
  std::vector<int64_t> shape_value;
  auto shape_symbol_list = shape.sizes().value();
  shape_value.reserve(shape_symbol_list.size());
  for (const auto& v : shape_symbol_list) {
    shape_value.emplace_back(v.static_size());
  }
  return shape_value;
}

c10::optional<std::vector<int64_t>> ConstantValueMap::GetShapeInto1DInt64Vector(
    const std::string& value_name) {
  if (ConstantValueMap::HasShape(value_name)) {
    auto shape_size = ConstantValueMap::GetShape(value_name).value();
    if (shape_size.isComplete()) {
      auto shape_value =
          ConstantValueMap::GetCompleteShapeInto1DInt64Vector(shape_size);
      return shape_value;
    }
  }
  return c10::nullopt;
}

c10::optional<std::vector<int64_t>> ConstantValueMap::
    GetShapeInto1DInt64VectorWithOneUnknown(const std::string& value_name) {
  if (ConstantValueMap::HasShape(value_name)) {
    auto shape_size = ConstantValueMap::GetShape(value_name).value();
    std::vector<int64_t> shape_value;
    if (shape_size.isComplete()) {
      shape_value =
          ConstantValueMap::GetCompleteShapeInto1DInt64Vector(shape_size);
      return shape_value;
    } else {
      size_t count_unknown = 0;
      auto shape_size_sizes = shape_size.sizes();
      if (shape_size_sizes.has_value()) {
        auto shape_symbol_list = shape_size_sizes.value();
        for (const auto& v : shape_symbol_list) {
          if (v.is_static()) {
            shape_value.emplace_back(v.static_size());
          } else {
            shape_value.emplace_back(-1);
            count_unknown += 1;
          }
        }
        if (count_unknown == 1) {
          return shape_value;
        }
      }
    }
  }
  return c10::nullopt;
}

// accessor<int64_t, 1> for 1DInt64 case.
std::vector<int64_t> ConstantValueMap::GetValueInto1DInt64Vector(
    const std::string& value_name) {
  auto value = ConstantValueMap::GetValue(value_name).value();
  auto value_int64_t = value.toType(at::ScalarType::Long);
  std::vector<int64_t> value_vector;
  value_vector.reserve(value_int64_t.size(0));
  auto value_size_a = value_int64_t.accessor<int64_t, 1>();
  for (auto i = 0; i < value_int64_t.size(0); i++) {
    value_vector.emplace_back(static_cast<int64_t>(value_size_a[i]));
  }
  return value_vector;
}

void ConstantValueMap::ClearMaps() {
  ConstantValueMap::getInstance().rankMap.clear();
  ConstantValueMap::getInstance().shapeMap.clear();
  ConstantValueMap::getInstance().tensorValueMap.clear();
}

// For debug only.
void ConstantValueMap::PrintMaps() {
  std::cout << "Print rank/shape Maps:" << std::endl;
  for (const auto& x : ConstantValueMap::getInstance().rankMap) {
    std::stringstream ss;
    if (ConstantValueMap::getInstance().shapeMap.find(x.first) !=
        ConstantValueMap::getInstance().shapeMap.end()) {
      auto shape_symbols =
          ConstantValueMap::getInstance().shapeMap[x.first].sizes();
      if (shape_symbols.has_value()) {
        for (const auto& shape_symbol : shape_symbols.value()) {
          if (shape_symbol.is_static()) {
            ss << shape_symbol.static_size() << ", ";
          } else {
            ss << "*, ";
          }
        }
      }
    }
    ss << " (rank = " << x.second << ")";
    std::cout << "node " << x.first << ": " << ss.str() << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Print Value Maps:" << std::endl;
  for (const auto& x : ConstantValueMap::getInstance().tensorValueMap) {
    std::cout << "node " << x.first << ": " << x.second << std::endl;
  }
}

} // namespace jit
} // namespace torch
