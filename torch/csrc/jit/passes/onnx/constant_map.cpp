#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace torch::jit {

// Meyer’s Singleton for C++ 14
ConstantValueMap& ConstantValueMap::getInstance() {
  static ConstantValueMap s;
  return s;
}

void ConstantValueMap::SetRank(
    const std::string& tensorName,
    size_t rankValue) {
  ConstantValueMap::getInstance().rankMap[tensorName] = rankValue;
  ConstantValueMap::getInstance().useInferredTypeMap[tensorName] = true;
}

bool ConstantValueMap::HasRank(const std::string& tensorName) {
  return ConstantValueMap::getInstance().rankMap.find(tensorName) !=
      ConstantValueMap::getInstance().rankMap.end();
}

std::optional<size_t> ConstantValueMap::GetRank(const std::string& tensorName) {
  if (!HasRank(tensorName)) {
    return std::nullopt;
  }
  return ConstantValueMap::getInstance().rankMap[tensorName];
}

void ConstantValueMap::SetAllGraphInputsStatic(bool all_static) {
  ConstantValueMap::getInstance().allGraphInputsStatic =
      std::make_optional(all_static);
}

std::optional<bool> ConstantValueMap::GetAllGraphInputsStatic() {
  return ConstantValueMap::getInstance().allGraphInputsStatic;
}

void ConstantValueMap::SetAllGraphInputsReliableComputed(bool computed) {
  ConstantValueMap::getInstance().allGraphInputsReliableComputed = computed;
}

bool ConstantValueMap::GetAllGraphInputsReliableComputed() {
  return ConstantValueMap::getInstance().allGraphInputsReliableComputed;
}

void ConstantValueMap::SetShape(
    const std::string& tensorName,
    const c10::SymbolicShape& shapeValue) {
  ConstantValueMap::getInstance().shapeMap[tensorName] = shapeValue;
  ConstantValueMap::getInstance().useInferredTypeMap[tensorName] = true;
}

bool ConstantValueMap::HasShape(const std::string& tensorName) {
  return ConstantValueMap::getInstance().shapeMap.find(tensorName) !=
      ConstantValueMap::getInstance().shapeMap.end();
}

std::optional<c10::SymbolicShape> ConstantValueMap::GetShape(
    const std::string& tensorName) {
  if (!HasShape(tensorName)) {
    return std::nullopt;
  }
  return ConstantValueMap::getInstance().shapeMap[tensorName];
}

void ConstantValueMap::SetValue(
    const std::string& tensorName,
    const at::Tensor& value) {
  ConstantValueMap::getInstance().tensorValueMap[tensorName] = value;
}

bool ConstantValueMap::HasValue(const std::string& tensorName) {
  return ConstantValueMap::getInstance().tensorValueMap.find(tensorName) !=
      ConstantValueMap::getInstance().tensorValueMap.end();
}

std::optional<at::Tensor> ConstantValueMap::GetValue(
    const std::string& tensorName) {
  if (!HasValue(tensorName)) {
    return std::nullopt;
  }
  return ConstantValueMap::getInstance().tensorValueMap[tensorName];
}

void ConstantValueMap::EraseValue(const std::string& tensorName) {
  ConstantValueMap::getInstance().tensorValueMap.erase(tensorName);
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

std::optional<std::vector<int64_t>> ConstantValueMap::GetShapeInto1DInt64Vector(
    const std::string& value_name) {
  if (ConstantValueMap::HasShape(value_name)) {
    auto shape_size = ConstantValueMap::GetShape(value_name).value();
    if (shape_size.isComplete()) {
      auto shape_value =
          ConstantValueMap::GetCompleteShapeInto1DInt64Vector(shape_size);
      return shape_value;
    }
  }
  return std::nullopt;
}

std::optional<std::vector<int64_t>> ConstantValueMap::
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
  return std::nullopt;
}

// accessor<int64_t, 1> for 1DInt64 case.
std::vector<int64_t> ConstantValueMap::GetValueInto1DInt64Vector(
    const std::string& value_name) {
  auto value = ConstantValueMap::GetValue(value_name).value();
  auto value_int64_t = value.toType(at::ScalarType::Long);
  std::vector<int64_t> value_vector;
  value_vector.reserve(value_int64_t.size(0));
  auto value_size_a = value_int64_t.accessor<int64_t, 1>();
  for (const auto i : c10::irange(value_int64_t.size(0))) {
    value_vector.emplace_back(static_cast<int64_t>(value_size_a[i]));
  }
  return value_vector;
}

void ConstantValueMap::SetTypeReliable(
    const std::string& tensorName,
    bool value) {
  ConstantValueMap::getInstance().typeReliableMap[tensorName] = value;
}

bool ConstantValueMap::HasTypeReliable(const std::string& tensorName) {
  return ConstantValueMap::getInstance().typeReliableMap.find(tensorName) !=
      ConstantValueMap::getInstance().typeReliableMap.end();
}

std::optional<bool> ConstantValueMap::GetTypeReliable(
    const std::string& tensorName) {
  if (!HasTypeReliable(tensorName)) {
    return std::nullopt;
  }
  return ConstantValueMap::getInstance().typeReliableMap[tensorName];
}

void ConstantValueMap::SetUseInferredType(
    const std::string& tensorName,
    bool value) {
  ConstantValueMap::getInstance().useInferredTypeMap[tensorName] = value;
}

bool ConstantValueMap::HasUseInferredType(const std::string& tensorName) {
  return ConstantValueMap::getInstance().useInferredTypeMap.find(tensorName) !=
      ConstantValueMap::getInstance().useInferredTypeMap.end();
}

std::optional<bool> ConstantValueMap::GetUseInferredType(
    const std::string& tensorName) {
  if (!HasUseInferredType(tensorName)) {
    return std::nullopt;
  }
  return ConstantValueMap::getInstance().useInferredTypeMap[tensorName];
}

void ConstantValueMap::SetShapeValue(
    const std::string& tensorName,
    const c10::SymbolicShape& shapeValue) {
  ConstantValueMap::getInstance().shapeValueMap[tensorName] = shapeValue;
}

bool ConstantValueMap::HasShapeValue(const std::string& tensorName) {
  return ConstantValueMap::getInstance().shapeValueMap.find(tensorName) !=
      ConstantValueMap::getInstance().shapeValueMap.end();
}

std::optional<c10::SymbolicShape> ConstantValueMap::GetShapeValue(
    const std::string& tensorName) {
  if (!HasShapeValue(tensorName)) {
    return std::nullopt;
  }
  return ConstantValueMap::getInstance().shapeValueMap[tensorName];
}

// Gets the inferredShapeData which is obtained by ONNX data propagation
ShapeDataMap& ConstantValueMap::GetInferredShapeData() {
  return ConstantValueMap::getInstance().inferredShapeData;
}

SymbolDimMap& ConstantValueMap::GetSymbolDimMap() {
  return ConstantValueMap::getInstance().symbolDimMap;
}

DimSymbolMap& ConstantValueMap::GetDimSymbolMap() {
  return ConstantValueMap::getInstance().dimSymbolMap;
}

template <typename Map>
void UpdateStrKey(
    Map& map,
    const std::string& old_key,
    const std::string& new_key) {
  TORCH_INTERNAL_ASSERT(old_key != new_key);
  if (map.find(old_key) == map.end()) {
    return;
  }
  map[new_key] = map[old_key];
  map.erase(old_key);
}

void ConstantValueMap::UpdateValueName(
    const std::string& old_name,
    const std::string& new_name) {
  if (old_name == new_name) {
    return;
  }
  UpdateStrKey<decltype(rankMap)>(
      ConstantValueMap::getInstance().rankMap, old_name, new_name);
  UpdateStrKey<decltype(shapeMap)>(
      ConstantValueMap::getInstance().shapeMap, old_name, new_name);
  UpdateStrKey<decltype(tensorValueMap)>(
      ConstantValueMap::getInstance().tensorValueMap, old_name, new_name);
  UpdateStrKey<decltype(typeReliableMap)>(
      ConstantValueMap::getInstance().typeReliableMap, old_name, new_name);
  UpdateStrKey<decltype(useInferredTypeMap)>(
      ConstantValueMap::getInstance().useInferredTypeMap, old_name, new_name);
  UpdateStrKey<decltype(shapeValueMap)>(
      ConstantValueMap::getInstance().shapeValueMap, old_name, new_name);
  UpdateStrKey<decltype(inferredShapeData)>(
      ConstantValueMap::getInstance().inferredShapeData, old_name, new_name);
}

void ConstantValueMap::ClearMaps() {
  ConstantValueMap::getInstance().rankMap.clear();
  ConstantValueMap::getInstance().shapeMap.clear();
  ConstantValueMap::getInstance().tensorValueMap.clear();
  ConstantValueMap::getInstance().typeReliableMap.clear();
  ConstantValueMap::getInstance().useInferredTypeMap.clear();
  ConstantValueMap::getInstance().shapeValueMap.clear();
  ConstantValueMap::getInstance().inferredShapeData.clear();
  ConstantValueMap::getInstance().symbolDimMap.clear();
  ConstantValueMap::getInstance().dimSymbolMap.clear();
  ConstantValueMap::getInstance().allGraphInputsStatic = std::nullopt;
  ConstantValueMap::getInstance().allGraphInputsReliableComputed = false;
}

// For debug only.
void ConstantValueMap::PrintMaps() {
  std::cout << "Rank/Shape Map:" << '\n';
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
    std::cout << "node " << x.first << ": " << ss.str() << '\n';
  }
  std::cout << '\n';
  std::cout << "Value Map:" << '\n';
  for (const auto& x : ConstantValueMap::getInstance().tensorValueMap) {
    std::cout << "node " << x.first << ": " << x.second << '\n';
  }
  std::cout << '\n';
  std::cout << "TypeReliable Map:" << '\n';
  size_t count = 0;
  for (const auto& x : ConstantValueMap::getInstance().typeReliableMap) {
    std::cout << "(node " << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << '\n';
    }
  }
  std::cout << '\n';
  std::cout << "UseInferredType Map:" << '\n';
  count = 0;
  for (const auto& x : ConstantValueMap::getInstance().useInferredTypeMap) {
    std::cout << "(node " << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << '\n';
    }
  }
  std::cout << '\n';
  std::cout << "ShapeValue Map:" << '\n';
  count = 0;
  for (const auto& x : ConstantValueMap::getInstance().shapeValueMap) {
    std::cout << "(node " << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << '\n';
    }
  }
  std::cout << '\n';
  std::cout << "InferredShape Map:" << '\n';
  count = 0;
  for (const auto& x : ConstantValueMap::getInstance().inferredShapeData) {
    std::cout << "(node " << x.first << ": ";
    for (const auto& dim : x.second.dim()) {
      if (dim.has_dim_param()) {
        std::cout << dim.dim_param() << " ";
      } else {
        std::cout << dim.dim_value() << " ";
      }
    }
    std::cout << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << '\n';
    }
  }
  std::cout << '\n';
  std::cout << "SymbolDim Map:" << '\n';
  count = 0;
  for (const auto& x : ConstantValueMap::getInstance().symbolDimMap) {
    std::cout << "(" << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << '\n';
    }
  }
  std::cout << "DimSymbol Map:" << '\n';
  count = 0;
  for (const auto& x : ConstantValueMap::getInstance().dimSymbolMap) {
    std::cout << "(" << x.first << ": " << x.second << "), ";
    count++;
    if (count % 10 == 0) {
      std::cout << '\n';
    }
  }
}

} // namespace torch::jit
