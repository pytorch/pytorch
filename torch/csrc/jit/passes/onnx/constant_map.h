#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <mutex>
#include <unordered_map>

namespace torch {
namespace jit {

class ConstantValueMap {
public:
  static ConstantValueMap& getInstance();

  static void SetRank(const std::string& tensorName, size_t rankValue);

  static void SetShape(const std::string& tensorName, c10::VaryingShape<int64_t>& shapeValue);
  static bool HasShape(const std::string& tensorName);
  static c10::VaryingShape<int64_t> GetShape(const std::string& tensorName);

  static void SetValue(const std::string& tensorName, const at::Tensor& value);
  static bool HasValue(const std::string& tensorName);
  static at::Tensor GetValue(const std::string& tensorName);

  static void PrintMaps();
  static void ClearMaps();
  ~ConstantValueMap()= default;

private:
  ConstantValueMap() {};
  ConstantValueMap& operator=(const ConstantValueMap&)= delete;

  std::unordered_map<std::string, size_t> rankMap;
  std::unordered_map<std::string, c10::VaryingShape<int64_t>> shapeMap;
  std::unordered_map<std::string, at::Tensor> tensorValueMap;
};

} // namespace jit
} // namespace torch
