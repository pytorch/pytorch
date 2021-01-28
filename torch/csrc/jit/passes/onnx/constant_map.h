#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <mutex>
#include <unordered_map>

namespace torch {
namespace jit {

class ConstantValueMap {
public:
  static ConstantValueMap& getInstance();

  static void SetRank(std::string tensorName, size_t rankValue);

  static void SetShape(std::string tensorName, c10::VaryingShape<int64_t>& shapeValue);
  static bool HasShape(std::string tensorName);
  static c10::VaryingShape<int64_t> GetShape(std::string tensorName);

  static void SetValue(std::string tensorName, at::Tensor value);
  static bool HasValue(std::string tensorName);
  static at::Tensor GetValue(std::string tensorName);

  static void PrintMaps();
  static void ClearMaps();
  ~ConstantValueMap()= default;
private:
  ConstantValueMap() {};
  ConstantValueMap& operator=(const ConstantValueMap&)= delete;

  static ConstantValueMap* instance;

  std::unordered_map<std::string, size_t> rankMap;
  std::unordered_map<std::string, c10::VaryingShape<int64_t>> shapeMap;
  std::unordered_map<std::string, at::Tensor> tensorValueMap;
};

} // namespace jit
} // namespace torch
