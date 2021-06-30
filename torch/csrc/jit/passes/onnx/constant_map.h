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
  static bool HasRank(const std::string& tensorName);
  static c10::optional<size_t> GetRank(const std::string& tensorName);

  static void SetShape(
      const std::string& tensorName,
      const c10::SymbolicShape& shapeValue);
  static bool HasShape(const std::string& tensorName);
  static c10::optional<c10::SymbolicShape> GetShape(
      const std::string& tensorName);

  static void SetValue(const std::string& tensorName, const at::Tensor& value);
  static bool HasValue(const std::string& tensorName);
  static c10::optional<at::Tensor> GetValue(const std::string& tensorName);

  static std::vector<int64_t> GetCompleteShapeInto1DInt64Vector(
      const c10::SymbolicShape& shape);
  static c10::optional<std::vector<int64_t>> GetShapeInto1DInt64Vector(
      const std::string& value_name);
  static c10::optional<std::vector<int64_t>>
  GetShapeInto1DInt64VectorWithOneUnknown(const std::string& value_name);
  static std::vector<int64_t> GetValueInto1DInt64Vector(
      const std::string& value_name);

  static void PrintMaps();
  static void ClearMaps();
  ~ConstantValueMap() = default;

 private:
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ConstantValueMap(){};
  // NOLINTNEXTLINE(modernize-use-equals-delete)
  ConstantValueMap& operator=(const ConstantValueMap&) = delete;

  std::unordered_map<std::string, size_t> rankMap;
  std::unordered_map<std::string, c10::SymbolicShape> shapeMap;
  std::unordered_map<std::string, at::Tensor> tensorValueMap;
};

} // namespace jit
} // namespace torch
