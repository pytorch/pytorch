#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <mutex>
#include <unordered_map>

namespace torch {
namespace jit {

class TORCH_API TensorShapeMap{
public:
  static TensorShapeMap& getInstance();
  static void SetDim(std::string dimName, size_t dimValue);
  static void SetShape(std::string dimName, c10::VaryingShape<int64_t>& shapeValue);
  static bool HasShape(std::string dimName);
  static c10::VaryingShape<int64_t> GetShape(std::string dimName);
  static void PrintMaps();
  ~TensorShapeMap()= default;
private:
  TensorShapeMap() {};
  TensorShapeMap& operator=(const TensorShapeMap&)= delete;

  static TensorShapeMap* instance;
  std::unordered_map<std::string, size_t> dimMap;
  std::unordered_map<std::string, c10::VaryingShape<int64_t>> shapeMap;  
};

class ConstantTensorMap{
public:
  static ConstantTensorMap& getInstance();  
  static void SetValue(std::string dimName, std::vector<int64_t>& value);
  static bool HasValue(std::string dimName);
  static std::vector<int64_t> GetValue(std::string dimName);
  static void PrintMaps();
  ~ConstantTensorMap()= default;
private:
  ConstantTensorMap() {};  
  ConstantTensorMap& operator=(const ConstantTensorMap&)= delete;

  static ConstantTensorMap* instance;
  std::unordered_map<std::string, std::vector<int64_t>> int64Map;
};

} // namespace jit
} // namespace torch
