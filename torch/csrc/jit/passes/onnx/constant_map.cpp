#include <torch/csrc/jit/passes/onnx/constant_map.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <string>
#include <iostream>
#include <sstream>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

std::mutex tensorShapeMutex;
std::mutex constantTensorMutex;

class TensorShapeMap;
class ConstantTensorMap;

TensorShapeMap* TensorShapeMap::instance= nullptr;
ConstantTensorMap* ConstantTensorMap::instance= nullptr;

TensorShapeMap& TensorShapeMap::getInstance(){
  std::lock_guard<std::mutex> myLock(tensorShapeMutex);
  if ( !instance ){
      instance= new TensorShapeMap();
  }
  // volatile int dummy{};
  return *instance;
}

void TensorShapeMap::SetDim(std::string dimName, size_t dimValue) {
  TensorShapeMap::getInstance().dimMap.emplace(dimName, dimValue);
}

void TensorShapeMap::SetShape(std::string dimName, c10::VaryingShape<int64_t>& shapeValue) {
  TensorShapeMap::getInstance().shapeMap.emplace(dimName, shapeValue);
}

bool TensorShapeMap::HasShape(std::string dimName) {
  return TensorShapeMap::getInstance().shapeMap.find(dimName) != 
    TensorShapeMap::getInstance().shapeMap.end();
}

c10::VaryingShape<int64_t> TensorShapeMap::GetShape(std::string dimName) {
  return TensorShapeMap::getInstance().shapeMap[dimName];
}

void TensorShapeMap::PrintMaps() {
  std::cout << "Print Dim Maps:" << std::endl;
  for (const auto& x: TensorShapeMap::getInstance().dimMap) {
    std::cout << x.first << ": " << x.second << std::endl;
  }
  std::cout << "Print Shape Maps:" << std::endl;
  for (const auto& x: TensorShapeMap::getInstance().shapeMap) {
    auto sizes = x.second.concrete_sizes();
    std::stringstream ss;
    if (sizes.has_value()) {
      for (const auto& sz : sizes.value()) {
        ss << sz << ", ";
      }
    } else if (TensorShapeMap::getInstance().dimMap.find(x.first) != 
               TensorShapeMap::getInstance().dimMap.end()) {
      auto dim = TensorShapeMap::getInstance().dimMap[x.first];
      for (auto i = 0; i < dim ; ++i) {
        ss << "*, ";
      }
    }
    std::cout << x.first << ": " << ss.str() << std::endl;
  }
}

ConstantTensorMap& ConstantTensorMap::getInstance(){
  std::lock_guard<std::mutex> myLock(constantTensorMutex);
  if ( !instance ){
      instance= new ConstantTensorMap();
  }
  // volatile int dummy{};
  return *instance;
}

void ConstantTensorMap::SetValue(std::string dimName, std::vector<int64_t>& value) {
  ConstantTensorMap::getInstance().int64Map.emplace(dimName, value);
}

bool ConstantTensorMap::HasValue(std::string dimName) {
  return ConstantTensorMap::getInstance().int64Map.find(dimName) != 
    ConstantTensorMap::getInstance().int64Map.end();
}

std::vector<int64_t> ConstantTensorMap::GetValue(std::string dimName) {
  return ConstantTensorMap::getInstance().int64Map[dimName];
}

void ConstantTensorMap::PrintMaps() {
  std::cout << "Print Value Maps:" << std::endl;
  for (const auto& x: ConstantTensorMap::getInstance().int64Map) {
    std::stringstream ss;
    for (const auto& sz : x.second) {
      ss << sz << ", ";
    }
    std::cout << x.first << ": " << ss.str() << std::endl;
  }
}

} // namespace jit
} // namespace torch
