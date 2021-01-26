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

std::mutex constantValueMutex;

class ConstantValueMap;

ConstantValueMap* ConstantValueMap::instance= nullptr;

ConstantValueMap& ConstantValueMap::getInstance(){
  std::lock_guard<std::mutex> myLock(constantValueMutex);
  if ( !instance ){
      instance= new ConstantValueMap();
  }
  // volatile int dummy{};
  return *instance;
}

void ConstantValueMap::SetRank(std::string tensorName, size_t rankValue) {
  ConstantValueMap::getInstance().rankMap.emplace(tensorName, rankValue);
}

void ConstantValueMap::SetShape(std::string tensorName, c10::VaryingShape<int64_t>& shapeValue) {
  ConstantValueMap::getInstance().shapeMap.emplace(tensorName, shapeValue);
}

bool ConstantValueMap::HasShape(std::string tensorName) {
  return ConstantValueMap::getInstance().shapeMap.find(tensorName) != 
    ConstantValueMap::getInstance().shapeMap.end();
}

c10::VaryingShape<int64_t> ConstantValueMap::GetShape(std::string tensorName) {
  return ConstantValueMap::getInstance().shapeMap[tensorName];
}

void ConstantValueMap::SetValue(std::string tensorName, at::Tensor value) {
  ConstantValueMap::getInstance().tensorValueMap.emplace(tensorName, value);
}

bool ConstantValueMap::HasValue(std::string tensorName) {
  return ConstantValueMap::getInstance().tensorValueMap.find(tensorName) != 
    ConstantValueMap::getInstance().tensorValueMap.end();
}

at::Tensor ConstantValueMap::GetValue(std::string tensorName) {
  return ConstantValueMap::getInstance().tensorValueMap[tensorName];
}

void ConstantValueMap::ClearMaps() {
  ConstantValueMap::getInstance().rankMap.clear();
  ConstantValueMap::getInstance().shapeMap.clear();
  ConstantValueMap::getInstance().tensorValueMap.clear();
}

void ConstantValueMap::PrintMaps() {
  std::cout << "Print rank/shape Maps:" << std::endl;
  for (const auto& x: ConstantValueMap::getInstance().rankMap) {
    std::stringstream ss;
    if (ConstantValueMap::getInstance().shapeMap.find(x.first) != 
        ConstantValueMap::getInstance().shapeMap.end()) {
      auto shape = ConstantValueMap::getInstance().shapeMap[x.first].concrete_sizes();
      if (shape.has_value()) {
        for (const auto& sz : shape.value()) {
          ss << sz << ", ";
        }
      }
    }
    ss << " (rank = " << x.second << ")";
    std::cout << "node " << x.first << ": " << ss.str() << std::endl;
  }
  std::cout << std::endl;
  std::cout << "Print Value Maps:" << std::endl;
  for (const auto& x: ConstantValueMap::getInstance().tensorValueMap) {
    std::cout << "node " << x.first << ": " << x.second << std::endl;
  }
}

} // namespace jit
} // namespace torch
