#pragma once
#include <memory>
#include <string>
#include "caffe2/core/net.h"

namespace caffe2 {
namespace script {

struct CompilationUnitImpl;
struct CompilationUnit {
  CompilationUnit();
  void define(const std::string& str);
  std::unique_ptr<NetBase> createNet(Workspace* ws, const std::string& name);
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

} // namespace script
}; // namespace caffe2
