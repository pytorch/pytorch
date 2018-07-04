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
  void defineExtern(const std::string& str, std::unique_ptr<NetDef> netdef);
  std::unique_ptr<NetBase> createNet(Workspace* ws, const std::string& name);
  std::string getProto(const std::string& functionName) const;
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

} // namespace script
}; // namespace caffe2
