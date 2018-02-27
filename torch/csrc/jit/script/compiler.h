#pragma once
#include <functional>
#include <memory>
#include <string>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/jit/script/error_report.h"
#include "torch/csrc/jit/script/tree_views.h"

namespace torch {
namespace jit {
namespace script {

struct Resolver {
  virtual std::vector<Value*> resolveCall(SourceRange location, Node* n) {
    throw ErrorReport(location) << "Unknown function " << n->kind().toString();
  }
};

struct CompilationUnitImpl;
struct CompilationUnit {
  CompilationUnit();
  void define(const std::string& source, Resolver* resolver);
  void defineFunction(const Def& def, Resolver* resolver);
  std::shared_ptr<Graph> getGraph(const std::string& func_name);
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

std::shared_ptr<Graph> jitScriptCompile(Def def, Resolver* resolver);

} // namespace script
} // namespace jit
} // namespace torch
