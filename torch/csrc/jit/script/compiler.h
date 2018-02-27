#pragma once
#include <functional>
#include <memory>
#include <string>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/jit/script/tree_views.h"

namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(Graph*, std::string)>;

struct CompilationUnitImpl;
struct CompilationUnit {
  CompilationUnit(ResolutionCallback rcb);
  void define(const std::string& source);
  void defineFunction(const Def& def);
  std::shared_ptr<Graph> getGraph(const std::string& func_name);
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

std::shared_ptr<Graph> jitScriptCompile(Def def, ResolutionCallback rcb);

} // namespace script
} // namespace jit
} // namespace torch
