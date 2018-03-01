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
  // Given an external call node where the signature does not correspond to an
  // existing ATen operator, this function returns a new node that calls into
  // the correct external op, or throws if that op cannot be found.
  //
  // The function simply instatntiates the new node and returns it. It is the
  // responsiblity of the caller to insert the node into the graph and delete
  // the original node.
  virtual Node* resolveCall(SourceRange location, Node* n) const {
    throw ErrorReport(location) << "Unknown function " << n->kind().toString();
  }
};

struct CompilationUnitImpl;
struct CompilationUnit {
  CompilationUnit();
  void define(const std::string& source, const Resolver& resolver);
  void defineFunction(const Def& def, const Resolver& resolver);
  std::shared_ptr<Graph> getGraph(const std::string& func_name);
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

std::shared_ptr<Graph> jitScriptCompile(Def def, const Resolver& resolver);

} // namespace script
} // namespace jit
} // namespace torch
