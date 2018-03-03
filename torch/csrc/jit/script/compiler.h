#pragma once
#include <functional>
#include <memory>
#include <string>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/error_report.h"
#include "torch/csrc/jit/script/tree_views.h"
#include "torch/csrc/jit/script/module.h"

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

void defineMethodsInModule(Module & m, const std::string& source, const Resolver& resolver);
void defineMethodsInModule(Module & m, const std::vector<Def>& definitions, const Resolver& resolver);
std::shared_ptr<Graph> defineFunction(Def def, const Resolver& resolver);

} // namespace script
} // namespace jit
} // namespace torch
