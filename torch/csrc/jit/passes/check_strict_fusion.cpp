
#include <torch/csrc/jit/passes/check_strict_fusion.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

namespace {

bool isStrictFusion(Value* value) {
  const auto class_name = getModuleName(value);
  return class_name.has_value() &&
      (*class_name == "__torch__.torch.jit.strict_fusion");
}

} // namespace

bool fusionGuardCheck(Symbol k) {
  return k == Symbol::prim("TensorExprDynamicGuard") || k == prim::TypeCheck ||
      k == prim::CudaFusionGuard;
}

void checkForUnfusedOps(Node* enter_node) {
  std::vector<Node*> unsupported_nodes;
  // TODO
  for (Node* node = enter_node->next(); node->kind() != prim::Exit;
       node = node->next()) {
    auto k = node->kind();
    if (fusionGuardCheck(k)) {
      continue;
    }
    // TODO: add support for guards that nvfuser adds
    if (node->kind() == prim::If &&
        fusionGuardCheck(node->input()->node()->kind())) {
      continue;
    }
    unsupported_nodes.push_back(node);
  }
  if (unsupported_nodes.size()) {
    std::stringstream ss;
    ss << "Found unfused operators: \n";
    for (Node* unfused : unsupported_nodes) {
      ss << "\t" << unfused->schema() << "\n";
    }
    // TODO: Context Manager source range lost right now
    throw ErrorReport(unsupported_nodes[0]->sourceRange()) << ss.str();
  }
}

void CheckStrictFusion(std::shared_ptr<Graph>& graph) {
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::Enter && isStrictFusion(n->input())) {
      checkForUnfusedOps(n);
    }
  }

  // TODO: compose/warn with auto diff ?
  // TODO: remove context manager after checks
  // TODO: what to do about control flow not taken (will always fail right now)
}

} // namespace jit
} // namespace torch
