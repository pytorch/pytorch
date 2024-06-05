#include <torch/csrc/jit/passes/eliminate_no_ops.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

namespace {

bool allInputsAreTensors(Node* node) {
  for (const auto* value : node->inputs()) {
    const auto& type = value->type();
    if (!type->castRaw<TensorType>()) {
      return false;
    }
  }
  return true;
}

bool cannotOptimize(Node* node) {
  const auto kind = node->kind();
  if (kind == aten::__is__ || kind == aten::__isnot__) {
    return allInputsAreTensors(node);
  }
  return false;
}

// Certain ops can make this optimization unsound. For example,
// consider the following graph:
//   %y : Tensor = aten::detach(%x)
//   %b : bool = aten::__is__(%y, %x) (= False)
// After remove detach, we would get
//   %b : bool = aten::__is__(%x, %x) (= True!)
bool containsInvalidOp(std::shared_ptr<Graph>& graph) {
  for (auto* node : graph->nodes()) {
    if (cannotOptimize(node)) {
      return true;
    }
  }
  return false;
}

} // namespace

bool EliminateNoOps(
    std::shared_ptr<Graph>& graph,
    std::unordered_set<c10::Symbol> custom_ops) {
  GRAPH_DUMP("Before EliminateNoOps: ", graph);
  if (containsInvalidOp(graph)) {
    return false;
  }
  // Ops here should be of the form x = f(x, ...)
  std::unordered_set<c10::Symbol> no_ops{aten::detach};
  no_ops.insert(custom_ops.begin(), custom_ops.end());

  bool changed = false;

  auto graph_it = DepthFirstGraphNodeIterator(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    auto it = no_ops.find(node->kind());
    if (it == no_ops.end()) {
      continue;
    }

    changed = true;
    node->output()->replaceAllUsesWith(node->input(0));
  }

  if (changed) {
    EliminateDeadCode(graph);
  }

  GRAPH_DUMP("After EliminateNoOps: ", graph);
  return changed;
}

} // namespace jit
} // namespace torch
