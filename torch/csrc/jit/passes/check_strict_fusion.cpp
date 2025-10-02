
#include <torch/csrc/jit/passes/check_strict_fusion.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch::jit {

namespace {

bool isStrictFusion(Value* value) {
  const auto class_name = getModuleName(value);
  return class_name.has_value() &&
      (*class_name == "__torch__.torch.jit.strict_fusion");
}

} // namespace

static bool fusionGuardCheck(Symbol k) {
  return k == Symbol::prim("TensorExprDynamicGuard") || k == prim::TypeCheck ||
      k == prim::CudaFusionGuard || k == prim::RequiresGradCheck;
}

static std::unordered_set<Node*> collectValuesUsedInGuard(
    Node* guarding_if,
    Node* enter_node) {
  // DFS to collect
  std::unordered_set<Node*> visited_nodes;
  std::vector<Node*> queue = {guarding_if};

  while (!queue.empty()) {
    Node* curr = queue[queue.size() - 1];
    queue.pop_back();
    visited_nodes.insert(curr);
    // these nodes directly test Tensor inputs, and are not part of additional
    // guards inserted
    if (fusionGuardCheck(curr->kind())) {
      continue;
    }
    for (Value* v : curr->inputs()) {
      Node* inp_node = v->node();
      if (inp_node->isBefore(enter_node) ||
          inp_node->owningBlock() != enter_node->owningBlock()) {
        continue;
      }
      if (visited_nodes.count(inp_node)) {
        continue;
      }
      queue.push_back(inp_node);
    }
  }
  return visited_nodes;
}

static void checkForUnfusedOps(Node* enter_node) {
  std::vector<Node*> unsupported_nodes;
  std::vector<Node*> guarding_ifs; // if multiple, we will throw
  for (Node* node = enter_node->next(); node->kind() != prim::Exit;
       node = node->next()) {
    if (node->kind() == prim::If &&
        fusionGuardCheck(node->input()->node()->kind())) {
      guarding_ifs.push_back(node);
      continue;
    }
    unsupported_nodes.push_back(node);
  }

  if (guarding_ifs.size() > 1) {
    std::stringstream ss;
    ss << "Found multiple fusions: \n";
    for (Node* n : guarding_ifs) {
      ss << *n << "\n";
    }
    throw(ErrorReport(enter_node->input()->node()->sourceRange()) << ss.str());
  }

  // autodiff/nnc both insert a number of guards, see
  // `CudaFusionViewGuard Example Graph`
  // to check for unfused nodes, look at node's whose outputs
  // are not depended on by the fusion guard
  // restrict search for all values after the first
  // node in the prim::Enter block

  std::unordered_set<Node*> guarding_check_nodes;
  if (guarding_ifs.size() == 1) {
    guarding_check_nodes =
        collectValuesUsedInGuard(guarding_ifs[0], enter_node);
  }
  std::vector<Node*> unfused_nodes_not_used_in_guard;
  for (Node* unfused : unsupported_nodes) {
    if (!guarding_check_nodes.count(unfused)) {
      unfused_nodes_not_used_in_guard.push_back(unfused);
    }
  }
  if (!unfused_nodes_not_used_in_guard.empty()) {
    std::stringstream ss;
    ss << "Found unfused operators: \n";
    for (Node* unfused : unfused_nodes_not_used_in_guard) {
      ss << "\t";
      if (unfused->maybeSchema()) {
        ss << unfused->schema();
      } else {
        unfused->kind().toDisplayString();
      }
      ss << "\n";
    }
    throw(ErrorReport(enter_node->input()->node()->sourceRange()) << ss.str());
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

  // TODO: remove context manager after checks
  // TODO: improve control flow not taken, right now always errors
}

} // namespace torch::jit
