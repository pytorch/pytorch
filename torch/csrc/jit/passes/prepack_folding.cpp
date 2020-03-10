#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/prepack_folding.h>

namespace torch {
namespace jit {

// Must run this pass after constant folding.
void FoldPrePackingOps(script::Module& m,
    PrePackingOpsFilterFn is_foldable_op) {
  // Since this pass can be called by quantization or other passes as well,
  // we need to make sure we generate a unique "packed_weight_\d+"
  // Thus static uid.
  static int64_t uid = 0;

  auto method = m.get_method("forward");
  auto graph = method.graph();
  std::stack<Block*> blocks_to_visit;
  std::unordered_set<Node*> nodes_to_delete;
  blocks_to_visit.push(graph->block());
  std::string attr_name_base("packed_weight_");
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (is_foldable_op(n)) {
        auto optional_outputs = runNodeIfInputsAreConstant(n);
        if (optional_outputs) {
          auto outputs = optional_outputs.value();
          TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
          auto attr_name = attr_name_base + c10::to_string(uid++);
          m.register_attribute(attr_name, n->output(0)->type(), outputs[0]);
          Value* prepack_op_value = n->output(0);
          WithInsertPoint ins(prepack_op_value->node());
          Value* packed_weight_attr =
            graph->insertGetAttr(graph->inputs()[0], attr_name)
                  ->setType(n->output(0)->type());
          prepack_op_value->replaceAllUsesWith(packed_weight_attr);
          nodes_to_delete.insert(n);
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  for (auto n : nodes_to_delete) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete) {
    n->destroy();
  }
}

} // namespace jit
} // namespace torch
