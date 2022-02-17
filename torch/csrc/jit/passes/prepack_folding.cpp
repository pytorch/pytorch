#include <stack>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/prepack_folding.h>

namespace torch {
namespace jit {

// Must run this pass after constant folding.
void PrePackingOpsFolder(
    script::Module& m,
    const PrePackingOpsFilterFn& is_foldable_op,
    const std::string& attr_prefix) {
  for (auto& method : m.get_methods()) {
    int64_t uid = 0; // int + method name gives unique identifier
    auto graph = method.graph();
    std::stack<Block*> blocks_to_visit;
    std::unordered_set<Node*> nodes_to_delete;
    blocks_to_visit.push(graph->block());
    std::string attr_name_base =
        attr_prefix + "_" + method.name() + "._jit_pass_packed_weight_";
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
            TORCH_CHECK(
                !(m.type()->findAttributeSlot(attr_name)),
                "Attribute name ",
                attr_name,
                " already exist in",
                " module of type:",
                m.type()->name()->qualifiedName(),
                ". Please make sure that",
                " FoldPrePackingOps is run at the top level module only.");
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
}

} // namespace jit
} // namespace torch
