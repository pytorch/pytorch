#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <torch/csrc/lazy/core/debug_util.h>

namespace {
  std::string GetFirstUserFrameInPythonIfEnabled() {
    static const auto LTC_ENABLE_SOURCE_INFO = std::getenv("LTC_ENABLE_SOURCE_INFO");
    if (!LTC_ENABLE_SOURCE_INFO) {
      return {};
    }

    return torch::lazy::GetFirstUserFrameInPython();
  }
}

namespace torch {
namespace lazy {

TSOpVector TsNode::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                         TSLoweringContext* loctx) const {
  // TODO(whc) beginning to invert the design here.  Move to provide a Lower()
  // method on each node, starting with codegen.  Once we delete most
  // non-codegen ops, make this pure-virtual and put Lower() on the remaining
  // non-codegen ops.  For now, returning empty list here triggers fallback to
  // old lowering path.
  return {};
}

TensorList::TensorList(OpList values)
  : TsNode(/*op=*/tensor_list_opkind,
           /*operands=*/values,
           /*shapes=*/std::vector<Shape>(),
         /*num_outputs=*/1,
         /*hash_seed=*/OperandHashes(values, /*seed=*/kHashSeed, enableDynamicShape())) {}

TSOpVector TensorList::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                             TSLoweringContext* loctx) const {

  std::vector<torch::jit::Value*> tensor_list;
  CHECK(!operands().empty());
  for (const torch::lazy::Output& operand : operands()) {
    tensor_list.emplace_back(loctx->GetOutputOp(operand));
  }
  auto graph = function->graph();
  auto listnode = graph->insertNode(graph->createList(tensor_list[0]->type(), tensor_list));
  return {listnode->output()};
}

}  // namespace lazy
}  // namespace torch
