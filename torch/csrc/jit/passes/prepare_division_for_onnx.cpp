#include "torch/csrc/jit/passes/prepare_division_for_onnx.h"
#include "torch/csrc/jit/constants.h"

namespace torch { namespace jit {

static void PrepareDivisionForONNXOnBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    for (auto sub : it->blocks()) {
      PrepareDivisionForONNXOnBlock(sub);
    }
    WithInsertPoint guard(*it);
    auto* subgraph = it->owningGraph();

    if (it->matches("aten::div(int a, int b) -> float")) {
      // Use onnx::cast before dividing
      std::vector<Value*> floattensor_inputs = fmap(it->inputs(), [&](Value* input) {
        Value* longtensor = subgraph->insertNode(subgraph->createNumToTensor(input))->output();
        // FLOAT = 1
        // https://github.com/onnx/onnx/blob/6bedd27b0307c9295039bd847895a27275160a98/onnx/onnx.in.proto#L282
        Node* cast = subgraph->create(onnx::Cast, {longtensor})->i_(attr::to, 1);
        return subgraph->insertNode(cast)->output();
      });

      it->replaceInput(0, floattensor_inputs[0]);
      it->replaceInput(1, floattensor_inputs[1]);
      it->output()->setType(CompleteTensorType::fromNumberType(FloatType::get()));
    }
  }
}

void PrepareDivisionForONNX(const std::shared_ptr<Graph>& graph) {
  PrepareDivisionForONNXOnBlock(graph->block());
}

}}

