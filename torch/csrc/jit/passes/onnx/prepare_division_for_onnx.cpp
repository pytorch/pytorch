#include <torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

// onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0,
// so before converting the ints to tensors we need to cast them to floats.
static void PrepareDivisionForONNXOnBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    for (auto sub : it->blocks()) {
      PrepareDivisionForONNXOnBlock(sub);
    }
    WithInsertPoint guard(*it);
    auto* subgraph = it->owningGraph();

    if (it->matches("aten::div(int a, int b) -> float")) {
      // Cast to Float before dividing
      std::vector<Value*> floattensor_inputs =
          fmap(it->inputs(), [&](Value* input) {
            auto* longtensor =
                subgraph->insertNode(subgraph->createNumToTensor(input))
                    ->output();
            auto* nonblocking = subgraph->insertConstant(0);
            auto* cast =
                subgraph->create(aten::_cast_Float, {longtensor, nonblocking});
            return subgraph->insertNode(cast)->output();
          });

      it->replaceInput(0, floattensor_inputs[0]);
      it->replaceInput(1, floattensor_inputs[1]);
      it->output()->setType(TensorType::fromNumberType(FloatType::get()));
    }
  }
}

void PrepareDivisionForONNX(const std::shared_ptr<Graph>& graph) {
  PrepareDivisionForONNXOnBlock(graph->block());
  GRAPH_DUMP("After PrepareDivisionForONNX: ", graph);
}

} // namespace jit
} // namespace torch
