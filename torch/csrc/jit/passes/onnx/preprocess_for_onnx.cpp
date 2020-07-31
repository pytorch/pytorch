#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/preprocess_for_onnx.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {

static void PreProcessForInplaceOps(Block* b, std::shared_ptr<Graph>& graph) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PreProcessForInplaceOps(child_block, graph);
    }
  }

  for (auto input : b->inputs()) {
    for (auto use : input->uses()) {
      Node* node = use.user;
      if (!torch::jit::isInplaceOpVariant(graph, node)) {
        continue;
      }

      auto it = std::find(node->inputs().begin(), node->inputs().end(), input);

      if (it != node->inputs().end()) {
        int index = std::distance(node->inputs().begin(), it);

        std::cerr
            << "Warning: ONNX Preprocess - Removing mutation on block inputs. "
            << "This changes graph semantics." << std::endl;

        // insert a clone node following the graph input:
        // Example for graph input node %0:
        //
        //  %2 : None = prim::Constant()
        //  %3 : Tensor = aten::clone(%0, %2)
        //  %5 : Tensor = aten::zero_(%3)
        auto newNode = node->owningGraph()->create(aten::clone, 1);
        newNode->copyMetadata(input->node());
        newNode->addInput(input);

        auto* noneNode = node->owningGraph()->create(prim::Constant);
        noneNode->output()->setType(NoneType::get());
        newNode->addInput(noneNode->output());

        newNode->insertBefore(node);
        noneNode->insertBefore(newNode);
        node->replaceInput(index, newNode->output());
        input->replaceAllUsesAfterNodeWith(node, newNode->output());
      }
    }
  }
}
} // namespace

void PreprocessForONNX(std::shared_ptr<Graph>& graph) {
  PreProcessForInplaceOps(graph->block(), graph);
}

} // namespace jit
} // namespace torch