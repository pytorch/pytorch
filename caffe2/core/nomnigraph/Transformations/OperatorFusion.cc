#include "nomnigraph/Transformations/OperatorFusion.h"

#include "nomnigraph/Support/Casting.h"
#include "nomnigraph/Support/Pointer.h"

namespace nom {
namespace transformations {

bool fuseConvRelu(repr::NNGraph *g) {
  for (auto node : g->getMutableNodes()) {
    // Skip non-operators (tensors and supplementary nodes).
    if (!isa<repr::NeuralNetOperator>(node->data())) {
      continue;
    }

    // Conv check.
    if (!isa<repr::Conv>(
            dyn_cast<repr::NeuralNetOperator>(node->data().get()))) {
      continue;
    }

    // Single output (somewhat redundant).
    if (node->getOutEdges().size() != 1) {
      continue;
    }

    // Single user check.
    auto *tensorNode = node->getOutEdges()[0]->head();
    if (tensorNode->getOutEdges().size() != 1) {
      continue;
    }

    // Followed by Relu check.
    auto *nextNode = tensorNode->getOutEdges()[0]->head();
    if (!isa<repr::Relu>(
            dyn_cast<repr::NeuralNetOperator>(nextNode->data().get()))) {
      continue;
    }

    // Now we do the swap.
    auto *convNode = node;
    auto *reluNode = nextNode;

    // TODO make this a little safer, static_cast is messy.
    auto conv = static_cast<repr::Conv *>(convNode->mutableData()->release());

    // Seize ownership of the conv node's data
    auto *convReluNode =
        g->createNode(util::make_unique<repr::ConvRelu>(std::move(conv)));

    for (const auto &inEdge : convNode->getInEdges()) {
      auto *parent = inEdge->tail();
      g->createEdge(parent, convReluNode);
    }
    for (const auto &outEdge : reluNode->getOutEdges()) {
      auto *child = outEdge->head();
      g->createEdge(convReluNode, child);
    }

    g->deleteNode(convNode);
    g->deleteNode(tensorNode);
    g->deleteNode(reluNode);

    return true;
  }
  return false;
}

} // namespace transformations
} // namespace nom
