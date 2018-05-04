#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/mobile.h"

namespace caffe2 {
namespace opt {

using namespace nom;

void sinkMaxPool(nom::repr::NNModule* nn) {
  for (auto node_pair : repr::nn::dataIterator<repr::MaxPool>(nn->dataFlow)) {
    repr::NNGraph::NodeRef max_pool_node;
    repr::MaxPool* max_pool;
    std::tie(max_pool, max_pool_node) = node_pair;

    if (repr::nn::getInputs(max_pool_node).size() != 1) {
      continue;
    }

    auto max_pool_outputs = repr::nn::getOutputs(max_pool_node);
    if (max_pool_outputs.size() != 1) {
      continue;
    }

    auto consumers = repr::nn::getConsumers(max_pool_outputs.front());
    if (consumers.size() != 1) {
      continue;
    }

    // TODO Sink MaxPool in more cases.
    auto relu_node = consumers.front();
    if (!repr::nn::is<repr::Relu>(relu_node)) {
      continue;
    }

    if (repr::nn::getOutputs(relu_node).size() != 1) {
      continue;
    }

    // input -> MaxPool -> intermediate -> Relu -> output
    nn->dataFlow.swapNodes(max_pool_node, relu_node);
    // input -> Relu -> intermediate -> MaxPool -> output
  }
}

} // namespace opt
} // namespace caffe2
