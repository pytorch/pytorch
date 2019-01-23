#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/mobile.h"
#include "caffe2/opt/passes.h"

namespace caffe2 {
namespace opt {

using namespace nom;

C10_EXPORT void sinkMaxPool(nom::repr::NNModule* nn) {
  for (auto max_pool_node :
       repr::nn::nodeIterator<repr::MaxPool>(nn->dataFlow)) {
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

REGISTER_OPT_PASS_FROM_FUNC(SinkMaxPool, sinkMaxPool);

} // namespace opt
} // namespace caffe2
