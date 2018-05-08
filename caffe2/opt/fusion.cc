#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"

namespace caffe2 {
namespace opt {

using namespace nom;

// $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// $$ X_{conv} = X * W + b_{conv} $$
// thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
// $$X_{bn} = X * \frac{sW}{\sqrt{\sigma + \epsilon}} + \frac{s(b_{conv} - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// or
// $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
// $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
bool fuseConvBNHelper(repr::NNModule* nn, caffe2::Workspace* ws) {
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    repr::NNGraph::NodeRef convNode;
    repr::Conv* conv;
    std::tie(conv, convNode) = node_pair;

    auto output = repr::nn::getOutputs(convNode).front();
    auto consumers = repr::nn::getConsumers(output);
    if (consumers.size() != 1) {
      continue;
    }
    auto consumer = consumers.front();
    if (!repr::nn::is<repr::BatchNormalization>(consumer)) {
      continue;
    }
    auto bnNode = consumer;
    auto bn = repr::nn::get<repr::BatchNormalization>(bnNode);

    auto convInputs = repr::nn::getInputs(convNode);
    if (convInputs.size() < 3) {
      assert(0 && "Invalid convolution input size (TODO: optional bias)");
      continue;
    }

    auto bnInputs = repr::nn::getInputs(bnNode);
    if (bnInputs.size() < 5) {
      assert(0 && "Invalid batch normalization input size");
      continue;
    }

#define EXPOSE_TENSOR_DATA(name, index, inputs)                              \
  auto name = repr::nn::get<repr::Tensor>(inputs[index]);                    \
  assert(ws->HasBlob(name->getName()) && "Blob not in workspace");           \
  auto name##Tensor = ws->GetBlob(name->getName())->GetMutable<TensorCPU>(); \
  auto name##Data = name##Tensor->mutable_data<float>();

    EXPOSE_TENSOR_DATA(filter, 1, convInputs);
    EXPOSE_TENSOR_DATA(biasConv, 2, convInputs);

    EXPOSE_TENSOR_DATA(scale, 1, bnInputs);
    EXPOSE_TENSOR_DATA(biasBN, 2, bnInputs);
    EXPOSE_TENSOR_DATA(mean, 3, bnInputs);
    EXPOSE_TENSOR_DATA(variance, 4, bnInputs);

#undef EXPOSE_TENSOR_DATA

    // Assume M{CHW,HWC}
    auto chwDim = filterTensor->dim32(1) * filterTensor->dim32(2) *
        filterTensor->dim32(3);
    for (auto c = 0; c < filterTensor->dim32(0); ++c) {
      float coeff =
          scaleData[c] / std::sqrt(varianceData[c] + bn->getEpsilon());
      for (auto i = 0; i < chwDim; ++i) {
        filterData[c * chwDim + i] *= coeff;
      }
      auto bias = (biasConvData[c] - meanData[c]) * coeff + biasBNData[c];
      biasConvData[c] = bias;
    }

    nn->dataFlow.deleteNode(bnNode);
    return true;
  }
  return false;
}

void fuseConvBN(nom::repr::NNModule* nn, caffe2::Workspace* ws) {
  while (fuseConvBNHelper(nn, ws)) {
  }
}
} // namespace opt
} // namespace caffe2
