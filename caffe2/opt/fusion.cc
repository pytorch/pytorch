#include "caffe2/opt/fusion.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/passes.h"

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
  for (auto convNode : repr::nn::nodeIterator<repr::Conv>(nn->dataFlow)) {
    auto output = repr::nn::getOutputs(convNode).front();
    auto consumers = repr::nn::getConsumers(output);
    NOM_REQUIRE_OR_CONT(consumers.size() == 1);

    auto consumer = consumers.front();
    NOM_REQUIRE_OR_CONT(repr::nn::is<repr::BatchNormalization>(consumer));

    auto bnNode = consumer;
    auto bn = repr::nn::get<repr::BatchNormalization>(bnNode);
    auto bnOutputs = nn::getOutputs(bnNode);
    NOM_REQUIRE_OR_CONT(bnOutputs.size() == 1);
    auto bnOutput = bnOutputs.front();

    auto convInputs = repr::nn::getInputs(convNode);
    CAFFE_ENFORCE(
        convInputs.size() >= 3,
        "Invalid convolution input size (TODO: optional bias)");

    auto bnInputs = repr::nn::getInputs(bnNode);
    CAFFE_ENFORCE(
        bnInputs.size() >= 5, "Invalid batch normalization input size");

#define EXPOSE_TENSOR_DATA(name, index, inputs)                            \
  auto name = repr::nn::get<repr::Tensor>(inputs[index]);                  \
  assert(ws->HasBlob(name->getName()) && "Blob not in workspace");         \
  auto name##Tensor = ws->GetBlob(name->getName())->GetMutableTensor(CPU); \
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

    nn->dataFlow.deleteNode(output);
    nn->dataFlow.createEdge(convNode, bnOutput);
    nn->dataFlow.deleteNode(bnNode);
    return true;
  }
  return false;
}

void fuseConvBN(nom::repr::NNModule* nn, caffe2::Workspace* ws) {
  while (fuseConvBNHelper(nn, ws)) {
  }
}

REGISTER_WS_OPT_PASS_FROM_FUNC(FuseConvBN, fuseConvBN);

} // namespace opt
} // namespace caffe2
