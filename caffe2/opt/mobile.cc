#include "caffe2/opt/converter.h"
#include "caffe2/opt/mobile.h"
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace opt {

using namespace nom;

caffe2::NetDef addNNPACK(caffe2::NetDef net, bool low_memory) {
  auto nn = convertToNNModule(net);
  for (auto node : nn.dataFlow.getMutableNodes()) {
    auto* nodeData = node->data().get(); // Let graph retain ownership.

    // Skip blobs.
    if (!isa<nom::repr::NeuralNetOperator>(nodeData)) {
      continue;
    }

    // Check if it is a convolution.
    auto nnOp = dyn_cast<nom::repr::NeuralNetOperator>(nodeData);
    if (!isa<nom::repr::Conv>(nnOp)) {
      continue;
    }

    // Requires X, W, b for NNPACK
    if (node->getInEdges().size() < 3) {
      continue;
    }

    std::string engine = "NNPACK";

    // Now do some specific checks to see if an NNPACK engine is correct.
    bool validTransformCandidate = true;
    auto conv = dyn_cast<nom::repr::Conv>(nnOp);

    if (conv->getLayout() != nom::repr::Conv::NNLayout::NCHW) {
      continue;
    }

    // NNPACK only supports stride == 1
    for (auto stride : conv->getStrides()) {
      if (stride != 1) {
        validTransformCandidate = false;
        break;
      }
    }
    if (!validTransformCandidate) {
      continue;
    }

    // NNPACK only supports 2DConv.
    const auto& kernelShape = conv->getKernelShape();
    if (kernelShape.size() != 2) {
      continue;
    }

    // Kx1 and 1xK convs are inefficient in NNPACK.
    if (kernelShape[0] != kernelShape[1]) {
      if (kernelShape[0] == 1 || kernelShape[1] == 1) {
        continue;
      }
    }

    // We're good to use our engine.
    auto annotation = conv->getMutableAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      continue;
    }
    auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
    op->set_engine(engine);
    if (!low_memory) {
      auto* precompute_argument = op->add_arg();
      precompute_argument->set_name("convolution_transform_strategy");
      precompute_argument->set_s("PRECOMPUTE");
    }
  }
  return convertToCaffe2Proto(nn, net);
}

namespace {

inline bool isNNPACKConvReluEfficient(std::string algo, repr::Conv* conv) {
  if (algo == "AUTO" || algo == "") {
    for (auto stride : conv->getStrides()) {
      if (stride > 1) {
        return false;
      }
    }
    for (auto kernel : conv->getKernelShape()) {
      if (kernel < 2) {
        return false;
      }
    }
  } else if (!(algo == "WINOGRAD" || algo == "WINOGRAD_FP16" ||
               algo == "FT8x8" || algo == "FT16x16")) {
    return false;
  }
  return true;
}

} // namespace

caffe2::NetDef fuseNNPACKConvRelu(caffe2::NetDef net) {
  auto nn = convertToNNModule(net);
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn.dataFlow)) {
    repr::NNGraph::NodeRef conv_node;
    repr::Conv* conv;
    std::tie(conv, conv_node) = node_pair;

    auto conv_outputs = repr::nn::getOutputs(conv_node);
    if (conv_outputs.size() != 1) {
      continue;
    }
    auto conv_output = conv_outputs.front();

    auto consumers = repr::nn::getConsumers(conv_output);
    if (consumers.size() != 1) {
      continue;
    }
    if (!repr::nn::is<repr::Relu>(consumers.front())) {
      continue;
    }
    auto relu_node = consumers.front();

    auto annotation = conv->getMutableAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      continue;
    }
    auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();

    // We only want to fuse for fast NNPACK convs
    if (op->engine() != "NNPACK") {
      continue;
    }
    caffe2::string algo = "AUTO";
    for (auto arg : op->arg()) {
      if (arg.name() == "algo") {
        algo = arg.s();
      }
    }
    if (!isNNPACKConvReluEfficient(algo, conv)) {
      continue;
    }

    auto relu_outputs = repr::nn::getOutputs(relu_node);
    if (relu_outputs.size() != 1) {
      continue;
    }
    auto relu_output = relu_outputs.front();

    auto output_tensor = repr::nn::get<repr::Tensor>(relu_output);
    auto output_node = relu_output;
    auto input_tensor = repr::nn::get<repr::Tensor>(repr::nn::getInputs(conv_node).front());

    // Conv cannot be in-place
    if (output_tensor->getName() != input_tensor->getName()) {
      nn.dataFlow.replaceNode(conv_output, relu_output);
      nn.dataFlow.deleteNode(relu_node);
      nn.dataFlow.deleteNode(conv_output);
    } else {
      nn.dataFlow.replaceNode(relu_output, conv_output);
      output_tensor = repr::nn::get<repr::Tensor>(conv_output);
      output_node = conv_output;
      nn.dataFlow.deleteNode(relu_node);
      nn.dataFlow.deleteNode(relu_output);
    }

    // We may have accidentally made the next op in-place
    // In future iterations of transformations this won't be an issue,
    // but current caffe2 predictor usage requires things like 
    // external_input and output to be unchanged.
    bool rectify_inplace = false;
    for (auto& consumer : repr::nn::getConsumers(output_node)) {
      for (auto& consumer_output : repr::nn::getOutputs(consumer)) {
        auto co_name = repr::nn::get<repr::Tensor>(consumer_output)->getName();
        if (co_name == output_tensor->getName()) {
          rectify_inplace = true;
        }
      }
    }
    if (rectify_inplace) {
      auto new_output = nn.dataFlow.createNode(make_unique<repr::Tensor>(output_tensor->getName() + "_fusion_fix"));
      nn.dataFlow.replaceNode(output_node, new_output);
    }

    auto* arg = op->add_arg();
    arg->set_name("activation");
    arg->set_s("Relu");
  }
  return convertToCaffe2Proto(nn, net);
}

} // namespace opt
} // namespace caffe2
