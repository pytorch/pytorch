#include "mobile.h"
#include "nomnigraph/Converters/Caffe2.h"

namespace caffe2 {
namespace opt {

caffe2::NetDef addNNPACK(caffe2::NetDef net, bool low_memory) {
  auto nn = nom::converters::convertFromCaffe2Proto(net);
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
    auto* op = static_cast<caffe2::OperatorDef*>(annotation->getSaved());
    op->set_engine(engine);
    if (!low_memory) {
      auto* precompute_argument = op->add_arg();
      precompute_argument->set_name("convolution_transform_strategy");
      precompute_argument->set_s("PRECOMPUTE");
    }
  }
  return nom::converters::convertToCaffe2Proto(nn);
}

} // namespace opt
} // namespace caffe2
