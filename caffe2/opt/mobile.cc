#include "caffe2/opt/mobile.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"
#include "caffe2/opt/passes.h"

namespace caffe2 {
namespace opt {

using namespace nom;

void addNNPACK(repr::NNModule* nn, bool low_memory) {
  for (auto node : nn->dataFlow.getMutableNodes()) {
    // Skip blobs.
    NOM_REQUIRE_OR_CONT(repr::nn::is<repr::NeuralNetOperator>(node));

    // Check if it is a convolution.
    auto nnOp = repr::nn::get<repr::NeuralNetOperator>(node);
    NOM_REQUIRE_OR_CONT(isa<nom::repr::Conv>(nnOp));

    // Requires X, W, b for NNPACK
    NOM_REQUIRE_OR_CONT(node->getInEdges().size() >= 3);

    std::string engine = "NNPACK";

    // Now do some specific checks to see if an NNPACK engine is correct.
    bool validTransformCandidate = true;
    auto conv = dyn_cast<nom::repr::Conv>(nnOp);

    NOM_REQUIRE_OR_CONT(conv->getLayout() == nom::repr::Conv::NNLayout::NCHW);

    // NNPACK only supports stride == 1
    for (auto stride : conv->getStrides()) {
      if (stride != 1) {
        validTransformCandidate = false;
        break;
      }
    }
    NOM_REQUIRE_OR_CONT(validTransformCandidate);

    // NNPACK only supports 2DConv.
    const auto& kernelShape = conv->getKernelShape();
    NOM_REQUIRE_OR_CONT(kernelShape.size() == 2);

    // Kx1 and 1xK convs are inefficient in NNPACK.
    if (kernelShape[0] != kernelShape[1]) {
      NOM_REQUIRE_OR_CONT(kernelShape[0] != 1 && kernelShape[1] != 1);
    }

    // We're good to use our engine.
    auto annotation = conv->getMutableAnnotation();
    NOM_REQUIRE_OR_CONT(annotation && isa<Caffe2Annotation>(annotation));

    auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
    op->set_engine(engine);
    if (!low_memory) {
      auto* precompute_argument = op->add_arg();
      precompute_argument->set_name("convolution_transform_strategy");
      precompute_argument->set_s("PRECOMPUTE");
    }
  }
}

namespace {

inline bool isNNPACKConvReluEfficient(
    const std::string& algo,
    const repr::Conv& conv) {
  if (algo == "AUTO" || algo == "") {
    for (auto stride : conv.getStrides()) {
      if (stride > 1) {
        return false;
      }
    }
    for (auto kernel : conv.getKernelShape()) {
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

void fuseNNPACKConvRelu(repr::NNModule* nn) {
  auto should_fuse = [](const repr::Conv& conv) {
    const auto annotation = conv.getAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      return false;
    }
    const auto& op = dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();

    // We only want to fuse for fast NNPACK convs
    if (op.engine() != "NNPACK") {
      return false;
    }
    caffe2::string algo = "AUTO";
    for (const auto &arg : op.arg()) {
      if (arg.name() == "algo") {
        algo = arg.s();
      }
    }
    if (!isNNPACKConvReluEfficient(algo, conv)) {
      return false;
    }
    return true;
  };

  auto postprocess = [](repr::NNGraph::NodeRef conv_node) {
    auto conv = repr::nn::get<repr::Conv>(conv_node);
    auto annotation = conv->getMutableAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      return;
    }
    auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
    auto* arg = op->add_arg();
    arg->set_name("activation");
    arg->set_s("Relu");
  };

  fuseActivation<repr::Conv, repr::Relu>(nn, should_fuse, postprocess);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPT_PASS_FROM_FUNC(FuseNNPACKConvRelu, fuseNNPACKConvRelu);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPT_PASS_FROM_FUNC(AddNNPACK, addNNPACK);

} // namespace opt
} // namespace caffe2
