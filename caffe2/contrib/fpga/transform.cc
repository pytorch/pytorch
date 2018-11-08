#include "caffe2/opt/converter.h"
#include "caffe2/opt/device.h"
#include "caffe2/opt/passes.h"
#include "nomnigraph/Support/Common.h"

namespace caffe2 {
namespace opt {

using namespace nom;
using namespace nom::repr;

static std::unordered_set<std::string> OpenCLCompatibleOperators = {};

void convertToOpenCL(NNModule* nn) {
  for (auto node : nn->dataFlow.getMutableNodes()) {
    NOM_REQUIRE_OR_CONT(nn::is<NeuralNetOperator>(node));
    auto nnOp = nn::get<NeuralNetOperator>(node);
    if (OpenCLCompatibleOperators.count(nnOp->getName())) {
      auto annotation = nnOp->getMutableAnnotation();
      NOM_REQUIRE_OR_RET(annotation);
      auto c2_annot = dyn_cast<Caffe2Annotation>(annotation);
      NOM_REQUIRE_OR_RET(c2_annot);
      c2_annot->setDeviceType(caffe2::DeviceTypeProto::PROTO_OPENCL);
      c2_annot->getMutableOperatorDef()->set_engine("FPGA");
    }
  }

  insertCopies(
      nn,
      [](NNGraph::NodeRef node) {
        if (nn::is<NeuralNetData>(node)) {
          return true;
        }
        auto nnOp = nn::get<NeuralNetOperator>(node);
        NOM_REQUIRE_OR_RET_FALSE(nnOp);
        auto annotation = nnOp->getAnnotation();
        NOM_REQUIRE_OR_RET_FALSE(annotation);
        auto c2_annot = dyn_cast<Caffe2Annotation>(annotation);
        NOM_REQUIRE_OR_RET_FALSE(c2_annot);
        auto device = c2_annot->getDeviceType();
        NOM_REQUIRE_OR_RET_FALSE(
            device == caffe2::DeviceTypeProto::PROTO_OPENCL);
        return true;
      },
      [](NNGraph& g) {
        auto copyTo = util::make_unique<CopyToOpenCL>();
        auto annot = make_unique<Caffe2Annotation>();
        annot->setDeviceType(caffe2::DeviceTypeProto::PROTO_OPENCL);
        OperatorDef op;
        op.set_type("CopyToOpenCL");
        op.set_engine("FPGA");
        annot->setOperatorDef(op);
        copyTo->setAnnotation(std::move(annot));
        return g.createNode(std::move(copyTo));
      },
      [](NNGraph& g) {
        auto copyTo = util::make_unique<CopyFromOpenCL>();
        auto annot = make_unique<Caffe2Annotation>();
        annot->setDeviceType(caffe2::DeviceTypeProto::PROTO_OPENCL);
        OperatorDef op;
        op.set_type("CopyFromOpenCL");
        op.set_engine("FPGA");
        annot->setOperatorDef(op);
        copyTo->setAnnotation(std::move(annot));
        return g.createNode(std::move(copyTo));
      });
}

REGISTER_OPT_PASS_FROM_FUNC(ConvertToOpenCL, convertToOpenCL);

} // namespace opt
} // namespace caffe2
