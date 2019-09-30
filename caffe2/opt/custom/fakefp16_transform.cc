#include "caffe2/opt/custom/fakefp16_transform.h"

#include "caffe2/opt/custom/glow_net_transform.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace opt {

void fakeFp16Transform(NetDef* net) {
  static const std::unordered_map<std::string, std::string>
      kFakeFp16OpConversionMap = {
          {"FC", "Fp16FCAcc16NNPI"},
          {"SparseLengthsSum", "SparseLengthsSumFakeFP16AccFP16"},
          {"SparseLengthsWeightedSum",
           "SparseLengthsWeightedSumFakeFP16AccFP16"},
          {"SparseLengthsMean", "SparseLengthsMeanFakeFP16AccFP16"},
          {"SparseLengthsSumFused8BitRowwise",
           "SparseLengthsSumFused8BitRowwiseFakeFP16AccFP16"},
          {"SparseLengthsWeightedSumFused8BitRowwise",
           "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16AccFP16"},
          {"SparseLengthsMeanFused8BitRowwise",
           "SparseLengthsMeanFused8BitRowwiseFakeFP16AccFP16"},
          {"BatchMatMul", "BatchMatMulFP16Acc16Fake"},
          {"Sigmoid", "SigmoidFakeFp16"},
          {"Tanh", "TanhFakeFp16"},
          {"Relu", "ReluFakeFp16"},
          {"Add", "AddFakeFp16"},
          {"Sub", "SubFakeFp16"},
          {"Mul", "MulFakeFp16"},
          {"Div", "DivFakeFp16"},
          {"Sum", "SumFakeFp16"},
          {"Sqr", "SqrFakeFp16"},
          {"LengthsSum", "LengthsSumFakeFp16"}};

  auto blacklist_pos = glow::ParseNetPositionList(FLAGS_onnxifi_blacklist);
  auto blacklist_type = glow::ParseBlackListOps(FLAGS_onnxifi_blacklist_ops);

  // A hack to only do fakefp16 transformation for operators which will be
  // lowered to ONNXIFI.
  // TODO(yingz): Use more deterministic logics to figure out operators which
  // can be lowered to ONNXIFI instead.
  int last_clip_idx = -1;
  for (int i = 0; i < net->op().size(); ++i) {
    const auto& op = net->op(i);
    if (op.type() == "Clip") {
      last_clip_idx = i;
    }
  }
  for (int i = 0; i < net->op().size(); ++i) {
    if (i <= last_clip_idx) {
      continue;
    }
    auto* op = net->mutable_op(i);
    auto net_pos =
        ArgumentHelper::GetSingleArgument<OperatorDef, int>(*op, "net_pos", -1);
    if (blacklist_pos.count(net_pos) || blacklist_type.count(op->type())) {
      continue;
    }
    auto it = kFakeFp16OpConversionMap.find(op->type());
    if (it != kFakeFp16OpConversionMap.end()) {
      op->set_type(it->second);
    }
  }
}

} // namespace opt
} // namespace caffe2
