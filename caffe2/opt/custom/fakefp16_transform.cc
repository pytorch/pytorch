#include "caffe2/opt/custom/fakefp16_transform.h"

#include "caffe2/opt/custom/glow_net_transform.h"
#include "caffe2/utils/proto_utils.h"

C10_DEFINE_bool(
    fake_fp16_conversion_use_fp16_acc,
    false,
    "Whether to enable fp16 accumulation for FC / BatchMatMul for fakefp16 "
    "operators.");

C10_DEFINE_bool(
    fake_fp16_conversion_use_nnpi,
    false,
    "Whether to simulate NNPI behavior for fakefp16 operators.");

namespace caffe2 {
namespace opt {

std::unordered_map<std::string, std::string> getFakeFp16OpMapping(
    bool use_fp16_acc,
    bool use_nnpi) {
  std::unordered_map<std::string, std::string> fake_fp16_op_conversion_map = {
      {"FC", "Fp16FCAcc32NNPI"},
      {"Int8FC", "Int8FCFakeAcc32NNPI"},
      {"FbFCPacked", "Fp16FCAcc32NNPI"},
      {"SparseLengthsSum", "SparseLengthsSumFakeFP16AccFP16"},
      {"SparseLengthsWeightedSum", "SparseLengthsWeightedSumFakeFP16AccFP16"},
      {"SparseLengthsMean", "SparseLengthsMeanFakeFP16AccFP16"},
      {"SparseLengthsSumFused4BitRowwise",
       "SparseLengthsSumFused4BitRowwiseFakeFP16NNPI"},
      {"SparseLengthsWeightedSumFused4BitRowwise",
       "SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI"},
      {"SparseLengthsSumFused8BitRowwise",
       "SparseLengthsSumFused8BitRowwiseFakeFP16NNPI"},
      {"SparseLengthsWeightedSumFused8BitRowwise",
       "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI"},
      {"SparseLengthsMeanFused8BitRowwise",
       "SparseLengthsMeanFused8BitRowwiseFakeFP16AccFP16"},
      {"BatchMatMul", "BatchMatMulFP16Acc32Fake"},
      {"Sigmoid", "SigmoidFakeFp16"},
      {"SpatialBN", "SpatialBNFakeFp16NNPI"},
      {"Tanh", "TanhFakeFp16"},
      {"Relu", "ReluFakeFp16"},
      {"Add", "AddFakeFp16"},
      {"Sub", "SubFakeFp16"},
      {"Mul", "MulFakeFp16"},
      {"Div", "DivFakeFp16"},
      {"Sum", "SumFakeFp16"},
      {"Sqr", "SqrFakeFp16"},
      {"LengthsSum", "LengthsSumFakeFp16"}};
  if (use_fp16_acc) {
    fake_fp16_op_conversion_map["FC"] = "Fp16FCAcc16NNPI";
    fake_fp16_op_conversion_map["FbFCPacked"] = "Fp16FCAcc16NNPI";
    fake_fp16_op_conversion_map["BatchMatMul"] = "BatchMatMulFP16Acc16Fake";
  }
  if (use_nnpi) {
    fake_fp16_op_conversion_map["Sigmoid"] = "SigmoidFakeFp16NNPI";
    fake_fp16_op_conversion_map["Tanh"] = "TanhFakeFp16NNPI";
  }
  return fake_fp16_op_conversion_map;
}

void fakeFp16Transform(NetDef* net) {
  static const std::unordered_map<std::string, std::string>
      kFakeFp16OpConversionMap = getFakeFp16OpMapping(
          FLAGS_fake_fp16_conversion_use_fp16_acc,
          FLAGS_fake_fp16_conversion_use_nnpi);

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
