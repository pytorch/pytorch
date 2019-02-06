#include <jit/custom_operator.h>
#include "caffe2/operators/layer_norm_op.h"
#include "caffe2/operators/roi_align_op.h"
#include "caffe2/operators/generate_proposals_op.h"

#define REGISTER_CAFFE2_OP(name) \
  static caffe2::CAFFE2_STRUCT_OP_REGISTRATION_##name CAFFE2_STRUCT_OP_REGISTRATION_DEFN_TORCH_##name; \
  static auto CAFFE2_OP_EXPORT_##name = torch::jit::RegisterOperators::Caffe2Operator(#name);

REGISTER_CAFFE2_OP(LayerNorm);
REGISTER_CAFFE2_OP(RoIAlign);
REGISTER_CAFFE2_OP(GenerateProposals);
