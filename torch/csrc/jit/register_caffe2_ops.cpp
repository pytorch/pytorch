#include <jit/custom_operator.h>

#define REGISTER_CAFFE2_OP(name) \
  static caffe2::CAFFE2_STRUCT_OP_REGISTRATION_##name CAFFE2_STRUCT_OP_REGISTRATION_DEFN_TORCH_##name; \
  static auto CAFFE2_OP_EXPORT_##name = torch::jit::RegisterOperators::Caffe2Operator(#name);
