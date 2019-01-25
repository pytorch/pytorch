#include <jit/custom_operator.h>

static auto ln_reg = torch::jit::RegisterOperators::Caffe2Operator("LayerNorm");
