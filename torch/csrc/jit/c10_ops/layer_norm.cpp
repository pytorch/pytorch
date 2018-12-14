#include <c10/core/dispatch/Dispatcher.h>
#include <c10/core/op_schemas/layer_norm.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/autograd/variable.h>
#include <caffe2/core/context.h>

using c10::C10Tensor;

namespace {
at::Tensor layer_norm(
    at::Tensor input,
    int64_t axis,
    double epsilon) {
  if (input.requires_grad()) {
    throw std::runtime_error("Autograd not yet supported for c10 ops.");
  }
  c10::core::ops::LayerNorm::Cache cache;
  caffe2::CPUContext context;
  C10Tensor c10_input = torch::autograd::Variable(input).data();
  C10Tensor c10_output = at::empty({0});
  C10Tensor c10_output_mean = at::empty({0});
  C10Tensor c10_output_stdev = at::empty({0});
  c10::Dispatcher<c10::core::ops::LayerNorm>::call(c10_input, c10_output, c10_output_mean, c10_output_stdev, (int)axis, (float)epsilon, &cache, static_cast<caffe2::BaseContext*>(&context));
  return torch::autograd::make_variable(c10_output, false);
}
}

static auto registry =
  torch::jit::RegisterOperators("caffe2::layer_norm_dont_use_this_op_yet", &layer_norm);
