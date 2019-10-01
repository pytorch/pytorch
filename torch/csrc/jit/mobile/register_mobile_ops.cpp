#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;

at::Tensor optional_to_tensor(c10::optional<at::Tensor> v) {
  return v.has_value() ? *v : at::Tensor();
}

static auto registry0 = torch::RegisterOperators().op(
  "_aten::add.Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Tensor b, at::Scalar c) ->at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "_aten::add.Scalar",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Scalar b, at::Scalar c) ->at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "_aten::_convolution",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias,
  std::vector<int64_t> stride, std::vector<int64_t> padding,
  std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding,
  int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  return at::_convolution(input, weight, optional_to_tensor(bias), stride, padding, dilation,
                     transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  })
).op(
  // Dummy operator that does nothing. Used to reserve a location of an operator table.
  "_prim::ListConstruct.int",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.float",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.bool",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.tensor",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.generic",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
);

