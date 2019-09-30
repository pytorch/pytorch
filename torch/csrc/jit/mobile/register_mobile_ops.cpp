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
  // Dummy operator that does nothing. Used to reserve a location of an operator table.
  "_aten::dummy",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_aten::add_.Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Tensor b, at::Scalar c) ->at::Tensor {
  return at::add(a, b, c);
  })
).op(
  "_aten::adaptive_avg_pool2d",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, std::vector<int64_t> b) ->at::Tensor {
  return at::adaptive_avg_pool2d(a, b);
  })
).op(
  "_aten::mm",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Tensor b) ->at::Tensor {
  return at::mm(a, b);
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
  "_aten::conv2d",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias,
  std::vector<int64_t> stride, std::vector<int64_t> padding,
  std::vector<int64_t> dilation, int64_t groups) {
  return at::conv2d(input, weight, optional_to_tensor(bias), stride, padding, dilation, groups);
  })
).op(
  "_aten::batch_norm",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [] (at::Tensor input, c10::optional<at::Tensor> weight, c10::optional<at::Tensor> bias,
  c10::optional<at::Tensor> running_mean, c10::optional<at::Tensor> running_var,
  bool training, double momentum, double eps, bool cudnn_enabled) {
  return at::batch_norm(input, optional_to_tensor(weight), optional_to_tensor(bias),
               optional_to_tensor(running_mean), optional_to_tensor(running_var),
               training, momentum, eps, cudnn_enabled);
  })
).op(
  "_aten::max_pool2d_with_indices",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
  std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  return at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
  })
).op(
  "_aten::max_pool2d",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
  std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
  return at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  })
).op(
  "_aten::threshold",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor self, at::Scalar threshold, at::Scalar value) {
  return at::threshold_(self, threshold, value);
  })
).op(
  "_aten::relu",
  torch::RegisterOperators::options().kernel<decltype(at::relu), &at::relu>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::relu_",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a) ->at::Tensor {
  return at::relu_(a);
  })
).op(
  "_aten::t",
  torch::RegisterOperators::options().kernel<decltype(at::t), &at::t>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::size.int",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, int64_t dim) ->int64_t {
  return at::size(a, dim);
})
).op(
  "_aten::addmm",
  torch::RegisterOperators::options().kernel<decltype(at::addmm), &at::addmm>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::view",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, std::vector<int64_t> list) ->at::Tensor {
  return a.view(list);
})
).op(
  "_aten::dim",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a) ->int64_t {
  return a.dim();
  })
).op(
  "_aten::eq",
  torch::RegisterOperators::options().catchAllKernel(
  [](int64_t a, int64_t b) ->bool {
  return a == b;
  })
).op(
  "_aten::log_softmax",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, int64_t b, c10::optional<int64_t> c) ->at::Tensor {
  if (c.has_value()) {
  return at::log_softmax(a, b, static_cast<c10::ScalarType>(c.value()));
  } else {
  return at::log_softmax(a, b);
  }
  })
).op(
  "_aten::Int",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a) ->int64_t {
  return a.item<int64_t>();
  })
).op(
  "_prim::NumToTensor",
  torch::RegisterOperators::options().catchAllKernel(
  [](at::Scalar s) ->at::Tensor {
  return at::scalar_to_tensor(s);
  })
);

