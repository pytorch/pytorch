#include <ATen/core/op_registration/op_registration.h>
#include "torch/csrc/jit/custom_operator.h"
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
//#include <c10/core/TensorTypeId.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;

at::Tensor optional_to_tensor(c10::optional<at::Tensor> v) {
  return v.has_value() ? *v : at::Tensor();
}

static auto registry0 = torch::RegisterOperators().op(
  "aten::matmul",
  torch::RegisterOperators::options().kernel<decltype(at::matmul), &at::matmul>(c10::TensorTypeId::CPUTensorId)
).op(
  "aten::add.Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Tensor b, at::Scalar c) ->at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "aten::add.Scalar",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Scalar b, at::Scalar c) ->at::Tensor {
    return at::add(a, b, c);
  })
//).op(
//  "aten::adaptive_avg_pool2d_Tensor_int[]__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor a, std::vector<int64_t> b) ->at::Tensor {
//    return at::adaptive_avg_pool2d(a, b);
//  })
//).op(
//  "aten::mm_Tensor_Tensor__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor a, at::Tensor b) ->at::Tensor {
//    return at::mm(a, b);
//  })
//).op(
//  "aten::_convolution_Tensor_Tensor_Tensor?_int[]_int[]_int[]_bool_int[]_int_bool_bool_bool__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias,
//    std::vector<int64_t> stride, std::vector<int64_t> padding,
//    std::vector<int64_t> dilation, bool transposed, std::vector<int64_t> output_padding,
//    int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
//  return at::_convolution(input, weight, optional_to_tensor(bias), stride, padding, dilation,
//                          transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
//  })
//).op(
//  "aten::conv2d_Tensor_Tensor_Tensor?_int[]_int[]_int[]_int__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias,
//    std::vector<int64_t> stride, std::vector<int64_t> padding,
//    std::vector<int64_t> dilation, int64_t groups) {
//  return at::conv2d(input, weight, optional_to_tensor(bias), stride, padding, dilation, groups);
//  })
//).op(
//  "aten::batch_norm_Tensor_Tensor?_Tensor?_Tensor?_Tensor?_bool_float_float_bool__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [] (at::Tensor input, c10::optional<at::Tensor> weight, c10::optional<at::Tensor> bias,
//    c10::optional<at::Tensor> running_mean, c10::optional<at::Tensor> running_var,
//    bool training, double momentum, double eps, bool cudnn_enabled) {
//  return at::batch_norm(input, optional_to_tensor(weight), optional_to_tensor(bias),
//                        optional_to_tensor(running_mean), optional_to_tensor(running_var),
//                        training, momentum, eps, cudnn_enabled);
//  })
//).op(
//  "aten::max_pool2d_with_indices_Tensor_int[]_int[]_int[]_int[]_bool__Tensor_Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
//  std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
//    return at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
//  })
//).op(
//  "aten::max_pool2d_Tensor_int[]_int[]_int[]_int[]_bool__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor self, std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
//  std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
//  return at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
//  })
//).op(
//  "aten::threshold__Tensor_Scalar_Scalar__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor self, at::Scalar threshold, at::Scalar value) {
//    return at::threshold_(self, threshold, value);
//  })
//).op(
//  "aten::relu_Tensor__Tensor",
//  torch::RegisterOperators::options().kernel<decltype(at::relu), &at::relu>(c10::TensorTypeId::CPUTensorId)
//).op(
//  "aten::relu__Tensor__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor a) ->at::Tensor {
//    return at::relu_(a);
//  })
//).op(
//  "aten::t_Tensor__Tensor",
//  torch::RegisterOperators::options().kernel<decltype(at::t), &at::t>(c10::TensorTypeId::CPUTensorId)
//).op(
//  "aten::size_Tensor_int__int",
//  torch::RegisterOperators::options().kernel<decltype(at::size), &at::size>(c10::TensorTypeId::CPUTensorId)
//).op(
//  "aten::view_Tensor_int[]__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor a, std::vector<int64_t> list) ->at::Tensor {
//    return a.view(list);
//  })
//).op(
//  "aten::dim_Tensor__int",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor a) ->int64_t {
//  return a.dim();
//  })
//).op(
//  "aten::eq_int_int__bool",
//  torch::RegisterOperators::options().catchAllKernel(
//  [](int64_t a, int64_t b) ->bool {
//  return a == b;
//  })
//).op(
//  "aten::log_softmax_Tensor_int__Tensor",
//  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
//  [](at::Tensor a, int64_t b) ->at::Tensor {
//  return at::log_softmax(a, b);                                                                        })
//).op(
//  "prim::Load___",
//  torch::RegisterOperators::options().catchAllKernel([]() {
//  })
).op(
  "prim::Store___",
  torch::RegisterOperators::options().catchAllKernel([]() {
  })
);


//class MyKernel : public OperatorKernel {
// public:
//  MyKernel(int value): value_(value) {}
//  int operator()() {
//    return value_;
//  }
//};

//static auto registry2 = c10::RegisterOperators().op(
//    "aten::constant6",
//    torch::RegisterOperators::options().kernel<MyKernel>(6)
//);
