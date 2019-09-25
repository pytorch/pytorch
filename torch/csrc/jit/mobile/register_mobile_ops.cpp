#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;

void zeros_func(c10::OperatorKernel* kernel, Stack* stack) {
  const auto options = c10::TensorOptions()
                           .dtype((std::move(peek(*stack, 1, 5))).toOptional<c10::ScalarType>())
                           .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
                           .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
                           .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
  auto result_ = at::zeros((std::move(peek(*stack, 0, 5))).toIntListRef(), options);
  drop(*stack, 5);
  pack(*stack, std::move(result_));
}

void device_func(c10::OperatorKernel* kernel, Stack* stack) {
  std::string s = stack->back().toString()->string();
  auto result = c10::Device(s);
  drop(*stack, 1);
  pack(*stack, std::move(result));
}

void to_device_func(c10::OperatorKernel* kernel, Stack* stack) {
  auto result_ = ((std::move(peek(*stack, 0, 5))).toTensor()).to(
      (std::move(peek(*stack, 1, 5))).toDevice(),
      (std::move(peek(*stack, 2, 5))).toScalarType(),
      (std::move(peek(*stack, 3, 5))).toBool(),
      (std::move(peek(*stack, 4, 5))).toBool()
      );
  drop(*stack, 5);
  pack(*stack, std::move(result_));
}

void to_dtype_layout_func(c10::OperatorKernel* kernel, Stack* stack) {
  const auto options = c10::TensorOptions()
                           .dtype((std::move(peek(*stack, 1, 7))).toScalarType())
                           .layout((std::move(peek(*stack, 2, 7))).toLayout())
                           .device((std::move(peek(*stack, 3, 7))).toDevice())
                           .pinned_memory((std::move(peek(*stack, 4, 7))).toBool());;
  auto result_ = ((std::move(peek(*stack, 0, 7))).toTensor()).to(options,
                                                                (std::move(peek(*stack, 5, 7))).toBool(),
                                                                (std::move(peek(*stack, 6, 7))).toBool());
  drop(*stack, 7);
  pack(*stack, std::move(result_));
}

void empty_like_detype_func(c10::OperatorKernel* kernel, Stack* stack) {
  const auto options = c10::TensorOptions()
                           .dtype((std::move(peek(*stack, 1, 6))).toScalarType())
                           .layout((std::move(peek(*stack, 2, 6))).toLayout())
                           .device((std::move(peek(*stack, 3, 6))).toDevice())
                           .pinned_memory((std::move(peek(*stack, 4, 6))).toBool());
  auto result_ = at::empty_like((std::move(peek(*stack, 0, 6))).toTensor(),
                                options,
                                (std::move(peek(*stack, 5, 6))).toOptional<c10::MemoryFormat>());
  drop(*stack, 6);
  pack(*stack, std::move(result_));
}

void arange_start_step_func(c10::OperatorKernel* kernel, Stack* stack) {
  const auto options = c10::TensorOptions()
                           .dtype((std::move(peek(*stack, 3, 7))).toOptional<c10::ScalarType>())
                           .layout((std::move(peek(*stack, 4, 7))).toOptional<c10::Layout>())
                           .device((std::move(peek(*stack, 5, 7))).toOptional<c10::Device>())
                           .pinned_memory((std::move(peek(*stack, 6, 7))).toOptional<bool>());
  auto result_ = at::arange((std::move(peek(*stack, 0, 7))).toScalar(),
                            (std::move(peek(*stack, 1, 7))).toScalar(),
                            (std::move(peek(*stack, 2, 7))).toScalar(),
                            options);
  drop(*stack, 7);
  pack(*stack, std::move(result_));
}

void lstm_data_func(c10::OperatorKernel* kernel, Stack* stack) {
  auto result_ = at::lstm(
      (std::move(peek(*stack, 0, 9))).toTensor(),
      (std::move(peek(*stack, 1, 9))).toTensor(),
      (std::move(peek(*stack, 2, 9))).toTensorListRef(),
      (std::move(peek(*stack, 3, 9))).toTensorListRef(),
      (std::move(peek(*stack, 4, 9))).toBool(),
      (std::move(peek(*stack, 5, 9))).toInt(),
      (std::move(peek(*stack, 6, 9))).toDouble(),
      (std::move(peek(*stack, 7, 9))).toBool(),
      (std::move(peek(*stack, 8, 9))).toBool()
      );
  drop(*stack, 9);
  pack(*stack, std::move(result_));
}

void scatter__src_func(c10::OperatorKernel* kernel, Stack* stack) {
  auto self = (std::move(peek(*stack, 0, 4))).toTensor();
  auto result_ = (self).scatter_(
      (std::move(peek(*stack, 1, 4))).toInt(),
      (std::move(peek(*stack, 2, 4))).toTensor(),
      (std::move(peek(*stack, 3, 4))).toTensor()
      );
//auto result_ = torch::autograd::scatter_(const_cast<Tensor&>(*this), dim, index, src);  drop(*stack, 4);
  pack(*stack, std::move(result_));
}

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
).op(
"_aten::embedding",
torch::RegisterOperators::options().kernel<decltype(at::embedding), &at::embedding>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::dropout",
  torch::RegisterOperators::options().kernel<decltype(at::dropout), &at::dropout>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::device(str a) -> Device",
  torch::RegisterOperators::options().catchAllKernel(
    device_func
)
).op(
  "_aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor",
  torch::RegisterOperators::options().catchAllKernel(
    to_device_func
)
).op(
  "_aten::to.dtype_layout(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor",
  torch::RegisterOperators::options().catchAllKernel(
  to_dtype_layout_func
)
).op(
"_aten::sort",
torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
[](at::Tensor a, int64_t dim, bool descending = false) ->std::tuple<at::Tensor, at::Tensor> {
  return at::sort(a, dim, descending);
})
).op(
  "_aten::index_select",
  torch::RegisterOperators::options().kernel<decltype(at::index_select), &at::index_select>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::_pack_padded_sequence",
  torch::RegisterOperators::options().kernel<decltype(at::_pack_padded_sequence), &at::_pack_padded_sequence>(c10::TensorTypeId::CPUTensorId)
).op(
  "_aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  lstm_data_func
)
).op(
  "_aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  scatter__src_func
).aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
//).op(
//  "_aten::zeros",
//  torch::RegisterOperators::options().catchAllKernel(
//    [](c10::IntArrayRef size, c10::optional<int64_t> type,
//       c10::optional<c10::Layout> layout, c10::optional<c10::Device> device,
//       c10::optional<bool> pin_memory) ->at::Tensor {
////    auto dtype = type.has_value() ? c10::optional<c10::ScalarType>(static_cast<c10::ScalarType>(type.value()))
////                                  : c10::optional<c10::ScalarType>(c10::ScalarType::Undefined);
//auto dtype = c10::optional<c10::ScalarType>(c10::ScalarType::Int);
//    const auto options = c10::TensorOptions()
//                             .dtype(dtype)
//                             .layout(layout)
//                             .device(device)
//                             .pinned_memory(pin_memory);
//    return at::zeros(size, options);
//})
//).op(
//"_aten::zeros",
//torch::RegisterOperators::options().catchAllKernel(
//    [](c10::OperatorKernel* kernel, Stack* stack) {
//      const auto options = c10::TensorOptions()
//                               .dtype((std::move(peek(*stack, 1, 5))).toOptional<c10::ScalarType>())
//                               .layout((std::move(peek(*stack, 2, 5))).toOptional<c10::Layout>())
//                               .device((std::move(peek(*stack, 3, 5))).toOptional<c10::Device>())
//                               .pinned_memory((std::move(peek(*stack, 4, 5))).toOptional<bool>());
//      auto result_ = at::zeros_like((std::move(peek(*stack, 0, 5))).toIntListRef(), options);
//      drop(*stack, 5);
//      pack(*stack, std::move(result_));
//    })
).op(
"_aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
torch::RegisterOperators::options().catchAllKernel(
    zeros_func)
).op(
"_aten::empty_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=contiguous_format) -> Tensor",
torch::RegisterOperators::options().catchAllKernel(
    empty_like_detype_func)
).op(
"_aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
torch::RegisterOperators::options().catchAllKernel(
  arange_start_step_func)
);

