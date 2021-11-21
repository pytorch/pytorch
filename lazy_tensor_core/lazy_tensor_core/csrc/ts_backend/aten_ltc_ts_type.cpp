#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include "ATen/MetaFunctions.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/random.h"
#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyNativeFunctions.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_autograd_ops_ts.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h"
#include "lazy_tensor_core/csrc/view_ops/as_strided.h"
#include "lazy_tensors/computation_client/metrics.h"
namespace torch_lazy_tensors {
namespace ir {
namespace ops {
// TODO(whc) forward declare these since they aren't defined in the autogenned
// header; this will be solved when moving cat() to codegen
std::vector<torch::lazy::Shape> compute_shape_cat(at::TensorList tensors,
                                                    int64_t dim);
}  // namespace ops
}  // namespace ir

namespace {

void CheckSubOperandTypes(at::ScalarType type1, at::ScalarType type2) {
  CHECK(type1 != at::kBool || type2 != at::kBool)
      << "Subtraction, the `-` operator, with two bool tensors is not "
         "supported. Use the `^` or `logical_xor()` operator instead.";
  CHECK(type1 != at::kBool && type2 != at::kBool)
      << "Subtraction, the `-` operator, with a bool tensor is not "
         "supported. If you are trying to invert a mask, use the `~` or "
         "`logical_not()` operator instead.";
}

std::pair<LazyTensor, LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  LazyTensor self_tensor;
  LazyTensor other_tensor;
  auto self_xtensor = TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = TryGetLtcTensor(other);
    self_tensor = GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = self_xtensor;
    other_tensor = GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<LazyTensor, LazyTensor>(self_tensor, other_tensor);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<LazyTensor, LazyTensor> operands =
      GetBinaryOperands(UnwrapNumber(self, dtype), UnwrapNumber(other, dtype));
  LazyTensor result = bin_op(operands.first, operands.second);
  return CreateAtenFromLtcTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  LazyTensor self_tensor = GetLtcTensor(self);
  LazyTensor result = bin_op(self_tensor, other);
  return CreateAtenFromLtcTensor(result);
}

at::Tensor subtensor(const at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

at::Tensor CreateLtcTensor(const at::Tensor& tensor,
                           const c10::optional<torch::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    return CreateAtenFromLtcTensor(LazyTensor::Create(tensor, *device));
  }
  return tensor;
}

}  // namespace

at::Tensor LazyNativeFunctions::alias(const at::Tensor& self) {
  LTC_FN_COUNTER("lazy::");
  return self;
}

at::Tensor LazyNativeFunctions::as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP(as_strided)>::call(self, size, stride,
                                                        storage_offset);
  }
  return CreateAtenFromLtcTensor(lazy_tensor_aten_ops::as_strided(
      self_tensor, std::move(xsize), std::move(xstride),
      Helpers::I64Optional(storage_offset)));
}

const at::Tensor& LazyNativeFunctions::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP(as_strided_)>::call(self, size, stride,
                                                         storage_offset);
  }
  lazy_tensor_aten_ops::as_strided_(self_tensor, std::move(xsize),
                                    std::move(xstride),
                                    Helpers::I64Optional(storage_offset));
  return self;
}

at::Tensor LazyNativeFunctions::bernoulli(
    const at::Tensor& self, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("lazy::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&ltc_eager_fallback,
                                        ATEN_OP(bernoulli)>::call(self,
                                                                  generator);
  }
  LazyTensor self_tensor = TryGetLtcTensor(self);
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::bernoulli(self_tensor));
}

at::Tensor& LazyNativeFunctions::bernoulli_(
    at::Tensor& self, double p, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("lazy::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP2(bernoulli_, float)>::call(self, p,
                                                                generator);
  }
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor LazyNativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("lazy::");
  auto lazy_tensors = GetLtcTensors(tensors);
  std::vector<torch::lazy::Value> values;
  values.reserve(lazy_tensors.size());
  for (auto& tensor : lazy_tensors) {
    values.emplace_back(tensor.GetIrValue());
  }

  auto shapes =
      torch_lazy_tensors::ir::ops::compute_shape_cat(tensors, dim);
  auto node =
      torch::lazy::MakeNode<ir::ops::Cat>(values, dim, std::move(shapes));
  auto result = CreateAtenFromLtcTensor(
      lazy_tensors[0].CreateFrom(torch::lazy::Value(node, 0)));
  return result;
}

at::Tensor LazyNativeFunctions::clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  auto self_lt = TryGetLtcTensor(self);
  return CreateAtenFromLtcTensor(self_lt.Create(self_lt.GetIrValue(), self_lt.GetDevice()));
}

at::Tensor LazyNativeFunctions::constant_pad_nd(const at::Tensor& self,
                                                at::IntArrayRef pad,
                                                const at::Scalar& value) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(lazy_tensor_aten_ops::constant_pad_nd(
      TryGetLtcTensor(self), Helpers::I64List(pad), value));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  // Lower to cudnn_convolution_backward when possbile
  if (at::globalContext().userEnabledCuDNN() &&
      torch::lazy::getBackend()->EagerFallbackDeviceType() == at::kCUDA) {
    LTC_FN_COUNTER("lazy::");
    auto result = lazy_tensor_aten_ops::convolution_backward_overrideable(
        TryGetLtcTensor(grad_output), TryGetLtcTensor(input),
        TryGetLtcTensor(weight), Helpers::I64List(stride),
        Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups, std::move(output_mask));
    return std::make_tuple(CreateAtenFromLtcTensor(std::get<0>(result)),
                           CreateAtenFromLtcTensor(std::get<1>(result)),
                           CreateAtenFromLtcTensor(std::get<2>(result)));
  }
  // Fallback otherwise
  if (groups > 1) {
    std::vector<at::Tensor> grad_input(groups);
    std::vector<at::Tensor> grad_weight(groups);
    std::vector<at::Tensor> grad_bias(groups);
    for (int g = 0; g < groups; ++g) {
      auto grad_output_g = subtensor(grad_output, 1, groups, g);
      auto input_g = subtensor(input, 1, groups, g);
      auto weight_g = subtensor(weight, 0, groups, g);
      auto x_result = torch_lazy_tensors::LazyNativeFunctions::
          convolution_backward_overrideable(
              grad_output_g, input_g, weight_g, stride, padding, dilation,
              transposed, output_padding, 1, output_mask);
      grad_input[g] = std::get<0>(x_result);
      grad_weight[g] = std::get<1>(x_result);
      grad_bias[g] = std::get<2>(x_result);
    }
    return {at::cat(grad_input, 1), at::cat(grad_weight, 0),
            grad_bias[0].defined() ? at::cat(grad_bias, 0) : grad_bias[0]};
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::convolution_backward_overrideable", 1);
  VLOG(3) << "LTC-TS convolution_backward_overrideable :"
          << " grad_output=" << grad_output.toString()
          << " input=" << input.toString() << " weight=" << weight.toString();
  const auto kernel_size = weight.sizes().slice(2);
  CHECK(kernel_size.size() == 2 || kernel_size.size() == 3);
  const at::DeviceType device_type =
      torch::lazy::getBackend()->EagerFallbackDeviceType();
  auto backend_device = bridge::GetSameBackendDeviceOrUseDefault(grad_output);
  if (transposed) {
    at::TensorOptions options = at::TensorOptions().device(device_type);
    auto&& x_result =
        kernel_size.size() == 2
            ? at::slow_conv_transpose2d_backward(
                  grad_output.to(device_type), input.to(device_type),
                  weight.to(device_type), kernel_size, stride, padding,
                  output_padding, dilation,
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Contiguous),
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Contiguous),
                  output_mask)
            : at::slow_conv_transpose3d_backward(
                  grad_output.to(device_type), input.to(device_type),
                  weight.to(device_type), kernel_size, stride, padding,
                  output_padding, dilation,
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Preserve),
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Preserve),
                  output_mask);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        CreateLtcTensor(std::get<0>(x_result),
                                bridge::GetLtcDevice(grad_output)),
        CreateLtcTensor(std::get<1>(x_result),
                                bridge::GetLtcDevice(grad_output)),
        CreateLtcTensor(std::get<2>(x_result),
                                bridge::GetLtcDevice(grad_output)));
  }
  auto&& x_result =
      kernel_size.size() == 2
          ? at::slow_conv_dilated2d_backward(
                grad_output.to(device_type), input.to(device_type),
                weight.to(device_type), kernel_size, stride, padding, dilation,
                output_mask)
          : at::slow_conv_dilated3d_backward(
                grad_output.to(device_type), input.to(device_type),
                weight.to(device_type), kernel_size, stride, padding, dilation,
                output_mask);
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      CreateLtcTensor(std::get<0>(x_result),
                              bridge::GetLtcDevice(grad_output)),
      CreateLtcTensor(std::get<1>(x_result),
                              bridge::GetLtcDevice(grad_output)),
      CreateLtcTensor(std::get<2>(x_result),
                              bridge::GetLtcDevice(grad_output)));
}

at::Tensor LazyNativeFunctions::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  LTC_FN_COUNTER("lazy::");
  return (bias && bias->defined())
             ? CreateAtenFromLtcTensor(
                   lazy_tensor_aten_ops::convolution_overrideable(
                       TryGetLtcTensor(input),
                       TryGetLtcTensor(weight),
                       TryGetLtcTensor(*bias), Helpers::I64List(stride),
                       Helpers::I64List(padding), Helpers::I64List(dilation),
                       transposed, Helpers::I64List(output_padding), groups))
             : CreateAtenFromLtcTensor(
                   lazy_tensor_aten_ops::convolution_overrideable(
                       TryGetLtcTensor(input),
                       TryGetLtcTensor(weight), Helpers::I64List(stride),
                       Helpers::I64List(padding), Helpers::I64List(dilation),
                       transposed, Helpers::I64List(output_padding), groups));
}

at::Tensor LazyNativeFunctions::_copy_from(const at::Tensor& self,
                                           const at::Tensor& dst,
                                           bool non_blocking) {
  LTC_FN_COUNTER("lazy::");
  auto dst_tensor = TryGetLtcTensor(dst);
  auto self_tensor = TryGetLtcTensor(self);
  if (!self_tensor) {
    static bool sync_update =
        lazy_tensors::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    CHECK(dst_tensor);
    dst_tensor.UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor.ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    if (!dst_tensor.CurrentIrValue()) {
      auto dst_tensor_data = dst_tensor.CurrentTensorData();
      CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor.CurrentTensorData();
      if (src_tensor_data) {
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        dst_tensor_data->copy_(self_tensor.ToTensor(/*detached=*/true));
      }
    } else {
      lazy_tensor_aten_ops::copy_(dst_tensor, self_tensor);
      auto* impl = dynamic_cast<LTCTensorImpl*>(dst.unsafeGetTensorImpl());
      impl->set_tensor(dst_tensor);
    }
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_copy_from_and_resize(const at::Tensor& self,
                                                      const at::Tensor& dst) {
  LTC_FN_COUNTER("lazy::");
  auto dst_tensor = TryGetLtcTensor(dst);
  auto self_tensor = TryGetLtcTensor(self);
  if (!self_tensor) {
    CHECK(dst_tensor);
    dst_tensor.UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor.ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is a lazy tensor
    LTCTensorImpl* dest_impl =
        dynamic_cast<LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor().UpdateFromTensorOut(self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

at::Tensor LazyNativeFunctions::empty(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  const auto device_type = torch::lazy::getBackend()->EagerFallbackDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  return CreateLtcTensor(x_result, bridge::GetLtcDevice(device));
}

at::Tensor LazyNativeFunctions::empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("lazy::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return torch_lazy_tensors::LazyNativeFunctions::as_strided(
      t, size, stride, /*storage_offset=*/0);
}

at::Tensor LazyNativeFunctions::expand(const at::Tensor& self,
                                       at::IntArrayRef size, bool implicit) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(lazy_tensor_aten_ops::expand(
      TryGetLtcTensor(self), size.vec()));
}

at::Tensor& LazyNativeFunctions::fill_(at::Tensor& self,
                                       const at::Scalar& value) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::fill_(self_tensor, value);
  return self;
}

at::Tensor LazyNativeFunctions::max_pool2d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool2dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor LazyNativeFunctions::max_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool3dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor LazyNativeFunctions::mul(const at::Tensor& self,
                                    const at::Tensor& other) {
  LTC_FN_COUNTER("lazy::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother) {
                      return lazy_tensor_aten_ops::mul(xself, xother);
                    });
}

at::Tensor LazyNativeFunctions::mul(const at::Tensor& self,
                                    const at::Scalar& other) {
  LTC_FN_COUNTER("lazy::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other) {
                      return lazy_tensor_aten_ops::mul(xself, other);
                    });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor input_tensor = TryGetLtcTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor.GetDevice();
  LazyTensor running_mean_tensor =
      GetOrCreateLtcTensor(running_mean, device);
  LazyTensor running_var_tensor =
      GetOrCreateLtcTensor(running_var, device);
  auto outputs = lazy_tensor_aten_ops::ts_native_batch_norm(
      TryGetLtcTensor(input), GetOrCreateLtcTensor(weight, device),
      GetOrCreateLtcTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(CreateAtenFromLtcTensor(std::get<0>(outputs)),
                         CreateAtenFromLtcTensor(std::get<1>(outputs)),
                         CreateAtenFromLtcTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor grad_out_tensor = TryGetLtcTensor(grad_out);
  const torch::lazy::BackendDevice& device = grad_out_tensor.GetDevice();
  LazyTensor null_tensor;
  bool running_stats = running_mean && running_mean->defined();
  CHECK_EQ(running_var && running_var->defined(), running_stats);
  auto gradients = lazy_tensor_aten_ops::ts_native_batch_norm_backward(
      TryGetLtcTensor(grad_out), TryGetLtcTensor(input),
      GetOrCreateLtcTensor(weight, device),
      running_stats ? GetOrCreateLtcTensor(running_mean, device)
                    : null_tensor,
      running_stats ? GetOrCreateLtcTensor(running_var, device)
                    : null_tensor,
      GetOrCreateLtcTensor(save_mean, device),
      GetOrCreateLtcTensor(save_invstd, device), train, eps,
      output_mask);
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? CreateAtenFromLtcTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? CreateAtenFromLtcTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? CreateAtenFromLtcTensor(std::get<2>(gradients))
                     : undefined);
}

// We need to explicitly override max pooling operators and just call the
// fallback for them because we've customized the autograd function for them
// (backward needs saved indices from forward).

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return at::native::call_fallback_fn<
      &ltc_eager_fallback, ATEN_OP(max_pool2d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
}

at::Tensor LazyNativeFunctions::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  return at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output, self,
                                                       kernel_size, stride,
                                                       padding, dilation,
                                                       ceil_mode, indices);
}

std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return at::native::call_fallback_fn<
      &ltc_eager_fallback, ATEN_OP(max_pool3d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
}

at::Tensor LazyNativeFunctions::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  return at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output, self,
                                                       kernel_size, stride,
                                                       padding, dilation,
                                                       ceil_mode, indices);
}

at::Tensor LazyNativeFunctions::permute(const at::Tensor& self,
                                        at::IntArrayRef dims) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::permute(self_tensor, Helpers::I64List(dims)));
}

at::Tensor& LazyNativeFunctions::random_(
    at::Tensor& self, int64_t from, c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("lazy::");

  if (generator && generator->defined()) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP2(random_, from)>::call(self, from, to,
                                                            generator);
  }

  auto selfTensor = TryGetLtcTensor(self);
  selfTensor.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor.GetIrValue(), from, to));
  return self;
}

at::Tensor& LazyNativeFunctions::random_(
    at::Tensor& self, int64_t to, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("lazy::");

  if (generator && generator->defined()) {
    return at::native::call_fallback_fn<&ltc_eager_fallback,
                                        ATEN_OP2(random_, to)>::call(self, to,
                                                                     generator);
  }

  auto selfTensor = TryGetLtcTensor(self);
  selfTensor.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor.GetIrValue(), c10::nullopt, to));
  return self;
}

at::Tensor& LazyNativeFunctions::random_(
    at::Tensor& self, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("lazy::");

  if (generator && generator->defined()) {
    return at::native::call_fallback_fn<&ltc_eager_fallback,
                                        ATEN_OP(random_)>::call(self,
                                                                generator);
  }

  auto selfTensor = TryGetLtcTensor(self);
  selfTensor.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor.GetIrValue(), c10::nullopt, c10::nullopt));
  return self;
}

at::Tensor LazyNativeFunctions::repeat(const at::Tensor& self,
                                       at::IntArrayRef repeats) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(lazy_tensor_aten_ops::repeat(
      TryGetLtcTensor(self), Helpers::I64List(repeats)));
}

at::Tensor LazyNativeFunctions::select(const at::Tensor& self, int64_t dim,
                                       int64_t index) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::select(TryGetLtcTensor(self), dim, index));
}

at::Tensor LazyNativeFunctions::slice(const at::Tensor& self, int64_t dim,
                                      c10::optional<int64_t> start,
                                      c10::optional<int64_t> end,
                                      int64_t step) {
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(lazy_tensor_aten_ops::slice(
      TryGetLtcTensor(self), dim, start_val, end_val, step));
}

at::Tensor LazyNativeFunctions::stack(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::stack(GetLtcTensors(tensors), dim));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::squeeze(TryGetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::squeeze(TryGetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::squeeze_(self_tensor);
  return self;
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::squeeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::sub(const at::Tensor& self,
                                    const at::Tensor& other,
                                    const at::Scalar& alpha) {
  LTC_FN_COUNTER("lazy::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother) {
                      return lazy_tensor_aten_ops::sub(xself, xother, alpha);
                    });
}

at::Tensor LazyNativeFunctions::sub(const at::Tensor& self,
                                    const at::Scalar& other,
                                    const at::Scalar& alpha) {
  LTC_FN_COUNTER("lazy::");
  CheckSubOperandTypes(self.scalar_type(), other.type());
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other) {
                      return lazy_tensor_aten_ops::sub(xself, other, alpha);
                    });
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::transpose(TryGetLtcTensor(self), 0, 1));
}

at::Tensor& LazyNativeFunctions::t_(at::Tensor& self) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor LazyNativeFunctions::tanh_backward(const at::Tensor& grad_output,
                                              const at::Tensor& output) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(lazy_tensor_aten_ops::tanh_backward(
      TryGetLtcTensor(grad_output), TryGetLtcTensor(output)));
}

at::Tensor LazyNativeFunctions::transpose(const at::Tensor& self, int64_t dim0,
                                          int64_t dim1) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::transpose(TryGetLtcTensor(self), dim0, dim1));
}

at::Tensor& LazyNativeFunctions::transpose_(at::Tensor& self, int64_t dim0,
                                            int64_t dim1) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::transpose_(self_tensor, dim0, dim1);
  return self;
}

at::Tensor LazyNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("lazy::");
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::unsqueeze(TryGetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::unsqueeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  lazy_tensor_aten_ops::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  LTC_FN_COUNTER("lazy::");
  LazyTensor self_tensor = TryGetLtcTensor(self);
  return CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::view(self_tensor, Helpers::I64List(size)));
}

void InitializeAtenBindings() {}

}  // namespace torch_lazy_tensors
