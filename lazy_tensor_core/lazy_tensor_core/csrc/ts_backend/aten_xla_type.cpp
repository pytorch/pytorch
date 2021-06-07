#include "lazy_tensor_core/csrc/ts_backend/aten_xla_type.h"

#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_autograd_ops_ts.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_xla_type_default.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/nnc_computation_client.h"

namespace torch_lazy_tensors {
namespace {

std::pair<LazyTensor, LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  LazyTensor self_tensor;
  LazyTensor other_tensor;
  auto self_xtensor = bridge::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::GetLtcTensor(other);
    self_tensor = bridge::GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = *self_xtensor;
    other_tensor = bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<LazyTensor, LazyTensor>(self_tensor, other_tensor);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<LazyTensor, LazyTensor> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  LazyTensor result = bin_op(operands.first, operands.second, dtype);
  return bridge::AtenFromLtcTensor(result);
}

}  // namespace

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Tensor& other,
                            const at::Scalar& alpha) {
  LTC_FN_COUNTER("xla::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenXlaType::alias(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return self;
}

at::Tensor AtenXlaType::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                   at::IntArrayRef stride,
                                   c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided(self, size, stride, storage_offset);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::as_strided(self_tensor, std::move(xsize), std::move(xstride),
                             Helpers::I64Optional(storage_offset)));
}

const at::Tensor& AtenXlaType::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided_(self, size, stride, storage_offset);
  }
  LazyTensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                          Helpers::I64Optional(storage_offset));
  return self;
}

at::Tensor AtenXlaType::constant_pad_nd(const at::Tensor& self,
                                        at::IntArrayRef pad,
                                        const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::constant_pad_nd(
      bridge::GetLtcTensor(self), Helpers::I64List(pad), value));
}

at::Tensor AtenXlaType::_copy_from(const at::Tensor& self,
                                   const at::Tensor& dst, bool non_blocking) {
  LTC_FN_COUNTER("xla::");
  auto dst_tensor = bridge::TryGetLtcTensor(dst);
  auto self_tensor = bridge::TryGetLtcTensor(self);
  if (!self_tensor) {
    static bool sync_update =
        lazy_tensors::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    LTC_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    if (!dst_tensor->CurrentIrValue()) {
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      LTC_CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/true));
      }
    } else {
      LazyTensor::copy_(*dst_tensor, *self_tensor);
      bridge::ReplaceLtcTensor(dst, *dst_tensor);
    }
  }
  return dst;
}

at::Tensor AtenXlaType::empty(at::IntArrayRef size,
                              c10::optional<at::ScalarType> dtype,
                              c10::optional<at::Layout> layout,
                              c10::optional<at::Device> device,
                              c10::optional<bool> pin_memory,
                              c10::optional<at::MemoryFormat> memory_format) {
  const auto device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(device));
}

at::Tensor AtenXlaType::empty_strided(at::IntArrayRef size,
                                      at::IntArrayRef stride,
                                      c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout,
                                      c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("xla::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor AtenXlaType::expand(const at::Tensor& self, at::IntArrayRef size,
                               bool implicit) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::expand(
      bridge::GetLtcTensor(self),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(size)));
}

at::Tensor AtenXlaType::gelu(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::gelu(bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::gelu_backward(const at::Tensor& grad,
                                      const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::gelu_backward(
      bridge::GetLtcTensor(grad), bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::max_pool2d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool2dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool3d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool3dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor& AtenXlaType::normal_(at::Tensor& self, double mean, double std,
                                 c10::optional<at::Generator> generator) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::normal_", 1);
  LTC_VLOG(3) << "TS normal_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  auto&& x_result = xlatens[0].normal_(mean, std, generator);
  std::vector<size_t> xlatens_update_indices = {0};
  bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  static_cast<void>(x_result);  // Avoid warnings in case not used
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return AtenXlaTypeDefault::max_pool2d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  return AtenXlaTypeDefault::max_pool2d_with_indices_backward(
      grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
      indices);
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return AtenXlaTypeDefault::max_pool3d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  return AtenXlaTypeDefault::max_pool3d_with_indices_backward(
      grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
      indices);
}

at::Tensor AtenXlaType::permute(const at::Tensor& self, at::IntArrayRef dims) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::permute(self_tensor, Helpers::I64List(dims)));
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  LTC_VLOG(3) << "XLA squeeze_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_();
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens,
                                 xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  LTC_VLOG(3) << "XLA squeeze_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_(dim);
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens,
                                 xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor AtenXlaType::t(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::GetLtcTensor(self), 0, 1));
}

at::Tensor& AtenXlaType::t_(at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor AtenXlaType::transpose(const at::Tensor& self, int64_t dim0,
                                  int64_t dim1) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::GetLtcTensor(self), dim0, dim1));
}

at::Tensor& AtenXlaType::transpose_(at::Tensor& self, int64_t dim0,
                                    int64_t dim1) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

at::Tensor AtenXlaType::unsqueeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::unsqueeze(bridge::GetLtcTensor(self), dim));
}

at::Tensor& AtenXlaType::unsqueeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenXlaType::view(const at::Tensor& self, at::IntArrayRef size) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::view(self_tensor, Helpers::I64List(size)));
}

void AtenXlaType::InitializeAtenBindings() {}

}  // namespace torch_lazy_tensors
