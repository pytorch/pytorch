#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/library.h>

#include "ATen/MetaFunctions.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/random.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include <torch/csrc/lazy/core/tensor_impl.h>
#include "lazy_tensor_core/csrc/ts_backend/LazyNativeFunctions.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_autograd_ops_ts.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h"
#include "lazy_tensors/computation_client/sys_util.h"
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

std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  torch::lazy::LazyTensor self_tensor;
  torch::lazy::LazyTensor other_tensor;
  auto self_xtensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = torch::lazy::TryGetLtcTensor(other);
    self_tensor = GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = self_xtensor;
    other_tensor = GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor>(self_tensor, other_tensor);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor> operands =
      GetBinaryOperands(torch::lazy::UnwrapNumber(self, dtype),
                        torch::lazy::UnwrapNumber(other, dtype));
  torch::lazy::LazyTensor result = bin_op(operands.first, operands.second);
  return torch::lazy::CreateAtenFromLtcTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  torch::lazy::LazyTensor self_tensor = torch::lazy::GetLtcTensor(self);
  torch::lazy::LazyTensor result = bin_op(self_tensor, other);
  return torch::lazy::CreateAtenFromLtcTensor(result);
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
    return torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(tensor, *device));
  }
  return tensor;
}

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::optional<c10::Device>& device) {
  if (!device) {
    return c10::nullopt;
  }
  if (device->type() != at::kLazy) {
    return c10::nullopt;
  }
  return torch::lazy::atenDeviceToBackendDevice(*device);
}

}  // namespace

at::Tensor LazyNativeFunctions::alias(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return self;
}

at::Tensor LazyNativeFunctions::as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  auto xsize = torch::lazy::ToI64Vector(size);
  auto xstride = torch::lazy::ToI64Vector(stride);
  if (!torch::lazy::AsStrided::StrideIsSupported(xstride)) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP(as_strided)>::call(self, size, stride,
                                                        storage_offset);
  }
  return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::as_strided(
      self_tensor, std::move(xsize), std::move(xstride), storage_offset));
}

const at::Tensor& LazyNativeFunctions::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  auto xsize = torch::lazy::ToI64Vector(size);
  auto xstride = torch::lazy::ToI64Vector(stride);
  if (!torch::lazy::AsStrided::StrideIsSupported(xstride)) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP(as_strided_)>::call(self, size, stride,
                                                         storage_offset);
  }
  lazy_tensor_aten_ops::as_strided_(self_tensor, std::move(xsize),
                                    std::move(xstride), storage_offset);
  return self;
}

at::Tensor LazyNativeFunctions::bernoulli(
    const at::Tensor& self, c10::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&ltc_eager_fallback,
                                        ATEN_OP(bernoulli)>::call(self,
                                                                  generator);
  }
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::bernoulli(self_tensor));
}

at::Tensor& LazyNativeFunctions::bernoulli_(
    at::Tensor& self, double p, c10::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP2(bernoulli_, float)>::call(self, p,
                                                                generator);
  }
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor LazyNativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto lazy_tensors = torch::lazy::GetLtcTensors(tensors);
  std::vector<torch::lazy::Value> values;
  values.reserve(lazy_tensors.size());
  for (auto& tensor : lazy_tensors) {
    values.emplace_back(tensor.GetIrValue());
  }

  auto shapes =
      torch_lazy_tensors::ir::ops::compute_shape_cat(tensors, dim);
  auto node =
      torch::lazy::MakeNode<ir::ops::Cat>(values, dim, std::move(shapes));
  auto result = torch::lazy::CreateAtenFromLtcTensor(
      torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), lazy_tensors[0].GetDevice()));
  return result;
}

at::Tensor LazyNativeFunctions::clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  auto self_lt = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(self_lt.Create(self_lt.GetIrValue(), self_lt.GetDevice()));
}

at::Tensor LazyNativeFunctions::_copy_from(const at::Tensor& self,
                                           const at::Tensor& dst,
                                           bool non_blocking) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto dst_tensor = torch::lazy::TryGetLtcTensor(dst);
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_tensor) {
    // providing a new 'eager' value (self) for an existing lazy tensor (dst)
    static bool sync_update =
        lazy_tensors::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    CHECK(dst_tensor);
    dst_tensor.UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    // materializing a lazy tensor (self) and copying its value into eager tensor (dst)
    // detached=false lets us skip a copy in `ToTensor`, which should be safe
    // becuase we are only going to use the tensor for dst.copy_()
    at::Tensor tensor = self_tensor.ToTensor(/*detached=*/false);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // Copying one lazy tensor to another
    if (!dst_tensor.CurrentIrValue()) {
      // if dest is not backed by IR (e.g. result of some lazy operation),
      // then it should have at::Tensor data backing it instead
      auto dst_tensor_data = dst_tensor.CurrentTensorData();
      CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor.CurrentTensorData();
      if (src_tensor_data) {
        // both src/dst are simply backed by at::Tensor data, no IR- do a straightforward copy
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        // src needs to be materialized before its result can be used for a copy into dst
        // since we use the src tensor only for making a copy, we don't need to detach it
        // note: it would be even more efficient if we could cause ToTensor to materialize the
        // value directly into dst's buffer (that would need to be detached though).
        dst_tensor_data->copy_(self_tensor.ToTensor(/*detached=*/false));
      }
    } else {
      lazy_tensor_aten_ops::copy_(dst_tensor, self_tensor);
      auto* impl = dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
      impl->set_tensor(dst_tensor);
    }
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_copy_from_and_resize(const at::Tensor& self,
                                                      const at::Tensor& dst) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto dst_tensor = torch::lazy::TryGetLtcTensor(dst);
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_tensor) {
    CHECK(dst_tensor);
    dst_tensor.UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor.ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is a lazy tensor
    auto* dest_impl =
        dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
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
  return CreateLtcTensor(x_result, GetLtcDevice(device));
}

at::Tensor LazyNativeFunctions::empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return torch_lazy_tensors::LazyNativeFunctions::as_strided(
      t, size, stride, /*storage_offset=*/0);
}

at::Tensor LazyNativeFunctions::expand(const at::Tensor& self,
                                       at::IntArrayRef size, bool implicit) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::expand(
      torch::lazy::TryGetLtcTensor(self), size.vec()));
}

at::Tensor& LazyNativeFunctions::fill_(at::Tensor& self,
                                       const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::fill_(self_tensor, value);
  return self;
}

at::Tensor LazyNativeFunctions::max_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool3dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor input_tensor = torch::lazy::TryGetLtcTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor.GetDevice();
  torch::lazy::LazyTensor running_mean_tensor =
      GetOrCreateLtcTensor(running_mean, device);
  torch::lazy::LazyTensor running_var_tensor =
      GetOrCreateLtcTensor(running_var, device);
  auto outputs = lazy_tensor_aten_ops::ts_native_batch_norm(
      torch::lazy::TryGetLtcTensor(input), GetOrCreateLtcTensor(weight, device),
      GetOrCreateLtcTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(torch::lazy::CreateAtenFromLtcTensor(std::get<0>(outputs)),
                         torch::lazy::CreateAtenFromLtcTensor(std::get<1>(outputs)),
                         torch::lazy::CreateAtenFromLtcTensor(std::get<2>(outputs)));
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
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor grad_out_tensor = torch::lazy::TryGetLtcTensor(grad_out);
  const torch::lazy::BackendDevice& device = grad_out_tensor.GetDevice();
  torch::lazy::LazyTensor null_tensor;
  bool running_stats = running_mean && running_mean->defined();
  CHECK_EQ(running_var && running_var->defined(), running_stats);
  auto gradients = lazy_tensor_aten_ops::ts_native_batch_norm_backward(
      torch::lazy::TryGetLtcTensor(grad_out), torch::lazy::TryGetLtcTensor(input),
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
      output_mask[0] ? torch::lazy::CreateAtenFromLtcTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? torch::lazy::CreateAtenFromLtcTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? torch::lazy::CreateAtenFromLtcTensor(std::get<2>(gradients))
                     : undefined);
}

// We need to explicitly override max pooling operators and just call the
// fallback for them because we've customized the autograd function for them
// (backward needs saved indices from forward).
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

at::Tensor & LazyNativeFunctions::normal_(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
    // Unconditionally fall back.
    // implementing normal_ via lazy tensor caused differences in results compared to eager.
    return at::native::call_fallback_fn<&ltc_eager_fallback, ATEN_OP(normal_)>::call(self, mean, std, generator);

    // if (force_eager_fallback(c10::Symbol::fromQualString("aten::normal_"))) {
    //   return at::native::call_fallback_fn<&ltc_eager_fallback, ATEN_OP(normal_)>::call(self, mean, std, generator);
    // }

    // if (generator.has_value()) {
    //   return at::native::call_fallback_fn<&ltc_eager_fallback, ATEN_OP(normal_)>::call(self, mean, std, generator);
    // }

    // TORCH_LAZY_FN_COUNTER("lazy::");
    // auto device = bridge::GetBackendDevice(self);
    // LazyTensor lazy_self = GetLtcTensorOrCreateForWrappedNumber(self, *device);
    // std::vector<torch::lazy::Shape> shapes = {torch::lazy::Shape(self.scalar_type(), self.sizes().vec())};
    // auto node = torch::lazy::MakeNode<ir::ops::Normal>(lazy_self.GetIrValue(), mean, std, std::move(shapes));
    // lazy_self.SetInPlaceIrValue(node);
    // return self;
};

at::Tensor LazyNativeFunctions::permute(const at::Tensor& self,
                                        at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::permute(
      self_tensor, torch::lazy::ToI64Vector(dims)));
}

at::Tensor& LazyNativeFunctions::random_(
    at::Tensor& self, int64_t from, c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER("lazy::");

  if (generator && generator->defined()) {
    return at::native::call_fallback_fn<
        &ltc_eager_fallback, ATEN_OP2(random_, from)>::call(self, from, to,
                                                            generator);
  }

  auto selfTensor = torch::lazy::TryGetLtcTensor(self);
  selfTensor.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor.GetIrValue(), from, to));
  return self;
}

at::Tensor& LazyNativeFunctions::random_(
    at::Tensor& self, int64_t to, c10::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER("lazy::");

  if (generator && generator->defined()) {
    return at::native::call_fallback_fn<&ltc_eager_fallback,
                                        ATEN_OP2(random_, to)>::call(self, to,
                                                                     generator);
  }

  auto selfTensor = torch::lazy::TryGetLtcTensor(self);
  selfTensor.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor.GetIrValue(), c10::nullopt, to));
  return self;
}

at::Tensor& LazyNativeFunctions::random_(
    at::Tensor& self, c10::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER("lazy::");

  if (generator && generator->defined()) {
    return at::native::call_fallback_fn<&ltc_eager_fallback,
                                        ATEN_OP(random_)>::call(self,
                                                                generator);
  }

  auto selfTensor = torch::lazy::TryGetLtcTensor(self);
  selfTensor.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor.GetIrValue(), c10::nullopt, c10::nullopt));
  return self;
}

at::Tensor LazyNativeFunctions::repeat(const at::Tensor& self,
                                       at::IntArrayRef repeats) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::repeat(
      torch::lazy::TryGetLtcTensor(self), torch::lazy::ToI64Vector(repeats)));
}

at::Tensor LazyNativeFunctions::select(const at::Tensor& self, int64_t dim,
                                       int64_t index) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::select(torch::lazy::TryGetLtcTensor(self), dim, index));
}

at::Tensor LazyNativeFunctions::slice(const at::Tensor& self, int64_t dim,
                                      c10::optional<int64_t> start,
                                      c10::optional<int64_t> end,
                                      int64_t step) {
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::slice(
      torch::lazy::TryGetLtcTensor(self), dim, start_val, end_val, step));
}

at::Tensor LazyNativeFunctions::stack(at::TensorList tensors, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::stack(torch::lazy::GetLtcTensors(tensors), dim));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::squeeze(torch::lazy::TryGetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::squeeze(torch::lazy::TryGetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::squeeze_(self_tensor);
  return self;
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::squeeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::sub(const at::Tensor& self,
                                    const at::Tensor& other,
                                    const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const torch::lazy::LazyTensor& xself, const torch::lazy::LazyTensor& xother) {
                      return lazy_tensor_aten_ops::sub(xself, xother, alpha);
                    });
}

at::Tensor LazyNativeFunctions::sub(const at::Tensor& self,
                                    const at::Scalar& other,
                                    const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  CheckSubOperandTypes(self.scalar_type(), other.type());
  return DoBinaryOp(self, other,
                    [&](const torch::lazy::LazyTensor& xself, const at::Scalar& other) {
                      return lazy_tensor_aten_ops::sub(xself, other, alpha);
                    });
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::transpose(torch::lazy::TryGetLtcTensor(self), 0, 1));
}

at::Tensor& LazyNativeFunctions::t_(at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor LazyNativeFunctions::transpose(const at::Tensor& self, int64_t dim0,
                                          int64_t dim1) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::transpose(torch::lazy::TryGetLtcTensor(self), dim0, dim1));
}

at::Tensor& LazyNativeFunctions::transpose_(at::Tensor& self, int64_t dim0,
                                            int64_t dim1) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::transpose_(self_tensor, dim0, dim1);
  return self;
}

at::Tensor LazyNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::unsqueeze(torch::lazy::TryGetLtcTensor(self), dim));
}

at::Tensor& LazyNativeFunctions::unsqueeze_(at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::view(self_tensor, torch::lazy::ToI64Vector(size)));
}

void InitializeAtenBindings() {}

}  // namespace torch_lazy_tensors
