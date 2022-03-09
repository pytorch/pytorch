#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/library.h>

#include "ATen/MetaFunctions.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/random.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/ops/to_copy.h"
#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include <torch/csrc/lazy/core/tensor_impl.h>
#include "lazy_tensor_core/csrc/ts_backend/LazyNativeFunctions.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_autograd_ops_ts.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h"
#include "lazy_tensors/computation_client/sys_util.h"
namespace torch_lazy_tensors {
namespace {

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
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor LazyNativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto lazy_tensors = torch::lazy::GetLtcTensors(tensors);
  std::vector<torch::lazy::Value> values;
  values.reserve(lazy_tensors.size());
  for (auto& tensor : lazy_tensors) {
    values.emplace_back(tensor->GetIrValue());
  }

  auto shapes = torch::lazy::compute_shape_cat(tensors, dim);
  auto node =
      torch::lazy::MakeNode<ir::ops::Cat>(values, dim, std::move(shapes));
  auto result = torch::lazy::CreateAtenFromLtcTensor(
      torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), lazy_tensors[0]->GetDevice()));
  return result;
}

at::Tensor LazyNativeFunctions::clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  auto self_lt = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(self_lt->Create(self_lt->GetIrValue(), self_lt->GetDevice()));
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
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    // materializing a lazy tensor (self) and copying its value into eager tensor (dst)
    // detached=false lets us skip a copy in `ToTensor`, which should be safe
    // becuase we are only going to use the tensor for dst.copy_()
    CHECK(self_tensor);
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/false);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // Copying one lazy tensor to another
    if (!dst_tensor->CurrentIrValue()) {
      // if dest is not backed by IR (e.g. result of some lazy operation),
      // then it should have at::Tensor data backing it instead
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        // both src/dst are simply backed by at::Tensor data, no IR- do a straightforward copy
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        // src needs to be materialized before its result can be used for a copy into dst
        // since we use the src tensor only for making a copy, we don't need to detach it
        // note: it would be even more efficient if we could cause ToTensor to materialize the
        // value directly into dst's buffer (that would need to be detached though).
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/false));
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
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    CHECK(self_tensor);
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is a lazy tensor
    auto* dest_impl =
        dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor()->UpdateFromTensorOut(self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_to_copy(const at::Tensor & self,
                                         c10::optional<at::ScalarType> dtype,
                                         c10::optional<at::Layout> layout,
                                         c10::optional<at::Device> device,
                                         c10::optional<bool> pin_memory,
                                         bool non_blocking,
                                         c10::optional<at::MemoryFormat> memory_format) {

    if (force_eager_fallback(at::aten::_to_copy)) {
      TORCH_INTERNAL_ASSERT(false,
        "Fallback is currently impossible for _to_copy since the fallback helper itself reinvokes _to_copy");
    }

    auto options = self.options();
    if (dtype) {
      // I put each of these setters in a conditional instead of doing `self.options().dtype(dtype).layout(layout)...
      // because calling .dtype(nullopt) on an options() that already has dtype appears to wipe it
      options = options.dtype(dtype);
    }
    if (layout) {
      options = options.layout(layout);
    }
    if (memory_format) {
      options = options.memory_format(memory_format);
    }
    if (pin_memory) {
      // TODO(whc) can we honor 'pin_memory' in some/all cases?
      options = options.pinned_memory(pin_memory);
      TORCH_WARN_ONCE("Pinned memory used in lazy _to_copy, check if the behavior is as intended");
    }

    TORCH_LAZY_FN_COUNTER("lazy::");
    auto lazy_self = torch::lazy::TryGetLtcTensor(self);
    if (!lazy_self && device && device->type() == c10::kLazy) {
      // Case 1: eager->lazy (we create a new lazy tensor)

      auto eager_tensor = self.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
      lazy_self = torch::lazy::GetOrCreateLtcTensor(eager_tensor,
                                                    torch::lazy::atenDeviceToBackendDevice(*device));
      return torch::lazy::CreateAtenFromLtcTensor(lazy_self);
    } else if(device && device->type() != c10::kLazy) {
      // Case 2: lazy->eager (forces a graph break since we are materializing a tensor)

      TORCH_INTERNAL_ASSERT(lazy_self);
      auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
      options = options.device(device);
      auto moved_eager_tensor = eager_tensor.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
      return moved_eager_tensor;
    } else if (device &&
               device->type() == c10::kLazy &&
               device->has_index() &&
               device->index() != self.device().index()) {
      // Case 3: lazy:0 -> lazy:1

      // TODO(whc) what do we actually want to do here?
      //   option 1: materialize, move eager tensor, create new lazy tensor
      //     - this should be our default, as it is what would happen before we implemented _to_copy
      //     - actually combines case 1 + case 2
      //   option 2: support multiple devices inside one lazy/TS executor (case 4)
      //     - but: we may have other assumptions that there is just one device per executor? so don't take this lightly

      TORCH_INTERNAL_ASSERT(lazy_self);
      auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
      // we move the eager tensor to the 'eager' equivalent of our lazy device
      // e.g. if our device is lazy:1, the backend maps that to cuda:1, which is what we use
      auto eager_device = c10::Device(torch::lazy::getBackend()->EagerFallbackDeviceType(), device->index());
      options = options.device(eager_device);
      auto moved_eager_tensor = eager_tensor.to(options, /*non_blocking=*/false, /*copy=*/true);
      lazy_self = torch::lazy::GetOrCreateLtcTensor(moved_eager_tensor,
                                                    torch::lazy::atenDeviceToBackendDevice(eager_device));
      return torch::lazy::CreateAtenFromLtcTensor(lazy_self);

    } else {
      // Case 4: lazy->lazy (special case: keep the _to_copy INSIDE the lazy graph)

      // Note: captured _to_copy will be executed with real eager tensors, not lazy tensors.
      // We DO NOT want to burn 'lazy:0' as the device into this captured IR, or we will try to
      // convert an eager tensor back to a lazy one inside the torchscript executor
      // lazy:0 -> lazy:1 is handled in case3, so we can safely drop the device argument
      device = c10::nullopt;

      auto shapes = torch::lazy::compute_shape__to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
      TORCH_INTERNAL_ASSERT(shapes.size() == 1);
      auto node = torch::lazy::MakeNode<ir::ops::ToCopy>(lazy_self->GetIrValue(),
                            dtype,
                            layout,
                            device,
                            pin_memory,
                            non_blocking,
                            memory_format,
                            std::move(shapes));

      auto result = torch::lazy::CreateAtenFromLtcTensor(
              torch::lazy::LazyTensor::Create(std::move(node), lazy_self->GetDevice()));
      return result;
    }
};


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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  auto input_tensor = torch::lazy::TryGetLtcTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  auto running_mean_tensor =
      GetOrCreateLtcTensor(running_mean, device);
  auto running_var_tensor =
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
  auto grad_out_tensor = torch::lazy::TryGetLtcTensor(grad_out);
  const torch::lazy::BackendDevice& device = grad_out_tensor->GetDevice();
  torch::lazy::LazyTensorPtr null_tensor;
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  selfTensor->SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor->GetIrValue(), from, to));
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
  selfTensor->SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor->GetIrValue(), c10::nullopt, to));
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
  selfTensor->SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Random>(
      selfTensor->GetIrValue(), c10::nullopt, c10::nullopt));
  return self;
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
  auto common_device = torch::lazy::GetBackendDevice(tensors);
  TORCH_INTERNAL_ASSERT(common_device);

  // TODO(whc) - highly suboptimal old code to support calling the canonicalization method,
  // could clean this up but prefer to actually delete canonicalizatoin,
  // but need to rewrite shape inference at the same time since it asserts dim <=0
  auto lazy_tensors = torch::lazy::GetLtcTensors(tensors);
  CHECK_GT(tensors.size(), 0);
  std::vector<torch::lazy::Value> values;
  for (auto tensor : lazy_tensors) {
    values.push_back(tensor->GetIrValue());
  }
  int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, lazy_tensors.front()->shape().Get().dim() + 1);

  return torch::lazy::CreateAtenFromLtcTensor(
    torch::lazy::LazyTensor::Create(
      torch::lazy::MakeNode<ir::ops::Stack>(torch::lazy::GetTensorList(tensors), canonical_dim),
      *common_device));
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::squeeze_(self_tensor);
  return self;
}

at::Tensor& LazyNativeFunctions::squeeze_(at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::squeeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::transpose(torch::lazy::TryGetLtcTensor(self), 0, 1));
}

at::Tensor& LazyNativeFunctions::t_(at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
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
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor LazyNativeFunctions::view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::view(self_tensor, torch::lazy::ToI64Vector(size)));
}

at::Tensor LazyNativeFunctions::_unsafe_view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::view(self_tensor, torch::lazy::ToI64Vector(size)));
}

void InitializeAtenBindings() {}

}  // namespace torch_lazy_tensors
