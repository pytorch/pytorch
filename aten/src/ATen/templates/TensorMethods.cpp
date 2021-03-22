#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/Stream.h>
#include <c10/macros/Macros.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

${static_dispatch_extra_headers}

namespace at {

using Stream = c10::Stream;

Tensor Tensor::cpu() const {
  return to(options().device(DeviceType::CPU), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: The Python version also accepts arguments
Tensor Tensor::cuda() const {
  return to(options().device(DeviceType::CUDA), /*non_blocking*/ false, /*copy*/ false);
}

Tensor Tensor::hip() const {
  return to(options().device(DeviceType::HIP), /*non_blocking*/ false, /*copy*/ false);
}

Tensor Tensor::vulkan() const {
  return to(options().device(DeviceType::Vulkan), /*non_blocking*/ false, /*copy*/ false);
}

Tensor Tensor::metal() const {
  return to(options().device(DeviceType::Metal), /*non_blocking*/ false, /*copy*/ false);
}

Tensor Tensor::toType(ScalarType t) const {
  return to(options().dtype(t), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: Deprecate me
Tensor Tensor::toBackend(Backend b) const {
  return to(options().device(backendToDeviceType(b)).layout(layout_from_backend(b)), /*non_blocking*/ false, /*copy*/ false);
}

TensorOptions Tensor::options() const {
  return TensorOptions().dtype(dtype())
                        .device(device())
                        .layout(layout());
}

${tensor_method_definitions}

caffe2::TypeMeta Tensor::dtype() const noexcept {
  return impl_->dtype();
}

Layout Tensor::layout() const noexcept {
  return impl_->layout();
}

int64_t Tensor::get_device() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->get_device();
}

int64_t get_device(Tensor self) {
  return self.get_device();
}

bool Tensor::is_cuda() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_cuda();
}

bool Tensor::is_xpu() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_xpu();
}

bool is_xpu(Tensor self) {
  // NB: this is not a native function to avoid dispatching overhead.
  return self.is_xpu();
}

bool Tensor::is_xla() const {
    return impl_->is_xla();
}

NamedTensorMeta* Tensor::get_named_tensor_meta() {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

const NamedTensorMeta* Tensor::get_named_tensor_meta() const {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

bool Tensor::has_names() const {
  // If a user is using unnamed tensors, then we can short-circuit right here.
  // Otherwise, impl::has_names attempts to retrieve names.
  if (!impl_->has_named_tensor_meta()) {
    return false;
  }
  return impl::has_names(unsafeGetTensorImpl());
}

bool is_cuda(Tensor self) {
  return self.is_cuda();
}

bool is_xla(Tensor self) {
    return self.is_xla();
}

bool Tensor::is_hip() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_hip();
}

bool is_hip(Tensor self) {
  return self.is_hip();
}

bool Tensor::is_sparse() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_sparse();
}

bool is_sparse(Tensor self) {
  return self.is_sparse();
}

bool Tensor::is_mkldnn() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_mkldnn();
}

bool is_mkldnn(Tensor self) {
  return self.is_mkldnn();
}

bool Tensor::is_mlc() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_mlc();
}

bool is_mlc(Tensor self) {
  return self.is_mlc();
}

bool Tensor::is_vulkan() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_vulkan();
}

bool Tensor::is_metal() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_metal();
}


bool is_vulkan(Tensor self) {
  return self.is_vulkan();
}

bool is_metal(Tensor self) {
  return self.is_metal();
}

bool Tensor::is_quantized() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_quantized();
}

bool Tensor::is_meta() const {
  return impl_->is_meta();
}

bool is_quantized(Tensor self) {
  return self.is_quantized();
}

#define DEFINE_CAST(T, name)                                        \
  template <>                                                       \
  TORCH_API T* Tensor::data_ptr() const {                           \
    TORCH_CHECK(                                                    \
        scalar_type() == ScalarType::name,                          \
        "expected scalar type "                                     \
        #name                                                       \
        " but found ",                                              \
        scalar_type());                                             \
    return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
AT_FORALL_QINT_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_ITEM(T, name)      \
  template <>                     \
  TORCH_API T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ITEM)
#undef DEFINE_ITEM

} //namespace at
