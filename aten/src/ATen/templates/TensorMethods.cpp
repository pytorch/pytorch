#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#ifdef USE_STATIC_DISPATCH
#include <ATen/TypeDefault.h>
#include <ATen/CPUType.h>
#include <ATen/QuantizedCPUType.h>
#endif

namespace at {

// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;

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

// all static to allow for inlining of the non-dynamic part of dispatch
${tensor_method_definitions}

caffe2::TypeMeta Tensor::dtype() const noexcept {
  return impl_->dtype();
}

Layout Tensor::layout() const noexcept {
  return impl_->layout();
}

Device Tensor::device() const {
  return impl_->device();
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

bool Tensor::is_quantized() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_quantized();
}

bool is_quantized(Tensor self) {
  return self.is_quantized();
}

#define DEFINE_CAST(T, name)                     \
  template <>                                    \
  T* Tensor::data_ptr() const {           \
    TORCH_CHECK(                                 \
        scalar_type() == ScalarType::name,       \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(scalar_type()));           \
    return static_cast<T*>(this->unsafeGetTensorImpl()->data());    \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
AT_FORALL_QINT_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

// TODO(@zasdfgbnm): Remove this!
// This is needed only when the migration of std::complex to c10::complex
// is not done. This should be removed once the migration is done.
template <>
std::complex<float>* Tensor::data_ptr() const {
  TORCH_CHECK(scalar_type() == ScalarType::ComplexFloat,
    "expected scalar type ComplexFloat but found ",
    c10::toString(scalar_type()));
  return static_cast<std::complex<float>*>(this->unsafeGetTensorImpl()->data());
}
template <>
std::complex<double>* Tensor::data_ptr() const {
  TORCH_CHECK(scalar_type() == ScalarType::ComplexDouble,
    "expected scalar type ComplexDouble but found ",
    c10::toString(scalar_type()));
  return static_cast<std::complex<double>*>(this->unsafeGetTensorImpl()->data());
}
// end TODO

#define DEFINE_ITEM(T, name)      \
  template <>                     \
  T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ITEM)
#undef DEFINE_ITEM

// TODO(@zasdfgbnm): Remove this!
// This is needed only when the migration of std::complex to c10::complex
// is not done. This should be removed once the migration is done.
template <>
std::complex<float> Tensor::item() const {
  // casting from c10::complex<float> to std::complex<float>
  return static_cast<std::complex<float>>(item().toComplexFloat());
}
template <>
std::complex<double> Tensor::item() const {
  // casting from c10::complex<double> to std::complex<double>
  return static_cast<std::complex<double>>(item().toComplexFloat());
}
// end TODO

} //namespace at
