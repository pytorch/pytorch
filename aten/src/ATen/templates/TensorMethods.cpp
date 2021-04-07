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
#include <ATen/core/op_registration/adaption.h>
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

NamedTensorMeta* Tensor::get_named_tensor_meta() {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

const NamedTensorMeta* Tensor::get_named_tensor_meta() const {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
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
