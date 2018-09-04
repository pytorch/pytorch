#pragma once

// ${generated_comment}

#include "ATen/Tensor.h"
#include "ATen/Scalar.h"
#include "ATen/core/SparseTensorRef.h"
#include "ATen/Type.h"
#include "ATen/core/TensorOptions.h"

namespace at {

inline Tensor Tensor::toType(const Type & t, bool non_blocking) const {
  if(type() == t)
    return *this;
  return t.copy(*this, non_blocking);
}

inline Tensor Tensor::cpu() const {
  return toType(type().cpu());
}

inline Tensor Tensor::cuda() const {
  return toType(type().cuda());
}

inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) {
  return type().copy_(*this, src, non_blocking);
}

inline Tensor Tensor::toType(ScalarType t) const {
  return toType(type().toScalarType(t));
}

inline Tensor Tensor::toBackend(Backend b) const {
  return toType(type().toBackend(b));
}

inline TensorOptions Tensor::options() const {
  return TensorOptions().dtype(dtype())
                        .device(device())
                        .layout(layout())
                        .is_variable(is_variable());
}

namespace detail {
inline Tensor to(
    const Tensor& tensor,
    const TensorOptions& options,
    bool non_blocking) {
  // Don't copy if the options match.
  if (tensor.options() == options) {
    return tensor;
  }
  AT_CHECK(tensor.is_variable() == options.is_variable(),
           "cannot change is_variable, from: ", tensor.is_variable(),
           " to: ", options.is_variable());
  return tensor.type().toBackend(options.backend()).toScalarType(options.dtype())
               .copy(tensor, non_blocking, options.device());
}
} // namespace detail

inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking)
    const {
  if (this->device() == device && this->dtype() == dtype) {
    return *this;
  }
  return detail::to(*this, options().device(device).dtype(dtype), non_blocking);
}

inline Tensor Tensor::to(ScalarType dtype, bool non_blocking) const {
  if (this->dtype() == dtype) {
    return *this;
  }
  return detail::to(*this, options().dtype(dtype), non_blocking);
}

inline Tensor Tensor::to(Device device, bool non_blocking) const {
  if (this->device() == device) {
    return *this;
  }
  return detail::to(*this, options().device(device), non_blocking);
}

// all static inline to allow for inlining of the non-dynamic part of dispatch
${tensor_method_definitions}

#define DEFINE_CAST(T, name, _)                  \
  template <>                                    \
  inline T* Tensor::data() const {               \
    AT_CHECK(                                    \
        type().scalarType() == ScalarType::name, \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        at::toString(type().scalarType()));      \
    return static_cast<T*>(this->data_ptr());    \
  }                                              \
  inline T* Tensor::to##name##Data() const {     \
    return data<T>();                            \
  }

AT_FORALL_SCALAR_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_TO_C_TYPE(T,name,_) \
inline T Tensor::toC##name () const { return _local_scalar().to##name (); }

AT_FORALL_SCALAR_TYPES(DEFINE_TO_C_TYPE)
#undef DEFINE_TO_C_TYPE

} //namespace at
