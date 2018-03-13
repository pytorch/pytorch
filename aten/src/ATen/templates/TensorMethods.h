#pragma once

#include "ATen/Tensor.h"
#include "ATen/Scalar.h"
#include "ATen/SparseTensorRef.h"

namespace at {

inline Tensor & Tensor::operator=(Tensor const & rhs) && {
  return copy_(rhs);
}

inline Tensor Tensor::toType(const Type & t, bool non_blocking) const {
  if(type() == t)
    return *this;
  return t.copy(*this, non_blocking);
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


// all static inline to allow for inlining of the non-dynamic part of dispatch
${tensor_method_definitions}

#define DEFINE_CAST(T,name,_) \
template<> \
inline T* Tensor::data() const { \
  AT_ASSERT(type().scalarType() == ScalarType::name, \
    "expected scalar type % s but found %s", #name, \
    at::toString(type().scalarType())); \
  return static_cast<T*>(this->data_ptr()); \
} \
inline T* Tensor::to##name##Data() const { return data<T>(); }

AT_FORALL_SCALAR_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_TO_C_TYPE(T,name,_) \
inline T Tensor::toC##name () const { return pImpl->localScalar().to##name (); }

AT_FORALL_SCALAR_TYPES(DEFINE_TO_C_TYPE)
#undef DEFINE_TO_C_TYPE

} //namespace at
