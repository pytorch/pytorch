#pragma once

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>

#include <string>
#include <stdexcept>

namespace at {

inline Tensor & Tensor::operator=(Tensor const & rhs) && {
  return copy_(rhs);
}
inline Tensor & Tensor::operator=(Tensor && rhs) && {
  return copy_(rhs);
}
inline Tensor & Tensor::operator=(Scalar v) && {
  return fill_(v);
}
inline Tensor Tensor::operator-() const {
  return neg();
}
inline Tensor& Tensor::operator+=(const Tensor & other) {
  return add_(other);
}
inline Tensor& Tensor::operator+=(Scalar other) {
  return add_(other);
}
inline Tensor& Tensor::operator-=(const Tensor & other) {
  return sub_(other);
}
inline Tensor& Tensor::operator-=(Scalar other) {
  return sub_(other);
}
inline Tensor& Tensor::operator*=(const Tensor & other) {
  return mul_(other);
}
inline Tensor& Tensor::operator*=(Scalar other) {
  return mul_(other);
}
inline Tensor& Tensor::operator/=(const Tensor & other) {
  return div_(other);
}
inline Tensor& Tensor::operator/=(Scalar other) {
  return div_(other);
}
inline Tensor Tensor::operator[](Scalar index) const {
  if (!index.isIntegral(false)) {
    AT_INDEX_ERROR("Can only index tensors with integral scalars");
  }
  return select(0, index.toLong());
}
inline Tensor Tensor::operator[](Tensor index) const {
  // These properties are checked in the Scalar constructor, but we already
  // check them here to provide more useful diagnostics for the user.
  if (!index.defined()) {
    AT_INDEX_ERROR("Can only index with tensors that are defined");
  }
  if (index.dim() != 0) {
    AT_INDEX_ERROR(
      "Can only index with tensors that are scalars (zero-dim)");
  }
  // The Scalar(Tensor) constructor is explicit, so we need to call it.
  return this->operator[](index.item());
}
inline Tensor Tensor::operator[](int64_t index) const {
  return select(0, index);
}

#define AT_FORALL_BINARY_OPS(_) \
_(+,x.add(y), y.add(x)) \
_(*,x.mul(y), y.mul(x)) \
_(-,x.sub(y), ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).sub_(y)) \
_(/,x.div(y), ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).div_(y)) \
_(%,x.remainder(y), ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).remainder_(y)) \
_(<,x.lt(y), y.gt(x)) \
_(<=,x.le(y), y.ge(x)) \
_(>,x.gt(y),y.lt(x)) \
_(>=,x.ge(y), y.le(x)) \
_(==,x.eq(y), y.eq(x)) \
_(!=,x.ne(y), y.ne(x))

#define DEFINE_OPERATOR(op,body,reverse_scalar_body) \
static inline Tensor operator op(const Tensor & x, const Tensor & y) { \
  return body; \
} \
static inline Tensor operator op(const Tensor & x, Scalar y) { \
  return body; \
} \
static inline Tensor operator op(Scalar x, const Tensor & y) { \
  return reverse_scalar_body; \
}


AT_FORALL_BINARY_OPS(DEFINE_OPERATOR)
#undef DEFINE_OPERATOR
#undef AT_FORALL_BINARY_OPS

}
