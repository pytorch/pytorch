#pragma once

#include "ATen/Tensor.h"
#include "ATen/Scalar.h"


namespace at {


inline Tensor & Tensor::operator=(Scalar v) && {
  return assign_(v);
}
inline Tensor & Tensor::assign_(Scalar v) {
  AT_ASSERT(dim() == 0, "attempting to assign a scalar to %d dim tensor", dim());
  pImpl->assign_(v);
  return *this;
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
inline Tensor Tensor::operator[](int64_t idx) const {
  return select(0, idx);
}

#define AT_FORALL_BINARY_OPS(_) \
_(+,x.add(y), y.add(x)) \
_(*,x.mul(y), y.mul(x)) \
_(-,x.sub(y), y.type().tensor().resize_(y.sizes()).fill_(x).sub_(y)) \
_(/,x.div(y), y.type().tensor().resize_(y.sizes()).fill_(x).div_(y)) \
_(%,x.remainder(y), y.type().tensor().resize_(y.sizes()).fill_(x).remainder_(y)) \
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
