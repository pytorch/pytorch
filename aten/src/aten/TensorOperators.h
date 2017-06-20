#pragma once

#include "TensorLib/Tensor.h"
#include "TensorLib/Scalar.h"


namespace tlib {

#define TLIB_FORALL_BINARY_OPS(_) \
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
Tensor operator op(const Tensor & x, const Tensor & y) { \
  return body; \
} \
Tensor operator op(const Tensor & x, Scalar y) { \
  return body; \
} \
Tensor operator op(const Scalar & x, Tensor y) { \
  return reverse_scalar_body; \
}

TLIB_FORALL_BINARY_OPS(DEFINE_OPERATOR)
#undef DEFINE_OPERATOR
#undef TLIB_FORALL_BINARY_OPS

}
