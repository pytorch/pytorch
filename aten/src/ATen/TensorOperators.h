#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

namespace at {

#define AT_FORALL_BINARY_OPS(_)                                             \
  _(+, x.add(y), y.add(x))                                                  \
  _(*, x.mul(y), y.mul(x))                                                  \
  _(-,                                                                      \
    x.sub(y),                                                               \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).sub_(y))       \
  _(/,                                                                      \
    x.div(y),                                                               \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).div_(y))       \
  _(%,                                                                      \
    x.remainder(y),                                                         \
    ::at::empty_like(y, at::MemoryFormat::Preserve).fill_(x).remainder_(y)) \
  _(&, x.bitwise_and(y), y.bitwise_and(x))                                  \
  _(|, x.bitwise_or(y), y.bitwise_or(x))                                    \
  _(^, x.bitwise_xor(y), y.bitwise_xor(x))                                  \
  _(<, x.lt(y), y.gt(x))                                                    \
  _(<=, x.le(y), y.ge(x))                                                   \
  _(>, x.gt(y), y.lt(x))                                                    \
  _(>=, x.ge(y), y.le(x))                                                   \
  _(==, x.eq(y), y.eq(x))                                                   \
  _(!=, x.ne(y), y.ne(x))

#define DEFINE_OPERATOR(op, body, reverse_scalar_body)                 \
  static inline Tensor operator op(const Tensor& x, const Tensor& y) { \
    return body;                                                       \
  }                                                                    \
  static inline Tensor operator op(const Tensor& x, const Scalar& y) { \
    return body;                                                       \
  }                                                                    \
  static inline Tensor operator op(const Scalar& x, const Tensor& y) { \
    return reverse_scalar_body;                                        \
  }

AT_FORALL_BINARY_OPS(DEFINE_OPERATOR)
#undef DEFINE_OPERATOR
#undef AT_FORALL_BINARY_OPS

} // namespace at
