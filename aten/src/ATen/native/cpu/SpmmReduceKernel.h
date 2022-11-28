#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

enum SPMM_REDUCE_OP {SPMM_SUM, SPMM_MAX, SPMM_MIN, SPMM_MEAN};

using spmm_reduce_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, SPMM_REDUCE_OP op);
using spmm_reduce_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, SPMM_REDUCE_OP op);
using spmm_reduce_backward_input_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, SPMM_REDUCE_OP op);
using spmm_reduce_backward_input_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, SPMM_REDUCE_OP op);
using spmm_reduce_backward_weight_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, SPMM_REDUCE_OP op);

DECLARE_DISPATCH(spmm_reduce_fn, spmm_reduce_stub);
DECLARE_DISPATCH(spmm_reduce_arg_fn, spmm_reduce_arg_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_fn, spmm_reduce_backward_input_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_arg_fn, spmm_reduce_backward_input_arg_stub);
DECLARE_DISPATCH(spmm_reduce_backward_weight_fn, spmm_reduce_backward_weight_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_arg_fn, spmm_reduce_backward_weight_arg_stub);

#define AT_DISPATCH_REDUCTION_TYPES(op, ...)                                   \
  [&] {                                                                        \
    switch (op) {                                                              \
      case SPMM_SUM: {                                                         \
        static constexpr SPMM_REDUCE_OP reduce = SPMM_SUM;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case SPMM_MEAN: {                                                        \
        static constexpr SPMM_REDUCE_OP reduce = SPMM_MEAN;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case SPMM_MIN: {                                                         \
        static constexpr SPMM_REDUCE_OP reduce = SPMM_MIN;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case SPMM_MAX: {                                                         \
        static constexpr SPMM_REDUCE_OP reduce = SPMM_MAX;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    }                                                                          \
  }()

}} // at::native
