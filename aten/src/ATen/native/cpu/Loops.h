#pragma once

// This file provides three functions to help write elementwise kernels:
//
//   cpu_kernel(TensorIterator iter, <lambda>)
//   cpu_kernel_vec(TensorIterator iter, <lambda>, <vec_lambda>)
//   cpu_apply_dim_kernel(TensorIterator iter, <lambda>)
//
// Both cpu_kernel and cpu_kernel_vec are designed for elementwise operations
// and they may generate vectorized code. The cpu_kernel implementation relies
// on the compiler's auto-vectorization. The cpu_kernel_vec implementation uses
// x86 SIMD intrinsics when available. These functions are only intended to be
// used in the ATen/native/cpu subdirectory, since files in other directories
// sare not compiled with AVX/AVX2 enabled.
//
// For example, to write a multiplication kernel for float:
//
//   cpu_kernel(iter, [](float a, float b) { return a * b; });
//
// Or you may write:
//
//   cpu_kernel_vec(iter,
//     [](float a, float b) { return a * b; },
//     [](Vec256<float> a, Vec256<float> b) { return a * b; });
//
// See BinaryOpsKernel.cpp for the complete implementation
//
// See README.md for more details.
//
// cpu_apply_dim_kernel is designed for dimension apply. For example, if you want
// to implement gather_out(result, dim, index, src), you may write:
//
//     cpu_apply_dim_kernel(iter,
//       [=](float *result_data, int64_t result_stride, int64_t *index_data, int64_t index_stride, float *src_data, int64_t src_stride) {
//         for (int64_t i = 0; i < size; i++) {
//           int64_t index = *(index_data + i * index_stride);
//           *(result_data + i * result_stride) = *(src_data + index * src_stride);
//         }
//       });

#include <stdint.h>
#include <c10/util/C++17.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cpu/IsContiguous.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cpu/vec256/vec256.h>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif

namespace at { namespace native { namespace {

using namespace vec256;

template <typename traits, std::size_t... I>
typename traits::ArgsTuple
dereference_impl(char* C10_RESTRICT data[], const int64_t* strides, int64_t i,
                 c10::guts::index_sequence<I...>) {
  return std::make_tuple(
      *(typename traits::template arg<I>::type*)
        (data[I] + i * strides[I])...);
}

template <typename traits>
typename traits::ArgsTuple
dereference(char* C10_RESTRICT data[], const int64_t* strides, int64_t i) {
  using Indices = c10::guts::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

template <typename traits, std::size_t... I>
typename traits::ArgsTuple
dereference_vec_impl(char* C10_RESTRICT data[],
                     const typename traits::result_type& opt_scalar,
                     size_t S,
                     int64_t i,
                     c10::guts::index_sequence<I...>) {
  using Vec = typename traits::result_type;
  using scalar_t = typename Vec::value_type;
  return std::make_tuple(
      S == I + 1 ?
      opt_scalar :
      Vec::loadu(data[I] + i * sizeof(scalar_t))...);
}

template <typename traits>
typename traits::ArgsTuple
dereference_vec(char* C10_RESTRICT data[], const typename traits::result_type& opt_scalar, size_t S, int64_t i) {
  using Indices = c10::guts::make_index_sequence<traits::arity>;
  return dereference_vec_impl<traits>(data, opt_scalar, S, i, Indices{});
}

// Basic loop operation (one output, N inputs). May be auto-vectorized
// by the compiler. Supports inputs and outputs of different types.
template <typename func_t>
static inline void
basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (int arg = 0; arg < ntensors; arg++) {
    strides[arg] = strides_[arg];
  }

  for (; i < n; i++) {
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    *out_ptr = c10::guts::apply(op, dereference<traits>(
      &data[1],
      &strides[1],
      i));
  }
}

// Explicitly vectorized loop implementation. All inputs and outputs must be
// the same type and contiguous with one exception: a single input may be
// a scalar (stride 0). It's position is indicated by the argument `S`. If `S`
// is 0, then there are no scalar inputs.
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t op, vec_func_t vop) {
  using traits = function_traits<vec_func_t>;
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vec256<scalar_t>;
  constexpr int ntensors = traits::arity + 1;

  char* C10_RESTRICT data[ntensors];
  for (int arg = 0; arg < ntensors; arg++) {
    data[arg] = data_[arg];
  }

  Vec opt_scalar = Vec(S > 0 ? *(scalar_t*)data[S] : scalar_t(0));
  int64_t i = 0;
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
    auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
    auto out1 = c10::guts::apply(vop, std::move(args1));
    auto out2 = c10::guts::apply(vop, std::move(args2));
    out1.store(data[0] + i * sizeof(scalar_t));
    out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
  }
  if (i < n) {
    int64_t strides[ntensors];
    for (int arg = 0; arg < ntensors; arg++) {
      strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
    }
    basic_loop(data, strides, i, n, op);
  }
}


template <typename traits, typename cb_t>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    c10::guts::index_sequence<>,
    const cb_t& cb) {
  cb(0);
}

template <typename traits, typename cb_t, size_t I0, size_t ...I>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    c10::guts::index_sequence<I0, I...>,
    const cb_t& cb) {
  if (is_contiguous_scalar<traits, I0 + 1>(strides)) {
    cb(I0 + 1);
  } else {
    unroll_contiguous_scalar_checks<traits>(strides, c10::guts::index_sequence<I...>{}, cb);
  }
}

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t op) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(iter.ntensors() >= traits::arity + 1);

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      basic_loop(data, strides, 0, n, op);
    } else {
      using Indices = c10::guts::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t _idx) {
        basic_loop(data, strides, 0, n, op);
      });
    }
  });
}

template <typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIterator& iter, func_t op, vec_func_t vop) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(iter.ntensors() >= traits::arity + 1);

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      return vectorized_loop(data, n, 0, op, vop);
    } else {
      using Indices = c10::guts::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          vectorized_loop(data, n, idx, op, vop);
        } else {
          basic_loop(data, strides, 0, n, op);
        }
      });
    }
  });
}

template <int64_t n, typename func_t, typename... Args>
struct dim_apply_helper {
  static inline void
  apply(char* data[], const int64_t* strides, func_t op, Args... args) {
    using traits = function_traits<func_t>;
    using ptr_t = typename traits::template arg<2 * (n - 1)>::type;
    using stride_t = typename traits::template arg<2 * (n - 1) + 1>::type;
    static_assert(std::is_same<stride_t, int64_t>::value, "type for strides must be int64_t");
    dim_apply_helper<n - 1, func_t, ptr_t, int64_t, Args...>::apply(data, strides, op, (ptr_t)(data[n - 1]), strides[n - 1], args...);
  }
};

template <typename func_t, typename... Args>
struct dim_apply_helper<0, func_t, Args...> {
  static inline void
  apply(char* data[], const int64_t* strides, func_t op, Args... args) {
    op(args...);
  }
};

template <typename func_t>
static inline void
dim_apply(char* data[], const int64_t* strides, func_t op) {
  using traits = function_traits<func_t>;
  static_assert(std::is_same<typename traits::result_type, void>::value, "return type must be void");
  constexpr int ntensors = traits::arity / 2;
  // use template metaprogramming to do:
  // op((scalar0_t *)data[0], strides[0], (scalar1_t *)data[1], strides[1], ...);
  dim_apply_helper<ntensors, func_t>::apply(data, strides, op);
}

template <typename func_t>
void cpu_apply_dim_kernel(TensorIterator& iter, func_t op) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(iter.ntensors() >= traits::arity / 2);

  iter.for_each([&](char** data, const int64_t* strides) {
    dim_apply(data, strides, op);
  });
}

}}}  // namespace at::native::<anonymous>

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
