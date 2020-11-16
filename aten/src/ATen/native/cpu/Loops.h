#pragma once

// This file provides two functions to help write elementwise kernels:
//
//   cpu_kernel(TensorIterator iter, <lambda>)
//   cpu_kernel_vec(TensorIterator iter, <lambda>, <vec_lambda>)
//
// Both functions may generate vectorized code. The cpu_kernel implementation
// relies on the compiler's auto-vectorization. The cpu_kernel_vec
// implementation uses x86 SIMD intrinsics when available. These functions
// are only intended to be used in the ATen/native/cpu subdirectory, since files
// in other directories are not compiled with AVX/AVX2 enabled. See README.md
// for more details.
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
//

#include <stdint.h>
#include <c10/util/C++17.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cpu/IsContiguous.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cpu/vec256/vec256.h>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif

namespace at { namespace native { namespace {

using namespace vec256;

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_impl(char* C10_RESTRICT data[], const int64_t* strides, int64_t i,
                 std::index_sequence<INDEX...>) {
  return std::make_tuple(
      *(typename traits::template arg<INDEX>::type*)
        (data[INDEX] + i * strides[INDEX])...);
}

template <typename traits>
typename traits::ArgsTuple
dereference(char* C10_RESTRICT data[], const int64_t* strides, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_vec_impl(char* C10_RESTRICT data[],
                     const typename traits::result_type& opt_scalar,
                     size_t S,
                     int64_t i,
                     std::index_sequence<INDEX...>) {
  using Vec = typename traits::result_type;
  using scalar_t = typename Vec::value_type;
  return std::make_tuple(
      S == INDEX + 1 ?
      opt_scalar :
      Vec::loadu(data[INDEX] + i * sizeof(scalar_t))...);
}

template <typename traits>
typename traits::ArgsTuple
dereference_vec(char* C10_RESTRICT data[], const typename traits::result_type& opt_scalar, size_t S, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_vec_impl<traits>(data, opt_scalar, S, i, Indices{});
}

template <typename func_t,
    typename std::enable_if<!std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    *out_ptr = c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[1],
        &strides[1],
        i));
  }
}

template <typename func_t,
    typename std::enable_if<std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  for (; i < n; i++) {
    c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[0],
        &strides[0],
        i));
  }
}

// Basic loop operation (one output, N inputs). May be auto-vectorized
// by the compiler. Supports inputs and outputs of different types.
template <typename func_t>
static inline void
basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (int arg = 0; arg < ntensors; arg++) {
    strides[arg] = strides_[arg];
  }

  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

// Explicitly vectorized loop implementation. All inputs and outputs must be
// the same type and contiguous with one exception: a single input may be
// a scalar (stride 0). It's position is indicated by the argument `S`. If `S`
// is 0, then there are no scalar inputs.
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
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
    auto out1 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args1));
    auto out2 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args2));
    out1.store(data[0] + i * sizeof(scalar_t));
    out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
  }
  if (i < n) {
    int64_t strides[ntensors];
    for (int arg = 0; arg < ntensors; arg++) {
      strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
    }
    basic_loop(data, strides, i, n, std::forward<func_t>(op));
  }
}


template <typename traits, typename cb_t>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<>,
    cb_t&& cb) {
  cb(0);
}

template <typename traits, typename cb_t, size_t INDEX0, size_t ...INDEX>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<INDEX0, INDEX...>,
    cb_t&& cb) {
  if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
    cb(INDEX0 + 1);
  } else {
    unroll_contiguous_scalar_checks<traits>(strides, std::index_sequence<INDEX...>{}, std::forward<cb_t>(cb));
  }
}

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t&& op) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      basic_loop(data, strides, 0, n, std::forward<func_t>(op));
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t _idx) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
      });
    }
  });
  iter.cast_outputs();
}

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU, but some kernels (like Fill)
  // explicitly dynamic_cast, so we give the opt-out of checking.
  c10::guts::if_constexpr<check_dynamic_cast>([&] {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  });

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      vectorized_loop(data, n, 0, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          vectorized_loop(data, n, idx, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
        } else {
          basic_loop(data, strides, 0, n, std::forward<func_t>(op));
        }
      });
    }
  });
  iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op, const Range& range) {
  using traits = function_traits<func_t>;
  constexpr bool result_void = std::is_void<typename traits::result_type>::value;
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity &&
                        ((result_void && iter.noutputs() == 0) || (!result_void && iter.noutputs() == 1)));
  // dynamic casting not currently supported on CPU
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      basic_loop(data, strides, 0, n, std::forward<func_t>(op));
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t _idx) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
      });
    }
  }, range);
  iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op) {
  cpu_serial_kernel(iter, op, {0, iter.numel()});
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop, const Range& range) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      vectorized_loop(data, n, 0, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          vectorized_loop(data, n, idx, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
        } else {
          basic_loop(data, strides, 0, n, std::forward<func_t>(op));
        }
      });
    }
  }, range);
  iter.cast_outputs();
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop) {
  cpu_serial_kernel_vec(iter, op, vop, {0, iter.numel()});
}

}}}  // namespace at::native::<anonymous>

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
