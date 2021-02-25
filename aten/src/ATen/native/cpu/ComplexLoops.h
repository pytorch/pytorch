#pragma once

#include <ATen/native/cpu/Loops.h>

namespace at { namespace native { namespace {

using namespace vec256;

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_vec_complex_impl(char* C10_RESTRICT data[],
                     const typename traits::template arg<0>::type opt_scalar,
                     size_t S,
                     int64_t i,
                     std::index_sequence<INDEX...>) {
  using Vec = typename traits::template arg<0>::type;
  using scalar_t = typename Vec::value_type;
  return std::make_tuple(
      S == INDEX + 1 ?
      opt_scalar :
      Vec::loadu(data[INDEX] + i * sizeof(scalar_t))...);
}

template <typename traits>
typename traits::ArgsTuple
dereference_vec_complex(char* C10_RESTRICT data[], const typename traits::template arg<0>::type opt_scalar,
                size_t S, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_vec_complex_impl<traits>(data, opt_scalar, S, i, Indices{});
}

template <typename func_t>
static inline void
execute_op_complex(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    result_type* out_ptr = (result_type*)(data[0] + 2 * i * strides[0]);
    *out_ptr = c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[1],
        &strides[1],
        i));
  }
}

// Basic loop operation (two outputs, N inputs). May be auto-vectorized
// by the compiler. Supports inputs and outputs of different types.
template <typename func_t>
static inline void
basic_loop_complex(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (int arg = 0; arg < ntensors; arg++) {
    strides[arg] = strides_[arg];
  }

  execute_op_complex(data, strides, i, n, std::forward<func_t>(op));
}

// Explicitly vectorized loop implementation.
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop_complex(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  using traits = function_traits<vec_func_t>;
  using scalar_t = typename function_traits<func_t>::result_type;
  using input_t = typename function_traits<func_t>::template arg<0>::type;
  using Vec_input = Vec256<input_t>;
  using Vec_output = Vec256<scalar_t>;
  constexpr int ntensors = traits::arity + 1;

  char* C10_RESTRICT data[ntensors];
  for (int arg = 0; arg < ntensors; arg++) {
    data[arg] = data_[arg];
  }
  Vec_input opt_scalar = Vec_input(S > 0 ? *(input_t*)data[S] : input_t(0));
  int64_t i = 0;
  int64_t j = 0;
  for (; i <= n - 2 * Vec_input::size(); i += 2 * Vec_input::size(), j += 4 * Vec_output::size()) {
  //   // const typename traits::template arg<INDEX>::type& opt_scalar,
    auto args1 = dereference_vec_complex<traits>(&data[1], opt_scalar, S, i);
    auto args2 = dereference_vec_complex<traits>(&data[1], opt_scalar, S, i + Vec_input::size());
    auto out1 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args1));
    auto out2 = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args2));
    std::get<0>(out1).store(data[0] + j * sizeof(scalar_t));
    std::get<1>(out1).store(data[0] + (j + Vec_output::size()) * sizeof(scalar_t));
    std::get<0>(out2).store(data[0] + (j + 2 * Vec_output::size()) * sizeof(scalar_t));
    std::get<1>(out2).store(data[0] + (j + 3 * Vec_output::size()) * sizeof(scalar_t));
  }
  if (i < n) {
    int64_t strides[ntensors];
    for (int arg = 0; arg < ntensors; arg++) {
      strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(input_t);
    }
    basic_loop_complex(data, strides, i, n, std::forward<func_t>(op));
  }
}

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec_complex(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop) {
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
      vectorized_loop_complex(data, n, 0, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          vectorized_loop_complex(data, n, idx, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
        } else {
          basic_loop_complex(data, strides, 0, n, std::forward<func_t>(op));
        }
      });
    }
  });
  iter.cast_outputs();
}

}}}  // namespace at::native::<anonymous>

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
