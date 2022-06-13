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
//     [](Vectorized<float> a, Vectorized<float> b) { return a * b; });
//
// See BinaryOpsKernel.cpp for the complete implementation
//
//

#include <stdint.h>
#include <c10/util/C++17.h>
#include <c10/util/Load.h>
#include <c10/util/irange.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cpu/IsContiguous.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cpu/vec/vec.h>

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif

namespace at { namespace native { inline namespace CPU_CAPABILITY {

using namespace vec;

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_impl(char* C10_RESTRICT data[], const int64_t* strides, int64_t i,
                 std::index_sequence<INDEX...>) {
  return std::make_tuple(
      c10::load<typename traits::template arg<INDEX>::type>(
          data[INDEX] + i * strides[INDEX])...);
}

template <typename traits>
typename traits::ArgsTuple
dereference(char* C10_RESTRICT data[], const int64_t* strides, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

template <typename T>
struct is_tuple_Impl : std::false_type {};

template <typename... U>
struct is_tuple_Impl<std::tuple <U...>> : std::true_type {};

template <typename T>
constexpr bool is_tuple() {
  return is_tuple_Impl<T>::value; //decay_t<T>
}

template <typename first, typename... Vec, std::size_t... INDEX>
inline void store_multi_vec(char* data, std::tuple <first, Vec...> outs, int64_t i, std::index_sequence<INDEX...>) {
  using scalar_t = typename first::value_type;
  (void)std::initializer_list<int>{(std::get<INDEX>(outs).store(data + (i + INDEX * first::size()) * sizeof(scalar_t)), 0)...};
  // int arr[] = {(std::get<INDEX>(outs).store(data + (i + INDEX * first::size()) * sizeof(scalar_t)), 0)...};
  // {(std::get<INDEX>(outs).store(data + (i + INDEX * first::size()) * sizeof(scalar_t)), 0)...};
}

template <typename Vec, std::size_t... INDEX>
auto
dereference_multi_vec_impl(char* data,
                           const Vec& opt_scalar,
                           bool scalar_vec,
                           std::index_sequence<INDEX...>) {
  using scalar_t = typename Vec::value_type;
  return std::make_tuple(
      scalar_vec ?
      opt_scalar :
      Vec::loadu(data + INDEX * sizeof(scalar_t))...);
}

template <typename Vec, std::size_t num, typename traits, std::size_t INDEX>
auto
dereference_multi_vec(char* C10_RESTRICT data,
                     const Vec& opt_scalar,
                     size_t S, 
                     int64_t i) {
  using scalar_t = typename Vec::value_type;
  using Indices = std::make_index_sequence<num>;
  return dereference_multi_vec_impl<Vec>(data + i * sizeof(scalar_t), opt_scalar, S == INDEX + 1, Indices{});

}

template <typename Vec, std::size_t num, typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_tuple_vec_impl(char* C10_RESTRICT data[],
                     const Vec& opt_scalar,
                     size_t S,
                     int64_t i,
                     std::index_sequence<INDEX...>) {
  return std::make_tuple(
      dereference_multi_vec<Vec, num, traits, INDEX>(data[INDEX], opt_scalar, S, i)...);
}

template <typename Vec, std::size_t num, typename traits>
typename traits::ArgsTuple
dereference_tuple_vec(char* C10_RESTRICT data[], const Vec& opt_scalar, size_t S, int64_t i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return dereference_tuple_vec_impl<Vec, num, traits>(data, opt_scalar, S, i, Indices{});
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
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

// the recursive variadic template for iterating over the returned tuple
template<class T, size_t N>
struct TupleOutput {
  static void handle(char *C10_RESTRICT data[], const int64_t *strides, int64_t i,
                     const T &tuple) {
    TupleOutput<T, N - 1>::handle(data, strides, i, tuple);

    auto output = std::get<N - 1>(tuple);
    using output_type = decltype(output);
    output_type * out_ptr = (output_type *)(data[N - 1] + i * strides[N - 1]);
    *out_ptr = output;
  }
};

// Base case for the above recursive template
template<class T>
struct TupleOutput<T, 1> {
  static void handle(char *C10_RESTRICT data[], const int64_t *strides, int64_t i,
                     const T &tuple) {
    auto output = std::get<0>(tuple);
    using output_type = decltype(output);
    output_type* out_ptr = (output_type *)(data[0] + i * strides[0]);
    *out_ptr = output;
  }
};

template<class... Args>
void handle_tuple_outputs(char* C10_RESTRICT data[],
                          const int64_t* strides,
                          int64_t i,
                          const std::tuple<Args...> &tuple) {
  TupleOutput<decltype(tuple), sizeof...(Args)>::handle(data, strides, i, tuple);
}

// Loop operation for `cpu_kernel_multiple_outputs`.
// 1. Use `c10::guts::apply` to make dynamic method invocation
//    for the lambda passed in `cpu_kernel_multiple_outputs`.
// 2. Iterate over the members of the returned tuple, set the corresponding
//    output tensor by the tuple member in `handle_tuple_outputs` function.
template <typename func_t>
static inline void
multiple_outputs_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;

  using result_type = typename traits::result_type;
  constexpr int num_outputs = std::tuple_size<result_type>::value;
  constexpr int ntensors = traits::arity + num_outputs;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  for (; i < n; i++) {
    auto output = c10::guts::apply(op, dereference<traits>(
      &data[num_outputs],
      &strides[num_outputs],
      i));
    handle_tuple_outputs(data, strides, i, output);
  }
}

template <typename func_t, typename vec_func_t, bool istuple>
struct vectorized_loop_core;

template <typename func_t, typename vec_func_t>
struct vectorized_loop_core<func_t, vec_func_t, true> {
  static inline void call(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
    using traits = function_traits<vec_func_t>;
    using result_type = typename function_traits<vec_func_t>::result_type;

    constexpr int ntensors = traits::arity + 1;

    char* C10_RESTRICT data[ntensors];
    for (const auto arg : c10::irange(ntensors)) {
      data[arg] = data_[arg];
    }
    
    using arg_type = typename traits::template arg<0>::type;
    using Vec_in = typename std::tuple_element<0, arg_type>::type;
    using Vec_out = typename std::tuple_element<0, result_type>::type;
    using input_scalar = typename Vec_in::value_type;
    using out_scalar = typename Vec_out::value_type;
    Vec_in opt_scalar = Vec_in(S > 0 ? *(input_scalar*)data[S] : input_scalar(0));

    constexpr int in_num = std::tuple_size<arg_type>::value;
    constexpr int out_num = std::tuple_size<result_type>::value;
    int64_t i = 0;
    for (; i <= n - in_num * Vec_in::size(); i += in_num * Vec_in::size()) {
      auto args = dereference_tuple_vec<Vec_in, in_num, traits>(&data[1], opt_scalar, S, i);
      auto outs = c10::guts::apply(std::forward<vec_func_t>(vop), std::move(args));
      store_multi_vec(data[0], outs, i, std::make_index_sequence<out_num>{});
    }

    if (i < n) {
      int64_t strides[ntensors];
      strides[0] = sizeof(out_scalar);
      for (const auto arg : c10::irange(1, ntensors)) {
        strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(input_scalar);
      }
      basic_loop(data, strides, i, n, std::forward<func_t>(op));
    }

  }
};

template <typename func_t, typename vec_func_t>
struct vectorized_loop_core<func_t, vec_func_t, false> {
  static inline void call(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
    using traits = function_traits<vec_func_t>;
    using scalar_t = typename function_traits<func_t>::result_type;
    using Vec = Vectorized<scalar_t>;
    constexpr int ntensors = traits::arity + 1;

    char* C10_RESTRICT data[ntensors];
    for (const auto arg : c10::irange(ntensors)) {
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
      for (const auto arg : c10::irange(ntensors)) {
        strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
      }
      basic_loop(data, strides, i, n, std::forward<func_t>(op));
    }
  }
};

// Explicitly vectorized loop implementation. All inputs and outputs must be
// the same type and contiguous with one exception: a single input may be
// a scalar (stride 0). It's position is indicated by the argument `S`. If `S`
// is 0, then there are no scalar inputs.
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  using traits = function_traits<vec_func_t>;
  using result_type = typename function_traits<vec_func_t>::result_type;
  constexpr bool istuple = is_tuple<result_type>();
  vectorized_loop_core<func_t, vec_func_t, istuple>::call(data_, n, S, std::forward<func_t>(op), std::forward<vec_func_t>(vop));
}


template <typename traits, typename cb_t>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* /*strides*/,
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

template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
  op_t op;
  vop_t vop;

  using traits = function_traits<op_t>;
  static constexpr int ntensors = traits::arity + 1;
  using data_t = std::array<char*, ntensors>;

  VectorizedLoop2d(const op_t &op, const vop_t &vop):
    op(op), vop(vop) {}

  static void advance(data_t &data, const int64_t *outer_strides) {
    for (const auto arg : c10::irange(data.size())) {
      data[arg] += outer_strides[arg];
    }
  }

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    data_t data;
    std::copy_n(base, ntensors, data.data());
    const int64_t *outer_strides = &strides[ntensors];

    if (is_contiguous<traits>(strides)) {
      for (const auto i : c10::irange(size1)) {
        (void)i;
        vectorized_loop(data.data(), size0, 0, op, vop);
        advance(data, outer_strides);
      }
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          for (const auto i : c10::irange(size1)) {
            (void)i;
            vectorized_loop(data.data(), size0, idx, op, vop);
            advance(data, outer_strides);
          }
        } else {
          for (const auto i : c10::irange(size1)) {
            (void)i;
            basic_loop(data.data(), strides, 0, size0, op);
            advance(data, outer_strides);
          }
        }
      });
    }
  }
};

template <typename op_t, typename vop_t>
VectorizedLoop2d<op_t, vop_t> make_vectorized_loop2d(
    const op_t &op, const vop_t &vop) {
  return VectorizedLoop2d<op_t, vop_t>(op, vop);
}

template <typename func_t>
void cpu_kernel(TensorIteratorBase& iter, func_t&& op, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    // basic loop can handle 1d slices with arbitrary strides, and 1d slices is all that
    // iter.for_each is ever sending to the loop lambda
      basic_loop(data, strides, 0, n, std::forward<func_t>(op));
  }, grain_size);
  iter.cast_outputs();
}

// This function helps write elementwise kernels that requires multiple outputs.
// It follows the similar structure of cpu_kernel.
// Instead of `basic_loop` function, a new `multiple_outputs_loop` function is
// manipulated to handle multiple return values.
// For now `needs_dynamic_casting` check is not added as the passed lambda (`func_t`)
// of `multiple_outputs_loop` returns `std::tuple` instead of `scalar_t`.
// The `gpu_kernel_multiple_outputs` is also implemented without this check,
// We could extend `needs_dynamic_casting` to support both `std::tuple` and
// `thrust::tuple` in the future.
template <typename func_t>
void cpu_kernel_multiple_outputs(TensorIteratorBase& iter, func_t&& op, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);

  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    multiple_outputs_loop(data, strides, 0, n, std::forward<func_t>(op));
  }, grain_size);
  iter.cast_outputs();
}

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU, but some kernels (like Fill)
  // explicitly dynamic_cast, so we give the opt-out of checking.
  c10::guts::if_constexpr<check_dynamic_cast>([&] {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  });

  iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
  iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIteratorBase& iter, func_t&& op, const Range& range) {
  using traits = function_traits<func_t>;
  constexpr bool result_void = std::is_void<typename traits::result_type>::value;
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity &&
                        ((result_void && iter.noutputs() == 0) || (!result_void && iter.noutputs() == 1)));
  // dynamic casting not currently supported on CPU
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    basic_loop(data, strides, 0, n, std::forward<func_t>(op));
  }, range);
  iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIteratorBase& iter, func_t&& op) {
  cpu_serial_kernel(iter, op, {0, iter.numel()});
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, const Range& range) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  iter.serial_for_each(make_vectorized_loop2d(op, vop), range);
  iter.cast_outputs();
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop) {
  cpu_serial_kernel_vec(iter, op, vop, {0, iter.numel()});
}

}}}  // namespace at::native::<anonymous>

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
