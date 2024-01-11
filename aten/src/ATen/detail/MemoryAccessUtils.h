#pragma once

#include <c10/core/DynamicCast.h>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/native/TensorIterator.h>

#include <thrust/tuple.h>

namespace at { namespace native { namespace memory {

namespace detail {

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... args) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t, typename offset_t>
  static C10_DEVICE void apply(policy_t &self, args_t *args, offset_t offset) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + offset;
    auto args_accessor = [&args] C10_DEVICE (int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
    self.load_single_arg(args_accessor, ptr);
  }
};

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static C10_DEVICE void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<int ntensors, int num_outputs, typename ...Args>
  C10_HOST_DEVICE static void apply(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<uint32_t, num_outputs> offsets,
      thrust::tuple<Args...> ret) {
    using T = typename thrust::tuple_element<current, thrust::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = thrust::get<current>(ret);
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
  C10_DEVICE scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return c10::load(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

template <int N>
struct LoadWithCast {
  using array_t = at::detail::Array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = at::detail::Array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  LoadWithCast(const TensorIteratorBase& iter) {
    assert(iter.ninputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i + iter.noutputs());
      element_sizes[i] = c10::elementSize(iter.dtype(i + iter.noutputs()));
    }
  }

  template<typename scalar_t>
  C10_DEVICE scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    return c10::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
  C10_DEVICE void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

template <int N = 1>
struct StoreWithCast {
  using array_t = at::detail::Array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = at::detail::Array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  StoreWithCast(const TensorIteratorBase& iter) {
    assert(iter.noutputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i);
      element_sizes[i] = c10::elementSize(iter.dtype(i));
    }
  }

  template<typename scalar_t>
  C10_DEVICE void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    c10::cast_and_store<scalar_t>(dtypes[arg], ptr, value);
  }
};

// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <int vec_size, typename scalar_t>
C10_DEVICE aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}

template <int vec_size>
C10_DEVICE aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
  // See NOTE [Loading boolean values]
  auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
  aligned_vector<bool, vec_size> ret;
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = bool(tmp.val[i]);
  }
  return ret;
}

}}} // namespace at::native::memory
