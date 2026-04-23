#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <c10/core/DynamicCast.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/thread_constants.h>

#include <thrust/tuple.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at::native::memory {

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
  static inline C10_HOST_DEVICE void with_args(Args... /*args*/) {}
};

// helper structs to be used with static_unroll to load arguments
// one by one

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int idx, int block_work_size) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + block_work_size * idx;
    auto args_accessor = [&args] __device__ (int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
    self.load_single_arg(args_accessor, ptr);
  }
};

#ifdef USE_ROCM
// Templated version of vectorized load helper.
// It can be used on heterogeneous input tensor element types.
template <int arg_index>
struct vectorized_templated_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t& self, args_t* args, int idx) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input

    // Delay pointer arithmetic to the policy loader where we know the actual
    // type of the current argument.
    char* ptr = (self.data[arg_index + 1]);
    auto args_accessor = [&args] __device__(int thread_unroll_idx) -> arg_t& {
      return std::get<arg_index>(args[thread_unroll_idx]);
    };
    self.template load_single_arg<arg_index>(args_accessor, ptr, idx);
  }
};
#endif

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static __device__ void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<typename data_t, typename offsets_t, typename ...Args>
  C10_HOST_DEVICE static void apply(
      const data_t& data,
      const offsets_t& offsets,
      thrust::tuple<Args...> ret) {
    using T = typename thrust::tuple_element<current, thrust::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = thrust::get<current>(ret);
  }
};

template <int arg_index>
struct reset_helper {
  template <typename args_t>
  static __device__ void apply(args_t *args, int idx) {
    std::get<arg_index>(args[idx]) = {};
  }
};

}  // namespace detail

struct LoadWithoutCast {
  template<typename scalar_t>
  __device__ scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return c10::load(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

template <int N>
struct LoadWithCast {
  using array_t = std::array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  LoadWithCast(const TensorIteratorBase& iter) {
    CUDA_KERNEL_ASSERT(iter.ninputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i + iter.noutputs());
      element_sizes[i] = c10::elementSize(iter.dtype(i + iter.noutputs()));
    }
  }

  template<typename scalar_t>
  __device__ scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    void *ptr = base_ptr + element_sizes[arg] * offset;
    return c10::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
  }
};

struct StoreWithoutCast {
  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
    *(reinterpret_cast<scalar_t *>(base_ptr) + offset) = value;
  }
};

template <int N = 1>
struct StoreWithCast {
  using array_t = std::array<at::ScalarType, std::max<int>(N, 1)>;
  using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

  array_t dtypes;
  size_array_t element_sizes;

  StoreWithCast(const TensorIteratorBase& iter) {
    CUDA_KERNEL_ASSERT(iter.noutputs() == N);
    #pragma unroll
    for (auto i = 0; i < N; ++i) {
      this->dtypes[i] = iter.dtype(i);
      element_sizes[i] = c10::elementSize(iter.dtype(i));
    }
  }

  template<typename scalar_t>
  __device__ void store(scalar_t value, char *base_ptr, uint32_t offset, int arg = 0) {
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
__device__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
#if defined(USE_ROCM)
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
#else
  auto *from = static_cast<const vec_t *>(__builtin_assume_aligned(base_ptr, alignof(vec_t)));
#endif
#if defined(USE_ROCM) && defined(__gfx942__)
  using longx2 = __attribute__((__vector_size__(4*sizeof(int)))) int;
  if constexpr (sizeof(vec_t) == sizeof(int)) {
   union {
     vec_t v;
     int   i;
   } tmpt = { .i = __builtin_nontemporal_load(reinterpret_cast<const int *>(&(from[offset]))) };
   return tmpt.v;
  }
  else if constexpr (sizeof(vec_t) == sizeof(long)) {
   union {
     vec_t v;
     long   i;
   } tmpt = { .i = __builtin_nontemporal_load(reinterpret_cast<const long *>(&(from[offset]))) };
   return tmpt.v;
  }
  else if constexpr (sizeof(vec_t) == sizeof(longx2)) {
   union {
     vec_t v;
     longx2  i;
   } tmpt = { .i = __builtin_nontemporal_load(reinterpret_cast<const longx2 *>(&(from[offset]))) };
   return tmpt.v;
  }
#endif
  return from[offset];
}

template <int vec_size>
__device__ aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
  // See NOTE [Loading boolean values]
  auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
  aligned_vector<bool, vec_size> ret;
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = bool(tmp.val[i]);
  }
  return ret;
}

namespace policies {

template <
    int num_threads,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int elems_per_thread,
    int num_outputs = 1,
    bool check_compute_bounds = true>
struct unroll_base {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  static constexpr int tws = elems_per_thread;
  static constexpr int block_work_size = elems_per_thread * num_threads;

  __device__ unroll_base(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        loader(l),
        storer(s) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    if constexpr (!check_compute_bounds) {
      return true;
    } else {
      return ((int)(threadIdx.x + thread_work_elem * num_threads) < remaining);
    }
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = threadIdx.x;
    int base_idx = thread_idx + block_work_size * idx;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      int linear_idx = base_idx + i * num_threads;
      auto offset = input_offset_calculator.get(linear_idx);
      if (thread_idx < remaining) {
        detail::static_unroll<detail::unroll_load_helper, arity>::with_args(
            *this, args, offset, loader, i, num_outputs);
      } else {
        detail::static_unroll<detail::reset_helper, arity>::with_args(args, i);
      }
      thread_idx += num_threads;
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    int thread_idx = threadIdx.x;
    int base_idx = thread_idx + block_work_size * idx;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      int linear_idx = base_idx + i * num_threads;
      int offset = output_offset_calculator.get(linear_idx)[0];
      if (thread_idx < remaining) {
        storer.store(from[i], data[0], offset);
      }
      thread_idx += num_threads;
    }
  }
};

// Utility type for all users of unroll that extract the num_threads value from
// the caller scope.
template <
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int elems_per_thread,
    int num_outputs = 1>
using unroll = unroll_base<
    num_threads(),
    data_t,
    inp_calc_t,
    out_calc_t,
    loader_t,
    storer_t,
    elems_per_thread,
    num_outputs>;

template <int vec_size, typename data_t, int elems_per_thread>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(elems_per_thread % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = elems_per_thread / vec_size;
  static constexpr int tws = elems_per_thread;

  data_t data;

  __device__ vectorized(data_t data) : data(data) {}

  __device__ inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load_single_arg(accessor_t to, scalar_t *from) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      auto v = load_vector<vec_size>(from, index);
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, idx, elems_per_thread * num_threads());
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + elems_per_thread * num_threads() * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

// Advance each pointer in data by offset elements of its respective type.
template <typename result_t, typename args_tuple, typename array_t, int... Is>
__device__ inline void advance_data_impl(
    array_t& data, int offset,
    std::integer_sequence<int, Is...>) {
  data[0] += offset * int{sizeof(result_t)};
  ((data[Is + 1] += offset * int{sizeof(std::tuple_element_t<Is, args_tuple>)}), ...);
}

template <typename result_t, typename args_tuple, typename array_t>
__device__ inline void advance_data(array_t& data, int offset) {
  static_assert(std::tuple_size_v<args_tuple> == std::tuple_size_v<array_t> - 1,
    "data array must have one more element than the number of arguments");
  advance_data_impl<result_t, args_tuple>(
      data, offset,
      std::make_integer_sequence<int, std::tuple_size_v<args_tuple>>{});
}

// Like vectorized, but with a configurable thread count and bounds-checked
// loads/stores if necessary.
// note: also expects that pointers are always pre-advanced between loads/stores.
template <bool has_remaining, int vec_size_, int vectors_per_unroll_, int num_threads_, typename data_t>
struct streaming_vectorized {
  static constexpr int vec_size = vec_size_;
  static constexpr int vectors_per_unroll = vectors_per_unroll_;
  static constexpr int num_threads = num_threads_;

  data_t data;
  int remaining;

  __device__ streaming_vectorized(data_t data, int remaining)
      : data(data), remaining(remaining) {}

  template <typename result_t, typename args_tuple>
  __device__ inline void advance(int offset) {
    advance_data<result_t, args_tuple>(data, offset);
    if constexpr (has_remaining) {
      remaining -= offset;
    }
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load_single_arg(accessor_t to, scalar_t *from) {
    int index = threadIdx.x;
    using vec_t = aligned_vector<scalar_t, vec_size>;
    #pragma unroll
    for (int unroll_vec = 0; unroll_vec < vectors_per_unroll; unroll_vec++) {
      int idx = index + unroll_vec * num_threads;
      vec_t v;
      // note: avoid putting too much faith into compiler optimization by
      // using something like if (!has_remaining || index + i * vec_size < remaining)
      if constexpr (has_remaining) {
        v = idx * vec_size < remaining ? load_vector<vec_size>(from, idx) : vec_t{};
      } else {
        v = load_vector<vec_size>(from, idx);
      }
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(unroll_vec * vec_size + j) = v.val[j];
      }
    }
  }

  template<typename args_t>
  __device__ inline void load(args_t *args) {
    constexpr int arity = std::tuple_size_v<args_t>;
    // no need for index here as pointers are pre-advanced between loads/stores
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, 0, 0);
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int unroll_vec) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    vec_t *to = static_cast<vec_t *>(__builtin_assume_aligned(data[0], alignof(vec_t)));
    vec_t *from_ = reinterpret_cast<vec_t *>(from);
    int index = unroll_vec * num_threads + threadIdx.x;
    // note: avoid putting too much faith into compiler optimization by
    // using something like if (!has_remaining || index * store_vec_size < remaining)
    if constexpr (has_remaining) {
      if (index * vec_size < remaining) {
        to[index] = *from_;
      }
    } else {
      to[index] = *from_;
    }
  }
};

#ifdef USE_ROCM
// This is similar to vectorized policy above, but this one supports
// heterogeneous input tensor types as templated parameters.
// Its use should be limited to frequently used heterogeneous data types
// as each instantiation will generate a separate kernel, leading to code
// bloating if applied to all combinations supported in PyTorch. Assumption: all
// tensors are contiguous, that is: stride == sizeof(type) for all tensors.
template <
    int vec_size,
    typename data_t,
    int elems_per_thread,
    int num_threads,
    typename CastToT,
    typename... CastFromTs> // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized_templated {
  static_assert(
      elems_per_thread % vec_size == 0,
      "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = elems_per_thread / vec_size;
  static constexpr int tws = elems_per_thread;
  static constexpr int block_work_size = elems_per_thread * num_threads;
  data_t data;

  __device__ vectorized_templated(data_t data) : data(data) {}

  __device__ inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template <int arg_index, typename accessor_t>
  __device__ inline void load_single_arg(accessor_t to, char* ptr, int idx) {
    // extract the arg_index-th input tensor element type from the
    // variadic template argument.
    using CastFromT =
        std::tuple_element_t<arg_index, std::tuple<CastFromTs...>>;
    // Delayed pointer arithmetic from the caller: this is the place
    // where we know the type of the argument.
    CastFromT* block_ptr =
        reinterpret_cast<CastFromT*>(ptr) + block_work_size * idx;
    int thread_idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      auto v = load_vector<vec_size>(block_ptr, index);
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = c10::convert<CastToT>(v.val[j]);
      }
    }
  }

  template <typename args_t>
  __device__ inline void load(args_t* args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    detail::static_unroll<detail::vectorized_templated_load_helper, arity>::
        with_args(*this, args, idx);
  }

  // Assume for now that from (temporary array per thread) is of the same
  // type as to (destination tensor), which is the case for
  // float(float,bfloat16) and functor add on float(float,float).
  template <typename scalar_t>
  __device__ inline void store(scalar_t* from, int idx) {
    using vec_t = aligned_vector<CastToT, vec_size>;
    CastToT* to = reinterpret_cast<CastToT*>(data[0]) + block_work_size * idx;
    vec_t* to_ = reinterpret_cast<vec_t*>(to);
    int thread_idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};
#endif

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
  //multi_outputs_unroll struct members and check_inbounds and load methods are copypasted from unroll struct
  //we don't use inheritance because of compiler bug in cuda 10.2+
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;
  StoreWithoutCast storer;
  static constexpr int tws = thread_work_size();

  __device__ multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
  data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((int)(threadIdx.x  + thread_work_elem*num_threads()) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads();
    }
  }


  template <typename return_t>
  __device__ inline void store(return_t *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= this->remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      memory::detail::static_unroll<detail::multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      thread_idx += num_threads();
    }
  }
};

}  // namespace policies

// This is only used in host, but we will wrap this into some templates
// which is C10_HOST_DEVICE, so we have to make this C10_HOST_DEVICE
// in order to compile
template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(const char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of_v<aligned_vector<scalar_t, 2>>;
  constexpr int vec4_alignment = std::alignment_of_v<aligned_vector<scalar_t, 4>>;
  constexpr int vec8_alignment = std::alignment_of_v<aligned_vector<scalar_t, 8>>;
#ifdef USE_ROCM
  constexpr int vec16_alignment = std::alignment_of_v<aligned_vector<scalar_t, 16>>;
  constexpr int type_size = sizeof(scalar_t);
  if (type_size == 1 && (address % vec16_alignment == 0)) {
    return 16;
  } else if (type_size <= 2 && (address % vec8_alignment == 0)) {
    return 8;
  } else
#else
  if (address % vec8_alignment == 0) {
   return 8;
  } else
#endif
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(char *pointer) {
  return can_vectorize_up_to<scalar_t>(static_cast<const char*>(pointer));
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static C10_HOST_DEVICE void apply(int &result, array_t pointers, traits /*_*/) {
    using arg_t = typename traits::template arg<i>::type;
    // `pointers` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    result = std::min<int>(result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, pointers, traits());
  return result;
}



template <typename T>
__inline__ size_t get_alignment(T ptr_or_size) {
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0) {
    return 16;
  } else if (val % 8 == 0) {
    return 8;
  } else if (val % 4 == 0) {
    return 4;
  } else if (val % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

template <>
__inline__ size_t get_alignment<size_t>(size_t size) {
  return get_alignment(reinterpret_cast<void*>(size));
}

template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

template <int Size>
union Vec;

template <>
union Vec<4> {
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8> {
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16> {
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u128, as_scalar;
  float f32[4];
};

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> ld_vec(const T* addr) {
  Vec<Alignment> vec;
  if constexpr (Alignment == 16) {
#if defined(USE_ROCM)
    vec.u128 = *reinterpret_cast<const uint4*>(addr);
  } else if constexpr (Alignment == 8) {
    vec.u64 = *reinterpret_cast<const uint64_t*>(addr);
  } else if constexpr (Alignment == 4) {
    vec.u32 = *reinterpret_cast<const uint32_t*>(addr);
#else
    asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(vec.u32[0]), "=r"(vec.u32[1]), "=r"(vec.u32[2]), "=r"(vec.u32[3])
        : "l"(addr)
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("ld.global.v2.u32 {%0,%1}, [%2];"
        : "=r"(vec.u32[0]), "=r"(vec.u32[1])
        : "l"(addr)
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("ld.global.u32 %0, [%1];" : "=r"(vec.u32) : "l"(addr) : "memory");
#endif
  } else {
    static_assert(dependent_false<T>);
  }
  return vec;
}

template <int Alignment, typename T>
__device__ __inline__ void st_vec(T* addr, const Vec<Alignment>& vec) {
  if constexpr (Alignment == 16) {
#if defined(USE_ROCM)
    reinterpret_cast<uint64_t*>(addr)[0] = vec.u64[0];
    reinterpret_cast<uint64_t*>(addr)[1] = vec.u64[1];
  } else if constexpr (Alignment == 8) {
    *reinterpret_cast<uint64_t*>(addr) = vec.u64;
  } else if constexpr (Alignment == 4) {
    *reinterpret_cast<uint32_t*>(addr) = vec.u32;
#else
    asm("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(addr),
          "r"(vec.u32[0]),
          "r"(vec.u32[1]),
          "r"(vec.u32[2]),
          "r"(vec.u32[3])
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("st.global.v2.u32 [%0], {%1,%2};"
        :
        : "l"(addr), "r"(vec.u32[0]), "r"(vec.u32[1])
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("st.global.u32 [%0], %1;" : : "l"(addr), "r"(vec.u32) : "memory");
#endif
  } else {
    static_assert(dependent_false<T>);
  }
}



} // namespace at::native::memory
