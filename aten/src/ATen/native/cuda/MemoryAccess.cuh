#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/core/DynamicCast.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/detail/MemoryAccessUtils.h>
#include <ATen/native/cuda/thread_constants.h>

#include <thrust/tuple.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {
namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template<typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t, int num_outputs = 1>
struct unroll {

  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;

  __device__ unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((threadIdx.x  + thread_work_elem*num_threads()) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
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

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    int thread_idx = threadIdx.x;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + block_work_size() * idx;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
      thread_idx += num_threads();
    }
  }
};

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
// Note:
// Functions in vectorized policy does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the caller manually.
template <int vec_size, typename data_t>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(thread_work_size() % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = thread_work_size() / vec_size;

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
    constexpr int arity = std::tuple_size<args_t>::value;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, block_work_size() * idx);
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + block_work_size() * idx;
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

  __device__ multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
  data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((threadIdx.x  + thread_work_elem*num_threads()) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
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
inline C10_HOST_DEVICE int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

template<int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static C10_HOST_DEVICE void apply(int &result, array_t pointers, traits _) {
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

}}} // namespace at::native::memory
