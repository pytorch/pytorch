
#pragma once

#include <ATen/detail/FunctionTraits.h>

namespace at { namespace native {

constexpr int num_threads = C10_WARP_SIZE * 2;
constexpr int thread_work_size = 4;
constexpr int block_work_size = thread_work_size * num_threads;

namespace modern { namespace detail {

template<typename func_t, int remaining=function_traits<func_t>::arity-1>
struct has_same_arg_types {
  using traits = function_traits<func_t>;
  static constexpr bool value = std::is_same<
      typename traits::template arg<remaining>::type,
      typename traits::template arg<remaining-1>::type
    >::value && has_same_arg_types<func_t, remaining-1>::value;
};

template<typename func_t>
struct has_same_arg_types<func_t, 0> {
  static constexpr bool value = true;
};

template<typename func_t>
struct has_same_arg_types<func_t, -1> {
  static constexpr bool value = true;
};

}}}} // namespace at::native::modern::detail

// Note:
// CUDA and ROCm get diverged in this PR:
//   https://github.com/pytorch/pytorch/pull/32383
// Because for some reason trying to enable vectorized
// memory access introduce regression on ROCm.

#ifndef __HIP_PLATFORM_HCC__
#include <ATen/native/cuda/CUDALoops.cuh>
#else
#include <ATen/native/cuda/ROCmLoops.cuh>
#endif

namespace at { namespace native {

// `needs_dynamic_casting` compares the types expected by iterator
// (i.e. dtypes of the operands) with the actual type of the arguments
// of func_t
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    if (iter.dtype(nargs) != c10::impl::CPPTypeToScalarType<typename traits::template arg<nargs - 1>::type>::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

template<typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    return iter.dtype(0) != c10::impl::CPPTypeToScalarType<typename traits::result_type>::value;
  }
};

template <typename func_t>
void gpu_kernel_impl(TensorIterator& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == traits::arity + 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  at::detail::Array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    dtypes[i] = iter.tensor(i).scalar_type();
  }

  int64_t numel = iter.numel();
  if (iter.is_trivial_1d()) {
    auto inner_strides = iter.get_inner_strides();
    at::detail::Array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      strides[i] = inner_strides[i];
    }

    if (needs_dynamic_casting<func_t>::check(iter)) {
      legacy::launch_kernel<launch_size_1d, 1>(numel, [=]GPU_LAMBDA(int idx) {
        void* out = data[0] + strides[0] * idx;
        arg0_t result = legacy::invoke(f, &data.data[1], &strides.data[1], &dtypes.data[1], idx);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      });
    } else if (iter.has_contiguous_first_dim() && modern::detail::has_same_arg_types<func_t>::value) {
      modern::launch_kernel(numel, f, data);
    } else {
      legacy::launch_kernel<launch_size_1d, 1>(numel, [=]GPU_LAMBDA(int idx) {
        arg0_t* out = (arg0_t*)(data[0] + strides[0] * idx);
        *out = legacy::invoke(f, &data.data[1], &strides.data[1], idx);
      });
    }
  } else {
    auto offset_calc = legacy::make_offset_calculator<traits::arity + 1>(iter);
    if (needs_dynamic_casting<func_t>::check(iter)) {
      legacy::launch_kernel<launch_size_nd, launch_bound2>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        void* out = data[0] + offsets[0];
        arg0_t result = legacy::invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      });
    } else {
      legacy::launch_kernel<launch_size_nd, launch_bound2>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = legacy::invoke(f, &data.data[1], &offsets.data[1], 1);
      });
    }
  }
}

template <typename func_t>
void gpu_kernel(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_cuda());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl(iter, f);
}

template <typename func_t>
void gpu_kernel_with_scalars(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    auto a = iter.scalar_value<arg1_t>(1);
    iter.remove_operand(1);
    gpu_kernel(iter, [=]GPU_LAMBDA(arg2_t b) {
      return f(a, b);
    });
  } else if (iter.is_cpu_scalar(2)) {
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    auto b = iter.scalar_value<arg2_t>(2);
    iter.remove_operand(2);
    gpu_kernel(iter, [=]GPU_LAMBDA(arg1_t a) {
      return f(a, b);
    });
  } else {
    gpu_kernel(iter, f);
  }
}

}} //namespace at::native
