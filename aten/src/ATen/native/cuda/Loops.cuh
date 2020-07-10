
#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>

namespace at { namespace native {

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define THREAD_WORK_SIZE 4
#define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * num_threads)

constexpr int num_threads = NUM_THREADS;
constexpr int thread_work_size = THREAD_WORK_SIZE;
constexpr int block_work_size = BLOCK_WORK_SIZE;

template<int N>
static OffsetCalculator<N> make_input_offset_calculator(const TensorIterator& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - 1);
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + 1).data();
    element_sizes[i] = iter.element_size(i + 1);
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

static OffsetCalculator<1> make_output_offset_calculator(const TensorIterator& iter) {
  std::array<const int64_t*, 1> strides;
  strides[0] = iter.strides(0).data();
  int64_t element_size = iter.element_size(0);
  return OffsetCalculator<1>(iter.ndim(), iter.shape().data(), strides.data(), &element_size);
}

}}  // namespace at::native

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

template <typename func_t>
void gpu_kernel(TensorIterator& iter, const func_t& f) {

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

template<typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  __device__ return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  AUnaryFunctor(func_t f_, arg1_t a_): f(f_), a(a_) {}
  private:
    func_t f;
    arg1_t a;
};

template<typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  __device__ return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  BUnaryFunctor(func_t f_, arg2_t b_): f(f_), b(b_) {}
  private:
    func_t f;
    arg2_t b;
};

template <typename func_t>
void gpu_kernel_with_scalars(TensorIterator& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<func_t> af(f, iter.scalar_value<arg1_t>(1));
    iter.remove_operand(1);
    const OptionalDeviceGuard device_guard(device_of(iter.tensor(1)));
    gpu_kernel(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<func_t> bf(f, iter.scalar_value<arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel(iter, bf);
  } else {
    gpu_kernel(iter, f);
  }
}

}} //namespace at::native
