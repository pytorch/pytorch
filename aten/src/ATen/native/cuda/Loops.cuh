#pragma once

// This file provides three functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//   gpu_apply_dim_kernel(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel paramter instead of copying to device memory.
// This should be  used in conjuction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//
// The gpu_apply_dim_kernel helps for writing dimension apply. For example, if you want
// to implement gather_out(result, dim, index, src), you may write:
//
//     cpu_apply_dim_kernel(iter,
//       [=] GPU_LAMBDA (float *result_data, int64_t result_stride, int64_t *index_data, int64_t index_stride, float *src_data, int64_t src_stride) {
//         for (int64_t i = 0; i < size; i++) {
//           int64_t index = *(index_data + i * index_stride);
//           *(result_data + i * result_stride) = *(src_data + index * src_stride);
//         }
//       });

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/macros/Macros.h>

// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type) \
  static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

#ifdef __HIP_PLATFORM_HCC__
static constexpr int launch_size_1d = 1024;
static constexpr int launch_size_nd = 1024;
static constexpr int launch_bound2 = 1;
#else
static constexpr int launch_size_1d = 512;
static constexpr int launch_size_nd = 128;
static constexpr int launch_bound2 = 4;
#endif


namespace at { namespace native {

template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int N, typename index_t = uint32_t>
static OffsetCalculator<N, index_t> make_offset_calculator(const TensorIterator& iter) {
  AT_ASSERT(N == iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N, index_t>(iter.ndim(), iter.shape().data(), strides.data());
}

template<int nt, int vt, typename func_t>
static void launch_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename traits, typename index_t, std::size_t... I>
C10_HOST_DEVICE typename traits::ArgsTuple
dereference_impl(char* const C10_RESTRICT data[], const index_t strides[], int i,
                 c10::guts::index_sequence<I...>) {
  return std::make_tuple(
      *(typename traits::template arg<I>::type*)
        (data[I] + i * strides[I])...);
}

template <typename traits, typename index_t>
C10_HOST_DEVICE typename traits::ArgsTuple
dereference(char* const C10_RESTRICT data[], const index_t strides[], int i) {
  using Indices = c10::guts::make_index_sequence<traits::arity>;
  return dereference_impl<traits>(data, strides, i, Indices{});
}

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

  int64_t numel = iter.numel();
  if (iter.is_trivial_1d()) {
    auto inner_strides = iter.get_inner_strides();
    at::detail::Array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      strides[i] = inner_strides[i];
    }
    launch_kernel<launch_size_1d, 1>(numel, [=]__device__(int idx) {
      arg0_t* out = (arg0_t*)(data[0] + strides[0] * idx);
      *out = c10::guts::apply(f, dereference<traits>(
          &data.data[1],
          &strides.data[1],
          idx));
    });
  } else {
    auto offset_calc = make_offset_calculator<traits::arity + 1>(iter);
    launch_kernel<launch_size_nd, launch_bound2>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      *out = c10::guts::apply(f, dereference<traits>(
          &data.data[1],
          &offsets.data[1],
          1));
    });
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

template <typename func_t>
using OffsetCalculatorForFunc = OffsetCalculator<function_traits<func_t>::arity / 2, int64_t>;

template <typename func_t, typename T>
using ArrayForFunc = at::detail::Array<T, function_traits<func_t>::arity / 2>;

template <int64_t n, typename func_t, typename... Args>
struct gpu_dim_apply_helper {
  C10_HOST_DEVICE static inline void
  apply(const ArrayForFunc<func_t, char*> &data, const ArrayForFunc<func_t, int64_t> &strides, func_t op, Args... args) {
    using traits = function_traits<func_t>;
    using ptr_t = typename traits::template arg<2 * (n - 1)>::type;
    using stride_t = typename traits::template arg<2 * (n - 1) + 1>::type;
    static_assert(std::is_same<stride_t, int64_t>::value, "type for strides must be int64_t");
    gpu_dim_apply_helper<n - 1, func_t, ptr_t, int64_t, Args...>::apply(data, strides, op, (ptr_t)(data[n - 1]), strides[n - 1], args...);
  }
};

template <typename func_t, typename... Args>
struct gpu_dim_apply_helper<0, func_t, Args...> {
  C10_HOST_DEVICE static inline void
  apply(const ArrayForFunc<func_t, char*> &data, const ArrayForFunc<func_t, int64_t> &strides, func_t op, Args... args) {
    op(args...);
  }
};

template <typename func_t>
__global__ void gpu_dim_apply(
  int64_t numel, OffsetCalculatorForFunc<func_t> calc, ArrayForFunc<func_t, char *> base_data,
  ArrayForFunc<func_t, int64_t> strides, const func_t& op)
{
  using traits = function_traits<func_t>;
  constexpr int64_t ntensors = traits::arity / 2;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t gridsize = gridDim.x * blockDim.x;
  for (int64_t linear_id = tid; linear_id < numel; linear_id += gridsize) {
    auto data = base_data;
    auto offsets = calc.get(linear_id);
    #pragma unroll
    for (int64_t i = 0; i < ntensors; i++) {
      data[i] += offsets[i];
    }
    // use template metaprogramming to do:
    // op((scalar0_t *)data[0], strides[0], (scalar1_t *)data[1], strides[1], ...);
    gpu_dim_apply_helper<ntensors, func_t>::apply(data, strides, op);
  }
}

template <typename func_t>
void gpu_apply_dim_kernel(TensorIterator& iter, const func_t& op) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);
  using traits = function_traits<func_t>;
  constexpr int64_t ntensors = traits::arity / 2;
  TORCH_INTERNAL_ASSERT(iter.ntensors() >= ntensors);
  auto offset_calc = make_offset_calculator<ntensors, int64_t>(iter);

  int64_t numel = iter.numel();
  ArrayForFunc<func_t, char *> data;
  for (int64_t i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }
  ArrayForFunc<func_t, int64_t> strides;
  for (int64_t i = 0; i < ntensors; i++) {
    strides[i] = iter.strides(i)[0] / iter.element_size(i);
  }

  int64_t block = launch_size_1d;
  int64_t grid = (numel + launch_size_1d - 1) / launch_size_1d;
  auto stream = at::cuda::getCurrentCUDAStream();

  gpu_dim_apply<func_t><<<grid, block, 0, stream>>>(numel, offset_calc, data, strides, op);
  AT_CUDA_CHECK(cudaGetLastError());
}

}} // namespace at::native
