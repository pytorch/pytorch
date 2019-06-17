#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
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

template<int N>
static OffsetCalculator<N> make_offset_calculator(const TensorIterator& iter) {
  AT_ASSERT(N == iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
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

template<typename func_t>
void gpu_nullary_kernel(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_nullary_kernel(sub_iter, f);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);

  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    launch_kernel<launch_size_1d, 1>(numel, [=]__device__(int idx) {
      arg0_t* out = (arg0_t*)&out_data[stride0 * idx];
      *out = f();
    });
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    launch_kernel<launch_size_nd, launch_bound2>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)&out_data[offsets[0]];
      *out = f();
    });
  }
}

template<typename func_t>
void gpu_unary_kernel(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_unary_kernel(sub_iter, f);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  const char* in1_data = (char*)iter.data_ptr(1);

  using traits = unary_function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::arg1_t;

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }
  if (iter.is_cpu_scalar(1)) {
    auto a = iter.scalar_value<arg1_t>(1);
    iter.remove_operand(1);
    gpu_nullary_kernel(iter, [=]GPU_LAMBDA(void) {
      return f(a);
    });
  } else if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    int stride1 = strides[1];
    launch_kernel<launch_size_1d, 1>(numel, [out_data, stride0, stride1, in1_data, f]__device__(int idx) {
      arg0_t* out = (arg0_t*)&out_data[stride0 * idx];
      arg1_t* in1 = (arg1_t*)&in1_data[stride1 * idx];
      *out = f(*in1);
    });
  } else {
    auto offset_calc = make_offset_calculator<2>(iter);
    launch_kernel<launch_size_nd, launch_bound2>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)&out_data[offsets[0]];
      arg1_t* in1 = (arg1_t*)&in1_data[offsets[1]];
      *out = f(*in1);
    });
  }
}

template<typename func_t>
void gpu_binary_kernel(TensorIterator& iter, const func_t& f) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_binary_kernel(sub_iter, f);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  const char* in1_data = (char*)iter.data_ptr(1);
  const char* in2_data = (char*)iter.data_ptr(2);

  using traits = binary_function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::arg1_t;
  using arg2_t = typename traits::arg2_t;

  int numel = iter.numel();
  if (numel == 0) {
    return;
  }
  if (iter.is_cpu_scalar(1)) {
    auto a = iter.scalar_value<arg1_t>(1);
    iter.remove_operand(1);
    gpu_unary_kernel(iter, [=]GPU_LAMBDA(arg2_t b) {
      return f(a, b);
    });
  } else if (iter.is_cpu_scalar(2)) {
    auto b = iter.scalar_value<arg2_t>(2);
    iter.remove_operand(2);
    gpu_unary_kernel(iter, [=]GPU_LAMBDA(arg1_t a) {
      return f(a, b);
    });
  } else if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    int stride1 = strides[1];
    int stride2 = strides[2];
    launch_kernel<launch_size_1d, 1>(numel, [stride0, stride1, out_data, in1_data, f, stride2, in2_data]__device__(int idx) {
      arg0_t* out = (arg0_t*)&out_data[stride0 * idx];
      arg1_t* in1 = (arg1_t*)&in1_data[stride1 * idx];
      arg2_t* in2 = (arg2_t*)&in2_data[stride2 * idx];
      *out = f(*in1, *in2);
    });
  } else {
    auto offset_calc = make_offset_calculator<3>(iter);
    launch_kernel<launch_size_nd, launch_bound2>(numel, [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      arg0_t* out = (arg0_t*)&out_data[offsets[0]];
      arg1_t* in1 = (arg1_t*)&in1_data[offsets[1]];
      arg2_t* in2 = (arg2_t*)&in2_data[offsets[2]];
      *out = f(*in1, *in2);
    });
  }
}

}} // namespace at::native
