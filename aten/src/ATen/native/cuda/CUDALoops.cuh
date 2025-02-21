#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
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

#include <iostream>
#include <tuple>
#include <type_traits>

#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeCast.h>

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type)                       \
  static_assert(                                              \
      __nv_is_extended_host_device_lambda_closure_type(type), \
      #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

namespace at::native {


template <typename args_t, size_t... Is>
constexpr auto sum_of_sizes(args_t args, std::index_sequence<Is...>) {
    if constexpr (sizeof...(Is) == 0) {
      return 0;
    } else {
      return (sizeof(std::tuple_element_t<Is, args_t>) + ...);
    }
}

template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else if constexpr (io_sizes < 4) {
    return 8;
  } else {
    return 4;
  }
}

template <int io_sizes>
constexpr auto io_block_work_size() {
  return num_threads() * elems_per_thread<io_sizes>();
}

#ifdef USE_ROCM
template <typename args_t, size_t... Is>
constexpr auto input_size(args_t args, std::index_sequence<Is...>) {
  if constexpr (sizeof...(Is) == 0) {
    return 0;
  } else {
    return sizeof(std::tuple_element_t<0, args_t>);
  }
}

template <int vec_size, int io_size>
constexpr auto calc_optimal_vec_size() {
  static_assert(vec_size != 0);
  static_assert(io_size != 0);
  if constexpr (io_size == 1 && vec_size >= 16) {
    return 16;
  } else if constexpr (io_size <= 2 && vec_size >= 8) {
    return 8;
  } else if constexpr (io_size <= 4 && vec_size >= 4) {
    return 4;
  } else if constexpr (vec_size >= 4) {
    return 4;
  } else if constexpr (vec_size >= 2) {
    return 2;
  } else {
    return 1;
  }
}
#endif

template <typename func_t>
constexpr auto calc_io_size(){
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
#ifdef USE_ROCM
  constexpr auto input_size = at::native::input_size(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return (input_size > 0) ? ((input_size < output_size) ? input_size : output_size) : output_size;
#else
  constexpr auto input_size = at::native::sum_of_sizes(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
  constexpr auto output_size = sizeof(typename traits::result_type);
  return input_size + output_size;
#endif
}

template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  int remaining = N - io_block_work_size<io_size>() * blockIdx.x;

  if (remaining < io_block_work_size<io_size>()) { // if this block handles the reminder,
                                       // just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast,
        elems_per_thread<io_size>()>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
#ifdef USE_ROCM
    constexpr auto optimal_vec_size = calc_optimal_vec_size<vec_size, io_size>();
#else
    constexpr auto optimal_vec_size = vec_size;
#endif
    elementwise_kernel_helper(
        f, memory::policies::vectorized<optimal_vec_size, array_t, elems_per_thread<io_size>()>(data));
  }
}

template <
    typename func_t,
    typename array_t,
    int elems_per_thread,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int remaining = N - elems_per_thread * num_threads() * blockIdx.x;
  auto policy = memory::policies::
      unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t, elems_per_thread>(
          data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  int64_t grid = (N + io_block_work_size<io_size>() - 1) / io_block_work_size<io_size>();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);

  switch (vec_size) {
#ifdef USE_ROCM
    case 16:
      vectorized_elementwise_kernel<16, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 8:
      vectorized_elementwise_kernel<8, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
#endif
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      unrolled_elementwise_kernel<func_t, array_t, elems_per_thread<io_size>()>
          <<<grid, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t, thread_work_size()>
      <<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
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

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
      dtypes[I], data[I] + i * strides[I])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data[1], &offsets[1], 1);
  });
}

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast(iter, f);
  }
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
#ifdef USE_ROCM
    std::array<ScalarType, ntensors> dtypes;
    auto inner_strides = iter.get_inner_strides();
    std::array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
      strides[i] = inner_strides[i];
    }
    launch_legacy_kernel<512, 1>(numel, [=]GPU_LAMBDA(int idx) {
      void* out = data[0] + strides[0] * idx;
      arg0_t result = invoke(f, &data[1], &strides[1], &dtypes[1], idx);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
#else
    auto loader = memory::LoadWithCast<traits::arity>(iter);
    auto storer = memory::StoreWithCast<1>(iter);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_unrolled_kernel(
        numel,
        f,
        data,
        input_offset_calculator,
        output_offset_calculator,
        loader,
        storer);
#endif
  } else {
    std::array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
  }
}

} // namespace at::native
