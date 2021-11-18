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

#include <type_traits>
#include <tuple>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/macros/Macros.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>
#include <c10/util/C++17.h>

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


namespace at { namespace native {

template<int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  int remaining = N - block_work_size() * blockIdx.x;

  if (remaining < block_work_size()) {  // if this block handles the reminder, just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<array_t, decltype(input_calc), decltype(output_calc),
                                           memory::LoadWithoutCast, memory::StoreWithoutCast>(
      data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else {  // if this block has a full `block_work_size` data to handle, use vectorized memory access
    elementwise_kernel_helper(f, memory::policies::vectorized<vec_size, array_t>(data));
  }
}

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(int N, func_t f, array_t data,
                                            inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  int remaining = N - block_work_size() * blockIdx.x;
  auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t>(data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template<int arity, typename func_t, typename array_t>
static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();

  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  switch (vec_size) {
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template<char const *name,
         typename result_type,
         typename compute_type,
         typename array_t,
         typename inp_calc_t,
         typename out_calc_t,
         typename loader_t,
         typename storer_t>
static inline void launch_jitted_unrolled_kernel(
  int64_t N, const std::string& f, array_t data,
  inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s, bool contiguous) {

  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size - 1) / block_work_size;

  // std::cout << "jitting is true!" << std::endl;
  // std::cout << "launch_unrolled_kernel!" << std::endl;
  static at::cuda::jit::NvrtcFunction fn;
  if (!fn.function) {
//    std::cout << "generating code\n";
    constexpr int nTensors = array_t::size();
    constexpr bool dynamic_casting = !std::is_same<decltype(l), memory::LoadWithoutCast>() || !std::is_same<decltype(s), memory::StoreWithoutCast>();
    // std::string common_type_str = at::cuda::jit::typeName<compute_type>();
    // std::string result_type_str = at::cuda::jit::typeName<result_type>();
    std::string string_name{name};
    std::string compute_type_str = at::cuda::jit::typeName<compute_type>();
    std::string result_type_str = at::cuda::jit::typeName<result_type>();
    auto code = at::cuda::jit::generate_code(nTensors, f, string_name,
                                             compute_type_str, result_type_str,
                                             contiguous, dynamic_casting);
    // std::cout << "generated code: " << std::endl;
    // std::cout << code << std::endl;
    fn = at::cuda::jit::jit_pwise_function(code, name); // TODO proper name
  }

  // packs args
  std::array<void*, 6> args = {
    (void*)&N,
    (void*)&data,
    (void*)&ic,
    (void*)&oc,
    (void*)&l,
    (void*)&s
  };
  at::cuda::jit::launch_jitted_pwise_function(fn, args, grid, num_threads());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<
  char const *name,
  typename result_type,
  typename compute_type,
  int arity,
  typename array_t>
static inline void launch_jitted_vectorized_kernel(int64_t N, const std::string& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size - 1) / block_work_size;

  //std::cout << "launch_jitted_vectorized_kernel!" << std::endl;
  const int vec_size = memory::jitted_can_vectorize_up_to<result_type, compute_type, arity>(data);
  //std::cout << "jitted vec_size is " << vec_size << std::endl;

  //TODO handle 2 and 4 vec sizes, so far only one vecsize will be compiled (4)
  static at::cuda::jit::NvrtcFunction fn4;
  static at::cuda::jit::NvrtcFunction fn2;
  static at::cuda::jit::NvrtcFunction fn1;
  at::cuda::jit::NvrtcFunction * fn_ptr;
  if (vec_size == 4) {
    fn_ptr = &fn4;
  } else if (vec_size == 2) {
    fn_ptr = &fn2;
  } else if (vec_size ==1) {
    fn_ptr = &fn1;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected vec_size for jitter vectorized kernel");
  }

  bool vectorized = vec_size > 1;

  if (!fn_ptr->function) {
//    std::cout << "generating code\n";
    constexpr int nTensors = array_t::size();
    std::string string_name{name};
    std::string compute_type_str = at::cuda::jit::typeName<compute_type>();
    std::string result_type_str = at::cuda::jit::typeName<result_type>();
    auto code = at::cuda::jit::generate_code(nTensors, f, string_name,
                                              compute_type_str, result_type_str,
                                              /*contiguous=*/true, /*dynamic_casting=*/false,
                                              vectorized, vec_size);
    //std::cout << "generated code: " << std::endl;
    //std::cout << code << std::endl;
    std::string kernel_name = vectorized ? string_name + "_vectorized" + std::to_string(vec_size) : string_name;
    *fn_ptr = at::cuda::jit::jit_pwise_function(code, kernel_name);
  }

  if (vectorized) {
    std::array<void*, 6> args = {
      (void*)&N,
      (void*)&data,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    };
    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto ic = TrivialOffsetCalculator<arity>();
    auto oc = TrivialOffsetCalculator<1>();
    auto l = memory::LoadWithoutCast();
    auto s = memory::StoreWithoutCast();

    std::array<void*, 6> args = {
      (void*)&N,
      (void*)&data,
      (void*)&ic,
      (void*)&oc,
      (void*)&l,
      (void*)&s
    };
    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

}

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
static inline void launch_unrolled_kernel(
  int64_t N, const func_t& f, array_t data,
  inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s, bool contiguous) {

  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <char const *name, typename result_type, typename compute_type, int arity>
void jitted_gpu_kernel_impl(TensorIteratorBase& iter, const std::string& f, const bool dynamic_casting) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  constexpr int ntensors = arity + 1;
  at::detail::Array<char*, ntensors> data;
  for (auto i = decltype(ntensors){0}; i < ntensors; ++i) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  if (!dynamic_casting) {
    if (contiguous) {
//      std::cout << "jitted_gpu_kernel_impl !dynamic_casting and contiguous" << std::endl;
      launch_jitted_vectorized_kernel<name, result_type, compute_type, arity>(numel, f, data);
    } else {
//      std::cout << "jitted_gpu_kernel_impl !dynamic_casting " << std::endl;
      auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      launch_jitted_unrolled_kernel<name, result_type, compute_type>(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer, contiguous);
    }
  } else {
    at::detail::Array<ScalarType, arity> dtypes;
    for (auto i = decltype(arity){0}; i < arity; ++i) {
      dtypes[i] = iter.dtype(i + 1);
    }
    auto loader = memory::LoadWithCast<arity>(dtypes);
    auto storer = memory::StoreWithCast(iter.dtype(0));
    if (contiguous) {
//      std::cout << "jitted_gpu_kernel_impl dynamic_casting and contiguous" << std::endl;
      auto input_offset_calculator = TrivialOffsetCalculator<arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_jitted_unrolled_kernel<name, result_type, compute_type>(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer, contiguous);
    } else {
//      std::cout << "jitted_gpu_kernel_impl dynamic_casting" << std::endl;
      auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      launch_jitted_unrolled_kernel<name, result_type, compute_type>(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer, contiguous);
    }
  }
}

template<int arity, typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f, const bool dynamic_casting) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  constexpr int ntensors = arity + 1;
  at::detail::Array<char*, ntensors> data;
  for (auto i = decltype(ntensors){0}; i < ntensors; ++i) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  if (!dynamic_casting) {
    if (contiguous) {
      launch_vectorized_kernel<arity>(numel, f, data);
    } else {
      auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer, contiguous);
    }
  } else {
    at::detail::Array<ScalarType, arity> dtypes;
    for (auto i = decltype(arity){0}; i < arity; ++i) {
      dtypes[i] = iter.dtype(i + 1);
    }
    auto loader = memory::LoadWithCast<arity>(dtypes);
    auto storer = memory::StoreWithCast(iter.dtype(0));
    if (contiguous) {
      auto input_offset_calculator = TrivialOffsetCalculator<arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer, contiguous);
    } else {
      auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer, contiguous);
    }
  }
}

}} // namespace at::native
