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

#include <array>
#include <tuple>
#include <type_traits>

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

#ifdef USE_ROCM
// Custom configuration for vectorized elementwise kernel
// with template instantiation.
namespace vectorized_templated_config {
constexpr int num_threads() {
  return 512;
}

constexpr int elems_per_thread() {
  return 32;
}

constexpr int block_work_size() {
  return elems_per_thread() * num_threads();
}
} // namespace vectorized_templated_config
#endif

template <typename args_t, size_t... Is>
constexpr auto sum_of_sizes(args_t args, std::index_sequence<Is...>) {
    if constexpr (sizeof...(Is) == 0) {
      return 0;
    } else {
      return (sizeof(std::tuple_element_t<Is, args_t>) + ...);
    }
}

#ifdef USE_ROCM
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
#else
template <int io_sizes>
constexpr auto elems_per_thread(){
  if constexpr (io_sizes == 1) {
    return 16;
  } else {
    return 8;
  }
}
#endif


//thread work size of 8 regresses the perf of elementwise kernel on cuda
//this doesn't change ROCm behavior as thread_work_size is already 4 on ROCm
constexpr int elementwise_thread_work_size() {return 4;}
constexpr int elementwise_block_work_size() {
  return elementwise_thread_work_size() * num_threads();
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

#ifndef USE_ROCM
// To save on binary size of libtorch_cuda.so, we split the vectorized_elementwise_kernel
// into two: one for vec_size=8 and one for vec_size=[2, 4], since vec8 is going to be
// used on sm_90 and sm_100 exclusively.
template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  if constexpr (vec_size == 8) {
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
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
      elementwise_kernel_helper(
      f, memory::policies::vectorized<vec_size, array_t, elems_per_thread<io_size>()>(data));
    }
#endif // __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
  } else {
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
      elementwise_kernel_helper(
      f, memory::policies::vectorized<vec_size, array_t, elems_per_thread<io_size>()>(data));
    }
  }
}

#else // USE_ROCM
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
    constexpr auto optimal_vec_size = calc_optimal_vec_size<vec_size, io_size>();
    elementwise_kernel_helper(
        f, memory::policies::vectorized<optimal_vec_size, array_t, elems_per_thread<io_size>()>(data));
  }
}
#endif // USE_ROCM

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
#ifdef USE_ROCM
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
#else
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = memory::can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  // Here we purposely omit vec8 for 1-byte data because of a bug in NVCC
  // that causes some numerical mismatches with uint8 on sm80 and sm90.
  // TODO: Revisit this after CUDA 12.8 update.
  cudaDeviceProp* p = at::cuda::getDeviceProperties(stream.device().index());
  const int computeCapability = p->major * 10 + p->minor;
  if (computeCapability != 90 && computeCapability != 100) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#endif
  switch (vec_size) {
#ifdef USE_ROCM
    case 16:
      vectorized_elementwise_kernel<16, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
#endif
    case 8:
      vectorized_elementwise_kernel<8, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
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

#ifdef USE_ROCM
template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    typename OutputType,
    typename... InputTypes>
C10_LAUNCH_BOUNDS_1(vectorized_templated_config::num_threads())
__global__ void vectorized_templated_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t inp_calc,
    out_calc_t out_calc,
    loader_t loader,
    storer_t storer) {
  int remaining = N -
      vectorized_templated_config::block_work_size() *
          (gridDim.x - blockIdx.x - 1);
  constexpr bool reverted_idx = true;

  if (remaining <
      vectorized_templated_config::block_work_size()) { // if this block handles
                                                        // the reminder,
    // just do a naive unrolled loop
    auto policy = memory::policies::unroll_base<
        vectorized_templated_config::num_threads(),
        array_t,
        inp_calc_t,
        out_calc_t,
        loader_t,
        storer_t,
        vectorized_templated_config::elems_per_thread()>(
        data, remaining, inp_calc, out_calc, loader, storer);
    elementwise_kernel_helper<reverted_idx>(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    auto policy = memory::policies::vectorized_templated<
        vec_size,
        array_t,
        vectorized_templated_config::elems_per_thread(),
        vectorized_templated_config::num_threads(),
        OutputType,
        InputTypes...>(data);
    elementwise_kernel_helper<reverted_idx>(f, policy);
  }
}

// This function assume trivial 1d and supports template specialization
// to avoid dynamic casting.
// Input vectorization size is based on runtime information, i.e.
// the actual data types of the input and output tensor and cannot
// be determined using the functor type, as in regular non-templated
// vectorized kernels. The caller is in charge of selecting the correct input
// vectorization length.
template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    typename OutputType,
    typename... InputTypes>
static inline void launch_vectorized_templated_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t grid = (N + vectorized_templated_config::block_work_size() - 1) /
      vectorized_templated_config::block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  switch (vec_size) {
    case 8:
      vectorized_templated_elementwise_kernel<
          8,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_templated_elementwise_kernel<
          4,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_templated_elementwise_kernel<
          2,
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          OutputType,
          InputTypes...>
          <<<grid, vectorized_templated_config::num_threads(), 0, stream>>>(
              N, f, data, ic, oc, l, s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    default:
      // vector size 1 is not handled as part of vectorize_templated kernel
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}
#endif

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

  int64_t grid = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size()>
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

#ifdef USE_ROCM
namespace {
template <
    typename TupleLike,
    typename FirstParamTy,
    typename SecondParamTy,
    size_t arity,
    size_t arg_num = 0>
struct check_binary_functor_types_for_specialization {
  constexpr static inline bool check() {
    if constexpr (arity != 2)
      return false;
    if constexpr (arg_num == 0) {
      using SelectedType = std::tuple_element_t<arg_num, TupleLike>;
      if constexpr (std::is_same_v<FirstParamTy, SelectedType>)
        return check_binary_functor_types_for_specialization<
            TupleLike,
            FirstParamTy,
            SecondParamTy,
            arity,
            arg_num + 1>::check();
    } else if constexpr (arg_num == 1) {
      using SelectedType2 = std::tuple_element_t<arg_num, TupleLike>;
      if constexpr (std::is_same_v<SecondParamTy, SelectedType2>)
        return check_binary_functor_types_for_specialization<
            TupleLike,
            FirstParamTy,
            SecondParamTy,
            arity,
            arg_num + 1>::check();
    }
    return false;
  }
};

// Bottom case: if we got this far, assume correct type matching except
// when there are no arguments (arity == 0).
template <
    typename TupleLike,
    typename FirstParamTy,
    typename SecondParamTy,
    size_t arity>
struct check_binary_functor_types_for_specialization<
    TupleLike,
    FirstParamTy,
    SecondParamTy,
    arity,
    arity> {
  constexpr static inline bool check() {
    if constexpr (arity != 0)
      return true;
    return false;
  }
};

template <typename TupleLike, typename FirstParamTy, typename SecondParamTy>
struct check_binary_functor_types_for_specialization<
    TupleLike,
    FirstParamTy,
    SecondParamTy,
    0,
    0> {
  constexpr static inline bool check() {
    return false;
  }
};

// The following is a list of type specializations for vectorized_templated
// elementwise kernel. The three types refer to runtime types of the output
// tensor, first tensor argument, and the second tensor argument used for a
// binary functor.
constexpr std::array rt_binary_specializations = {
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<BFloat16>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<BFloat16>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<Half>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<float>::value,
         c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<float>::value}),
    std::array<c10::ScalarType, 3>(
        {c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<Half>::value,
         c10::CppTypeToScalarType<float>::value})};

bool check_binary_rt_types_for_specialization(TensorIteratorBase& iter) {
  if (iter.ninputs() != 2)
    return false;
  for (auto spec : rt_binary_specializations)
    if (iter.dtype(0) == spec[0] && iter.input_dtype(0) == spec[1] &&
        iter.input_dtype(1) == spec[2])
      return true;
  return false;
}

template <int arg_index>
struct type_specialized_kernel_launcher {
  template <
      typename func_t,
      typename array_t,
      typename inp_calc_t,
      typename out_calc_t,
      typename loader_t,
      typename storer_t>
  static void apply(
      ScalarType ret_t,
      ScalarType arg0_t,
      ScalarType arg1_t,
      int64_t numel,
      func_t f,
      array_t data,
      inp_calc_t input_offset_calculator,
      out_calc_t output_offset_calculator,
      loader_t loader,
      storer_t storer) {
    if (ret_t == rt_binary_specializations[arg_index][0] &&
        arg0_t == rt_binary_specializations[arg_index][1] &&
        arg1_t == rt_binary_specializations[arg_index][2])
      launch_vectorized_templated_kernel<
          func_t,
          array_t,
          inp_calc_t,
          out_calc_t,
          loader_t,
          storer_t,
          decltype(c10::impl::ScalarTypeToCPPType<
                   rt_binary_specializations[arg_index][0]>::t),
          decltype(c10::impl::ScalarTypeToCPPType<
                   rt_binary_specializations[arg_index][1]>::t),
          decltype(c10::impl::ScalarTypeToCPPType<
                   rt_binary_specializations[arg_index][2]>::t)>(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
  }
};

} // namespace
#endif

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
    // Attempt to call specialized vectorized elementwise kernel
    // that enables interleaving.
    if (check_binary_rt_types_for_specialization(iter) &&
        memory::can_vectorize_up_to<func_t>(data) > 1) {
      // constexpr to reduce the amount of kernels generated for
      // vectorized templated elementwise and limit which functors are actually
      // applied to the load and store at compile time.
      using func_tuple = typename traits::ArgsTuple;
      if constexpr (
          std::is_same_v<float, arg0_t> && traits::arity == 2 &&
          check_binary_functor_types_for_specialization<
              func_tuple,
              float,
              float,
              traits::arity,
              /*arg_num=*/0>::check()) {
        // If we got here, we know we are in one of the specialized cases. We
        // need to translate the runtime type to a statically known type. This
        // is effectively hoisting to the host the switch over runtime type in
        // the kernel in fetch_and_cast. Loader, storer, offset calculators are
        // only needed for the reminder loop.
        auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
        auto output_offset_calculator = TrivialOffsetCalculator<1>();
        auto loader = memory::LoadWithCast<traits::arity>(iter);
        auto storer = memory::StoreWithCast<1>(iter);
        memory::detail::static_unroll<
            type_specialized_kernel_launcher,
            rt_binary_specializations.size()>::
            with_args(
                iter.dtype(0),
                iter.input_dtype(0),
                iter.input_dtype(1),
                numel,
                f,
                data,
                input_offset_calculator,
                output_offset_calculator,
                loader,
                storer);
        return;
      }
    }
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
