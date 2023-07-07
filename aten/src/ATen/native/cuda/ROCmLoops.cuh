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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/macros/Macros.h>
#include <c10/core/ScalarType.h>
#include <c10/core/DynamicCast.h>


#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type) \
  static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

static constexpr int launch_size_1d = 512;
static constexpr int launch_size_nd = 128;
static constexpr int launch_bound2 = 4;


namespace at { namespace native {

// See [NOTE: Complex Operator Unification]
// std::complex and thrust::complex don't work with some !needs_dynamic_casting optimizations.
// They always currently map to !needs_dynamic_casting even though we sometimes rely on the ability
// to reinterpret_cast between these representations.
// In order to separate these concerns, we have a check for non-c10 complex separately.
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct uses_non_c10_complex {
  constexpr static bool check() {
    using traits = function_traits<func_t>;
    using type = typename traits::template arg<nargs - 1>::type;
    constexpr bool non_c10_complex =
        std::is_same<std::complex<float>, type>::value
        || std::is_same<std::complex<double>, type>::value
        || std::is_same<thrust::complex<float>, type>::value
        || std::is_same<thrust::complex<double>, type>::value;

    if constexpr (non_c10_complex) {
      return true;
    } else {
      return uses_non_c10_complex<func_t, nargs - 1>::check();
    }
  }
};

template<typename func_t>
struct uses_non_c10_complex<func_t, 0> {
  constexpr static bool check() {
    using traits = function_traits<func_t>;
    using type = typename traits::result_type;
    constexpr bool non_c10_complex =
        std::is_same<std::complex<float>, type>::value
        || std::is_same<std::complex<double>, type>::value
        || std::is_same<thrust::complex<float>, type>::value
        || std::is_same<thrust::complex<double>, type>::value;

    return non_c10_complex;
  }
};

// NOTE: @zasdfgbnm is currently working on rewriting the gpu loops.
// Some of the old codes has been moved to namespace legacy, and
// new codes will be put into namespace modern. These two namespaces
// will coexists for a while until the rewrite is done. Once the rewrite
// is done, we will remove the legacy and modern namespace and everything
// will be in at::native directly.
namespace legacy {

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i,
            std::index_sequence<INDEX...>) {
  return f(c10::load<typename traits::template arg<INDEX>::type>(data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], const ScalarType dtypes[], int i,
            std::index_sequence<I...>) {
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(dtypes[I], data[I] + i * strides[I])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], const ScalarType dtypes[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

} // namespace legacy

// See the note for namespace legacy above.
namespace modern {

namespace detail {

template <typename func_t, typename array_t, std::size_t... I>
__device__ inline constexpr decltype(auto) invoke_with_array_impl(func_t f, array_t t, std::index_sequence<I...>)
{
    return f(t[I]...);
}
template <typename func_t, typename array_t>
__device__ inline constexpr decltype(auto) invoke_with_array(func_t f, array_t a) {
  constexpr auto arity = function_traits<func_t>::arity;
  return invoke_with_array_impl(f, a, std::make_index_sequence<arity>{});
}

namespace arg_type {

// We need a way to compute the argument type of a function. But
// for nullary function, it does not really have an argument type
// in this case, we still need to return a valid type, but we don't
// really care what type this is.

struct dont_care {};

template <typename func_t, std::size_t arity>
struct arg_type_helper {
  using type = typename function_traits<func_t>::template arg<0>::type;
};

template <typename func_t>
struct arg_type_helper<func_t, 0> {
  using type = dont_care;
};

template <typename func_t>
using type = typename arg_type_helper<func_t, function_traits<func_t>::arity>::type;

}  // namespace arg_type

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

}  // namespace detail

template<typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void elementwise_kernel(int N, func_t f, array_t data) {
  // Assumption:
  // 1. all arguments of `f` have the same type, which could be different from the return type of `f`
  // 2. all tensors are contiguous, that is: stride == sizeof(type) for all tensors

  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using arg_t = detail::arg_type::type<func_t>;
  constexpr int arity = traits::arity;

  // We need to create array to hold all the arguments, for nullary `f`, this means array of size 0.
  // Unfortunately the compiler don't allow us to create array of 0 size, so for this case, we create
  // an array of size 1 and just don't use it.
  constexpr int nargs = traits::arity == 0 ? 1 : traits::arity;

  int tid = threadIdx.x;
  int idx = block_work_size() * blockIdx.x + tid;

  // compute base pointers
  return_t *result_base = reinterpret_cast<return_t *>(data[0]) + idx;
  arg_t *args_base[nargs];
  #pragma unroll
  for (int i = 0; i < arity; i++) {
    args_base[i] = reinterpret_cast<arg_t *>(data[i + 1]) + idx;
  }

  // fetch data
  return_t results[thread_work_size()];
  arg_t args[thread_work_size()][nargs];
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (idx + num_threads() * i < N) {
      #pragma unroll
      for (int j = 0; j < arity; j++) {
        args[i][j] = c10::load(args_base[j] + i * num_threads());
      }
    }
  }

  // compute
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (idx + num_threads() * i < N) {
      results[i] = detail::invoke_with_array<func_t, arg_t[nargs]>(f, args[i]);
    }
  }

  // store data
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (idx + num_threads() * i < N) {
      *(result_base + i * num_threads()) = results[i];
    }
  }
}

// TODO (@zasdfgbnm): this function assume trivial 1d and no dynamic casting
template<typename func_t, typename array_t, std::enable_if_t<detail::has_same_arg_types<func_t>::value, int> = 0>
static void launch_kernel(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename func_t, typename array_t, std::enable_if_t<!detail::has_same_arg_types<func_t>::value, int> = 0>
static void launch_kernel(int64_t N, const func_t& f, array_t data) {}

} // namespace modern


template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == traits::arity + 1);
  bool non_c10_complex = uses_non_c10_complex<func_t>::check();

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  at::detail::Array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    dtypes[i] = iter.dtype(i);
  }

  int64_t numel = iter.numel();
  if (iter.is_trivial_1d()) {
    auto inner_strides = iter.get_inner_strides();
    at::detail::Array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      strides[i] = inner_strides[i];
    }

    // TODO: can non_c10_complex go through the other path?  Need to verify.
    if (needs_dynamic_casting<func_t>::check(iter) || non_c10_complex) {
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
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    // TODO: can non_c10_complex go through the other path?  Need to verify.
    if (needs_dynamic_casting<func_t>::check(iter) || non_c10_complex) {
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

}} // namespace at::native
