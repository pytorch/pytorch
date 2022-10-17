#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_lerp_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at { namespace native {

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t lerp_(const scalar_t start, const scalar_t end, const scalar_t weight) {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t start_val = start;
  opmath_t end_val = end;
  opmath_t weight_val = weight;
  // Conditional for better numeric. This has been discussed in
  // https://github.com/pytorch/pytorch/pull/18871
  return (std::abs(weight_val) < 0.5)
      ? start_val + weight * (end_val - start_val)
      : end_val - (end_val - start_val) * (opmath_t{1} - weight_val);
}

// Ref: [GPU Pro Tip: Lerp Faster in C++](https://developer.nvidia.com/blog/lerp-faster-cuda/)
template<>
inline C10_HOST_DEVICE float lerp_(const float start, const float end, const float weight) {
  return fma(weight, end, fma(-weight, start, start));
}

template<>
inline C10_HOST_DEVICE double lerp_(const double start, const double end, const double weight) {
  return fma(weight, end, fma(-weight, start, start));
}

namespace {
template <typename T>
struct LerpFunctor {
  inline C10_DEVICE T operator()(const T self, const T end, const T weight) {
    return lerp_(self, end, weight);
  }
};

template<typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpListFunctor {
  using opmath_t = at::opmath_type<T>;
  template<typename Op> __device__ __forceinline__ void operator() (
      int chunk_size,
      TensorListMetadata<depth>& tl,
      Op op) {
    static_assert(depth == 3 || depth == 4, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned = init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
        load_store(r_args[2], args[2], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = op(
              static_cast<opmath_t>(r_args[0][ii]),
              static_cast<opmath_t>(r_args[1][ii]),
              static_cast<opmath_t>(r_args[2][ii])
          );
        }
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = op(
              static_cast<opmath_t>(r_args[0][ii]),
              static_cast<opmath_t>(r_args[1][ii]),
              static_cast<opmath_t>(r_args[2][ii])
          );
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

template<typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template<typename Op> __device__ __forceinline__ void operator() (
      int chunk_size,
      TensorListMetadata<depth>& tl,
      Op op,
      opmath_t alpha) {
    static_assert(depth == 2 || depth == 3, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    bool all_aligned = init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for(int i_start = threadIdx.x; i_start * kILP < n && i_start * kILP < chunk_size; i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for(int ii = 0; ii < kILP; ii++) {
            r_args[0][ii] = op(
                static_cast<opmath_t>(r_args[0][ii]),
                static_cast<opmath_t>(r_args[1][ii]),
                alpha
            );
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start , 0);
      }
    }
    else {
      for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for(int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = op(
              static_cast<opmath_t>(r_args[0][ii]),
              static_cast<opmath_t>(r_args[1][ii]),
              alpha
          );
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};
} // namespace

std::vector<at::Tensor> foreach_tensor_lerp_ternary_cuda(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3})) {
    return foreach_tensor_lerp_ternary_slow(tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists {tensors1.vec(), tensors2.vec(), tensors3.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, tensors1[0].scalar_type(), "foreach_tensor_lerp_ternary_cuda",
      [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          multi_tensor_apply<4>(
              tensor_lists,
              TernaryOpListFunctor<scalar_t, /* depth */ 4, /* r_args_depth */ 3, /* res_arg_index */ 3>(),
              LerpFunctor<opmath_t>());
      }
  );

  return tensor_lists[3];
}

void foreach_tensor_lerp_ternary_cuda_(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3})) {
    return foreach_tensor_lerp_ternary_slow_(tensors1, tensors2, tensors3);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists {tensors1.vec(), tensors2.vec(), tensors3.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, tensors1[0].scalar_type(), "foreach_tensor_lerp_ternary_cuda_",
        [&]() {
            using opmath_t = typename at::opmath_type<scalar_t>;
            multi_tensor_apply<3>(
                tensor_lists,
                TernaryOpListFunctor<scalar_t, /* depth */ 3, /* r_args_depth */ 3, /* res_arg_index */ 0>(),
                LerpFunctor<opmath_t>());
        }
  );
}

std::vector<at::Tensor> foreach_tensor_lerp_list_cuda(TensorList tensors1, TensorList tensors2, const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2})) {
    return foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists {tensors1.vec(), tensors2.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, tensors1[0].scalar_type(), "foreach_tensor_lerp_scalar_cuda",
      [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          multi_tensor_apply<3>(
              tensor_lists,
              TernaryOpScalarFunctor<scalar_t, /* depth */ 3, /* r_args_depth */ 2, /* res_arg_index */ 2>(),
              LerpFunctor<opmath_t>(),
              weight.to<opmath_t>());
      }
  );

  return tensor_lists[2];
}

void foreach_tensor_lerp_list_cuda_(TensorList tensors1, TensorList tensors2, const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2})) {
    return foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists {tensors1.vec(), tensors2.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, tensors1[0].scalar_type(), "foreach_tensor_lerp_scalar_cuda_",
        [&]() {
            using opmath_t = typename at::opmath_type<scalar_t>;
            multi_tensor_apply<2>(
                tensor_lists,
                TernaryOpScalarFunctor<scalar_t, /* depth */ 2, /* r_args_depth */ 2, /* res_arg_index */ 0>(),
                LerpFunctor<opmath_t>(),
                weight.to<opmath_t>());
        }
  );
}
} } // namespace at::native
