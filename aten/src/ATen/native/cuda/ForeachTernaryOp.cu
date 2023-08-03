#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_clamp_native.h>
#include <ATen/ops/_foreach_lerp_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T>
struct LerpFunctor {
  inline C10_DEVICE T operator()(const T self, const T end, const T weight) {
    return lerp(self, end, weight);
  }
};

std::vector<at::Tensor> foreach_tensor_lerp_ternary_cuda(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3})) {
    return foreach_tensor_ternary_lerp_slow(tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            LerpFunctor<opmath_t>());
      });

  return tensor_lists[3];
}

void foreach_tensor_lerp_ternary_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3})) {
    return foreach_tensor_ternary_lerp_slow_(tensors1, tensors2, tensors3);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_cuda_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });
  increment_version(tensors1);
}

std::vector<at::Tensor> foreach_tensor_lerp_list_cuda(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2})) {
    return foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), vec_res};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });

  return tensor_lists[2];
}

void foreach_tensor_lerp_list_cuda_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2})) {
    return foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_cuda_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });
}

namespace {
template <typename T>
__forceinline__ C10_DEVICE T clamp(const T v, const T lower, const T upper) {
  // Propagate nan, which doesn't propagate automatically for ROCm
  if (at::_isnan(v)) {
    return v;
  }
  if (at::_isnan(lower)) {
    return lower;
  }
  if (at::_isnan(upper)) {
    return upper;
  } else {
    return ::min(::max(v, lower), upper);
  }
}
} // namespace

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct ClampFunctor {
  using opmath_t = at::opmath_type<T>;
  __forceinline__ C10_DEVICE T operator()(
      const int chunk_size,
      TensorListMetadata<depth>& tl,
      opmath_t lower,
      opmath_t upper) {
    static_assert(depth == 1 || depth == 2, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              clamp(static_cast<opmath_t>(r_args[0][ii]), lower, upper);
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<r_args_depth>(r_args, args, i_start, chunk_size, n);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              clamp(static_cast<opmath_t>(r_args[0][ii]), lower, upper);
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

std::vector<at::Tensor> foreach_tensor_clamp_scalar_kernel_cuda(
    TensorList self,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  check_foreach_api_restrictions(self);
  if (!can_use_fast_route({self})) {
    return foreach_tensor_clamp_scalar_kernel_slow(self, min, max);
  }
  TORCH_CHECK(
      min.has_value() || max.has_value(),
      "Either `min` or `max` must be specified");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (const auto& t : self) {
    result.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{self.vec(), result};
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "foreach_tensor_clamp_scalar_kernel_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            ClampFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 1,
                /* res_arg_index */ 1>(),
            min.has_value() ? min.value().to<opmath_t>()
                            : std::numeric_limits<opmath_t>::min(),
            max.has_value() ? max.value().to<opmath_t>()
                            : std::numeric_limits<opmath_t>::max());
      });
  return tensor_lists[1];
}

void foreach_tensor_clamp_scalar_kernel_cuda_(
    TensorList self,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  check_foreach_api_restrictions(self);
  if (!can_use_fast_route({self})) {
    return foreach_tensor_clamp_scalar_kernel_slow_(self, min, max);
  }
  TORCH_CHECK(
      min.has_value() || max.has_value(),
      "Either `min` or `max` must be specified");
  std::vector<std::vector<at::Tensor>> tensor_lists{self.vec()};
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "foreach_tensor_clamp_scalar_kernel_cuda",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<1>(
            tensor_lists,
            ClampFunctor<
                scalar_t,
                /* depth */ 1,
                /* r_args_depth */ 1,
                /* res_arg_index */ 0>(),
            min.has_value() ? min.value().to<opmath_t>()
                            : std::numeric_limits<opmath_t>::min(),
            max.has_value() ? max.value().to<opmath_t>()
                            : std::numeric_limits<opmath_t>::max());
      });
}

} // namespace at::native
