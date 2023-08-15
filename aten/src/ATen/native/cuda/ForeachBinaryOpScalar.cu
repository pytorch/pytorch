#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_clamp_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpScalarFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
  return tensor_lists[1];
}

template <typename T, template <class> class Op>
void foreach_binary_op_(TensorList tensors, const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1>(
      tensor_lists,
      BinaryOpScalarFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
  increment_version(tensors);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_half_bfloat16_(TensorList tensors, const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors,
    const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

#define FOREACH_BINARY_OP_SCALAR(FUNCTION, NAME, OP, DIVISION_OP)     \
  void foreach_tensor_##NAME##_scalar_kernel_cuda_(                   \
      TensorList tensors, const Scalar& scalar) {                     \
    check_foreach_api_restrictions(tensors);                          \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_( \
          tensors, scalar);                                           \
    }                                                                 \
                                                                      \
    FUNCTION##_<OP>(tensors, scalar);                                 \
  }                                                                   \
                                                                      \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_cuda(     \
      TensorList tensors, const Scalar& scalar) {                     \
    check_foreach_api_restrictions(tensors);                          \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(  \
          tensors, scalar);                                           \
    }                                                                 \
                                                                      \
    return FUNCTION<OP>(tensors, scalar);                             \
  }

FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*div_op*/ false);
// See [Why is foreach_pow's division_op=true?]
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_half_bfloat16,
    pow,
    power_functor,
    /*div_op*/ true);
std::vector<Tensor> foreach_scalar_pow_list_kernel_cuda(
    const Scalar& scalar,
    TensorList exponent) {
  check_foreach_api_restrictions(exponent);
  if (!can_use_fast_route(exponent)) {
    return at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent);
  }
  return all_types_complex_half_bfloat16<reverse_power_functor>(
      exponent, scalar);
}

// In the case of division, integer inputs will result in float.
// Currently multi tensor apply can only return result of the same type as
// input.
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /*div_op*/ true);

// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void foreach_tensor_sub_scalar_kernel_cuda_(
    TensorList tensors,
    const Scalar& scalar) {
  check_foreach_api_restrictions(tensors);
  at::native::sub_check(tensors[0], scalar);

  if (!can_use_fast_route(tensors, scalar)) {
    return at::native::foreach_tensor_sub_scalar_kernel_slow_(tensors, scalar);
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, std::minus>(tensors, scalar); });
}

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_cuda(
    TensorList tensors,
    const Scalar& scalar) {
  check_foreach_api_restrictions(tensors);
  at::native::sub_check(tensors[0], scalar);

  if (!can_use_fast_route(tensors, scalar)) {
    return at::native::foreach_tensor_sub_scalar_kernel_slow(tensors, scalar);
  }

  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, std::minus>(tensors, scalar);
      });
}

FOREACH_BINARY_OP_SCALAR(all_types_half_bfloat16, clamp_max, minimum, false);
FOREACH_BINARY_OP_SCALAR(all_types_half_bfloat16, clamp_min, maximum, false);

namespace {
template <typename T>
__forceinline__ C10_DEVICE T clamp(
    const T v,
    const T lower,
    const T upper,
    const at::native::detail::ClampLimits& clamp_kind) {
  // Propagate nan, which doesn't propagate automatically for ROCm
  if (at::_isnan(v)) {
    return v;
  } else if (clamp_kind == at::native::detail::ClampLimits::Min) {
    return ::max(v, lower);
  } else if (clamp_kind == at::native::detail::ClampLimits::Max) {
    return ::min(v, upper);
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
      opmath_t upper,
      const at::native::detail::ClampLimits& clamp_kind) {
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
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = clamp(
              static_cast<opmath_t>(r_args[0][ii]), lower, upper, clamp_kind);
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
          r_args[0][ii] = clamp(
              static_cast<opmath_t>(r_args[0][ii]), lower, upper, clamp_kind);
        }
        store_args(args[res_arg_index], r_args[0], i_start, chunk_size, n);
      }
    }
  }
};

at::native::detail::ClampLimits get_clamp_kind(
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  if (min.has_value() && max.has_value()) {
    return at::native::detail::ClampLimits::MinMax;
  } else if (min.has_value()) {
    return at::native::detail::ClampLimits::Min;
  } else {
    return at::native::detail::ClampLimits::Max;
  }
}

std::vector<at::Tensor> foreach_tensor_clamp_scalar_kernel_cuda(
    TensorList self,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  check_foreach_api_restrictions(self);
  if (!can_use_fast_route(
          ArrayRef<TensorList>{self}, ArrayRef<Scalar>{}, true)) {
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
                            : max.value().to<opmath_t>(),
            max.has_value() ? max.value().to<opmath_t>()
                            : min.value().to<opmath_t>(),
            get_clamp_kind(min, max));
      });
  return tensor_lists[1];
}

void foreach_tensor_clamp_scalar_kernel_cuda_(
    TensorList self,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  check_foreach_api_restrictions(self);
  if (!can_use_fast_route(
          ArrayRef<TensorList>{self}, ArrayRef<Scalar>{}, true)) {
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
                            : max.value().to<opmath_t>(),
            max.has_value() ? max.value().to<opmath_t>()
                            : min.value().to<opmath_t>(),
            get_clamp_kind(min, max));
      });
}
} // namespace at::native
