#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>
#include <functional>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_copy_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_tensor_list_op(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<3>(
      tensor_lists,
      BinaryOpListAlphaFunctor<
          T,
          /* depth */ 3,
          /* r_args_depth */ 2,
          /* res_arg_index */ 2>(),
      Op<opmath_t>(),
      alpha.to<opmath_t>());

  return tensor_lists[2];
}

template <typename T, template <class> class Op>
void foreach_tensor_list_op_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpListAlphaFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 2,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      alpha.to<opmath_t>());
  increment_version(tensors1);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_cuda",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

#define FOREACH_BINARY_OP_LIST(FUNCTION, NAME, OP, DIVISION_OP)     \
  void foreach_tensor_##NAME##_list_kernel_cuda_(                   \
      TensorList tensors1, TensorList tensors2) {                   \
    check_foreach_api_restrictions(tensors1, tensors2);             \
    if (!can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {     \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_( \
          tensors1, tensors2);                                      \
    }                                                               \
                                                                    \
    FUNCTION##_<OP>(tensors1, tensors2);                            \
  }                                                                 \
                                                                    \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_cuda(     \
      TensorList tensors1, TensorList tensors2) {                   \
    check_foreach_api_restrictions(tensors1, tensors2);             \
    if (!can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {     \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(  \
          tensors1, tensors2);                                      \
    }                                                               \
                                                                    \
    return FUNCTION<OP>(tensors1, tensors2);                        \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(FUNCTION, NAME, OP)               \
  void foreach_tensor_##NAME##_list_kernel_cuda_(                      \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
    if (!can_use_fast_route({tensors1, tensors2}, alpha)) {            \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(    \
          tensors1, tensors2, alpha);                                  \
    }                                                                  \
                                                                       \
    FUNCTION##_<OP>(tensors1, tensors2, alpha);                        \
  }                                                                    \
                                                                       \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_cuda(        \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
    if (!can_use_fast_route({tensors1, tensors2}, alpha)) {            \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(     \
          tensors1, tensors2, alpha);                                  \
    }                                                                  \
                                                                       \
    return FUNCTION<OP>(tensors1, tensors2, alpha);                    \
  }

FOREACH_BINARY_OP_LIST_ALPHA(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus);
FOREACH_BINARY_OP_LIST_ALPHA(
    all_types_complex_bool_half_bfloat16,
    sub,
    std::minus);
FOREACH_BINARY_OP_LIST(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*division_op*/ false);
FOREACH_BINARY_OP_LIST(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /*division_op*/ true);
// NOTE(crcrpar): `all_types_half_bfloat16` does not cover bool, so temporarily
// set `division_op` to true.
FOREACH_BINARY_OP_LIST(
    all_types_half_bfloat16,
    clamp_max,
    minimum,
    /*division_op*/ true);
FOREACH_BINARY_OP_LIST(
    all_types_half_bfloat16,
    clamp_min,
    maximum,
    /*division_op*/ true);
// NOTE(crcrpar): [Why is foreach_pow's division_op=true?]
// To push integer inputs to slow path. This is because with integer type inputs
// the fast path behaves differently from the slow one. Need to investigate
// later.
FOREACH_BINARY_OP_LIST(
    all_types_complex_half_bfloat16,
    pow,
    power_functor,
    /*division_op*/ true);

template <typename dst_t, typename src_t = dst_t>
struct Copy {
  __device__ __forceinline__ dst_t operator()(const src_t& x) {
    return static_cast<dst_t>(x);
  }
};

template <typename dst_t>
struct Copy<dst_t, c10::complex<double>> {
  __device__ __forceinline__ dst_t operator()(const c10::complex<double>& x) {
    if constexpr (!(std::is_same_v<dst_t, c10::complex<double>> ||
                    std::is_same_v<dst_t, c10::complex<float>>)) {
      return static_cast<dst_t>(x.real());
    } else {
      return static_cast<dst_t>(x);
    }
  }
};

template <typename dst_t>
struct Copy<dst_t, c10::complex<float>> {
  __device__ __forceinline__ dst_t operator()(const c10::complex<float>& x) {
    if constexpr (!(std::is_same_v<dst_t, c10::complex<double>> ||
                    std::is_same_v<dst_t, c10::complex<float>>)) {
      return static_cast<dst_t>(x.real());
    } else {
      return static_cast<dst_t>(x);
    }
  }
};

#define AT_DISPATCH_SOURCE_TYPES(TYPE, NAME, ...)                                                \
  AT_DISPATCH_SWITCH(                                                                            \
      TYPE,                                                                                      \
      NAME,                                                                                      \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                                                           \
          at::ScalarType::Byte,                                                                  \
          src_t,                                                                                 \
          __VA_ARGS__) AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Char, src_t, __VA_ARGS__) \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                                                       \
              at::ScalarType::Long, src_t, __VA_ARGS__)                                          \
              AT_PRIVATE_CASE_TYPE_USING_HINT(                                                   \
                  at::ScalarType::Short, src_t, __VA_ARGS__)                                     \
                  AT_PRIVATE_CASE_TYPE_USING_HINT(                                               \
                      at::ScalarType::Int, src_t, __VA_ARGS__)                                   \
                      AT_PRIVATE_CASE_TYPE_USING_HINT(                                           \
                          at::ScalarType::Double, src_t, __VA_ARGS__)                            \
                          AT_PRIVATE_CASE_TYPE_USING_HINT(                                       \
                              at::ScalarType::Float, src_t, __VA_ARGS__)                         \
                              AT_PRIVATE_CASE_TYPE_USING_HINT(                                   \
                                  at::ScalarType::ComplexDouble,                                 \
                                  src_t,                                                         \
                                  __VA_ARGS__)                                                   \
                                  AT_PRIVATE_CASE_TYPE_USING_HINT(                               \
                                      at::ScalarType::ComplexFloat,                              \
                                      src_t,                                                     \
                                      __VA_ARGS__)                                               \
                                      AT_PRIVATE_CASE_TYPE_USING_HINT(                           \
                                          at::ScalarType::Half,                                  \
                                          src_t,                                                 \
                                          __VA_ARGS__)                                           \
                                          AT_PRIVATE_CASE_TYPE_USING_HINT(                       \
                                              at::ScalarType::BFloat16,                          \
                                              src_t,                                             \
                                              __VA_ARGS__)                                       \
                                              AT_PRIVATE_CASE_TYPE_USING_HINT(                   \
                                                  at::ScalarType::Bool,                          \
                                                  src_t,                                         \
                                                  __VA_ARGS__)                                   \
                                                  AT_PRIVATE_CASE_TYPE_USING_HINT(               \
                                                      at::ScalarType::                           \
                                                          Float8_e4m3fn,                         \
                                                      src_t,                                     \
                                                      __VA_ARGS__)                               \
                                                      AT_PRIVATE_CASE_TYPE_USING_HINT(           \
                                                          at::ScalarType::                       \
                                                              Float8_e4m3fnuz,                   \
                                                          src_t,                                 \
                                                          __VA_ARGS__)                           \
                                                          AT_PRIVATE_CASE_TYPE_USING_HINT(       \
                                                              at::ScalarType::                   \
                                                                  Float8_e5m2,                   \
                                                              src_t,                             \
                                                              __VA_ARGS__)                       \
                                                              AT_PRIVATE_CASE_TYPE_USING_HINT(   \
                                                                  at::ScalarType::               \
                                                                      Float8_e5m2fnuz,           \
                                                                  src_t,                         \
                                                                  __VA_ARGS__))

namespace {

template <
    typename T,
    typename src_t,
    int depth,
    int r_args_depth,
    int res_arg_index>
struct CopyFunctor {
  static_assert(depth == 2 && r_args_depth == 1 && res_arg_index == 1);
  template <typename Op>
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<depth>& tl,
      Op op) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    src_t* src_ptr = (src_t*)tl.addresses[0][tensor_loc];
    src_ptr += chunk_idx * chunk_size;
    T* self_ptr = (T*)tl.addresses[1][tensor_loc];
    self_ptr += chunk_idx * chunk_size;

    const bool all_aligned{is_aligned(src_ptr) && is_aligned(self_ptr)};

    n -= chunk_idx * chunk_size;
    src_t src_args[kILP];
    T r_args[kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(src_args, src_ptr, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[ii] = static_cast<T>(op(src_args[ii]));
        }
        // store
        load_store(self_ptr, r_args, i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          const auto i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            src_args[ii] = src_ptr[i];
          }
        }
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[ii] = static_cast<T>(op(src_args[ii]));
        }
        store_args(self_ptr, r_args, i_start, chunk_size, n);
      }
    }
  }
};

} // anonymous namespace

void foreach_tensor_copy_list_kernel_cuda_(
    TensorList self,
    TensorList src,
    const bool non_blocking) {
  check_foreach_api_restrictions(self, src);
  if (!(_check_tensors_share_device_and_dtype(
            {self, src}, /* skip_dtype_check */ true) &&
        std::all_of(
            src.cbegin(),
            src.cend(),
            [&](const auto& t) -> bool {
              return t.dtype() == src[0].dtype();
            }) &&
        _check_tensors_share_sizes_and_strides({self, src}))) {
    return at::native::foreach_tensor_copy_list_kernel_slow_(
        self, src, non_blocking);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{src.vec(), self.vec()};

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND7(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Bool,
      ScalarType::Float8_e4m3fn,
      ScalarType::Float8_e4m3fnuz,
      ScalarType::Float8_e5m2,
      ScalarType::Float8_e5m2fnuz,
      self[0].scalar_type(),
      "foreach_tensor_copy",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        AT_DISPATCH_SOURCE_TYPES(src[0].scalar_type(), "foreach_tensor_copy", [&] {
          if constexpr (std::is_same_v<scalar_t, src_t>) {
            multi_tensor_apply<2>(
                tensor_lists,
                UnaryOpFunctor<
                    scalar_t,
                    /* depth */ 2,
                    /* r_args_depth */ 1,
                    /* res_arg_index */ 1>(),
                Copy<opmath_t, opmath_t>());
          } else {
            // Ref:
            // https://github.com/pytorch/pytorch/blob/656134c38f4737d13c3f43fc5c59470bc23c1d2f/aten/src/ATen/native/Copy.cpp#L299-L301
            if (!self[0].is_complex() && src[0].is_complex()) {
              TORCH_WARN_ONCE(
                  "Casting complex values to real discards the imaginary part");
            }
            multi_tensor_apply<2>(
                tensor_lists,
                CopyFunctor<
                    scalar_t,
                    src_t,
                    /* depth */ 2,
                    /* r_args_depth */ 1,
                    /* res_arg_index */ 1>(),
                Copy<scalar_t, src_t>());
          }
        });
      });
  increment_version(self);
}

#undef AT_DISPATCH_SOURCE_TYPES

} // namespace at::native
