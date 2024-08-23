#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>
#include <limits>
#include <utility>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/foreach_reduce.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_global_norm_native.h>
#include <ATen/ops/_foreach_max_native.h>
#include <ATen/ops/_foreach_norm_native.h>

#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

template <
    typename T,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpMaxFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<depth>& tl,
      T* output_per_tensor_ptr,
      const int max_chunks_per_tensor) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* x = (T*)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    __shared__ T s_vals[512];
    T vals[kILP];
    T r_x[kILP];
    for (int64_t i = 0; i < kILP; i++) {
      vals[i] = T(std::numeric_limits<T>::lowest());
      r_x[i] = T(std::numeric_limits<T>::lowest());
    }

    if (n % kILP == 0 && (chunk_size & kILP) == 0 && is_aligned(x)) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          vals[ii] = max_propagate_nan(vals[ii], r_x[ii]);
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            vals[ii] = max_propagate_nan(vals[ii], x[i]);
          }
        }
      }
    }

    auto val = T(std::numeric_limits<T>::lowest());
    for (int i = 0; i < kILP; i++) {
      val = max_propagate_nan(val, vals[i]);
    }
    auto final_val = at::native::cuda_utils::BlockReduceMax(val, s_vals);

    if (threadIdx.x == 0) {
      output_per_tensor_ptr
          [(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
           chunk_idx] = final_val;
    }
  }
};

template <typename T>
__global__ void lpmax_cleanup(
    const T* output_per_tensor,
    TensorListAddresses addr_struct,
    int max_chunks_per_tensor) {
  __shared__ T vals[512];
  const T* output_this_tensor =
      output_per_tensor + blockIdx.x * max_chunks_per_tensor;
  T val = std::numeric_limits<T>::lowest();
  for (size_t i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) {
    val = max_propagate_nan(val, output_this_tensor[i]);
  }
  T final_val = at::native::cuda_utils::BlockReduceMax(val, vals);
  if (threadIdx.x == 0) {
    *(T*)addr_struct.addresses[blockIdx.x] = final_val;
  }
}

std::vector<Tensor> foreach_tensor_max_cuda(TensorList tensors) {
  check_foreach_api_restrictions(tensors);
  if (!can_use_fast_route(tensors)) {
    return foreach_tensor_max_slow(tensors);
  }

  // for parity with max in ReduceAllOps.cpp, as max(empty) is ???
  TORCH_CHECK(
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [](const auto& t) { return t.numel() > 0; }),
      "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");

  const size_t ntensors = tensors.size();
  int max_chunks_per_tensor = -1;

  for (const auto t : c10::irange(ntensors)) {
    int max_chunks_this_tensor =
        (tensors[t].numel() + kChunkSize - 1) / kChunkSize;
    if (max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }
  const auto options = tensors[0].options();
  auto output_per_tensor = at::zeros(
      {static_cast<int64_t>(ntensors) * max_chunks_per_tensor}, options);

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(ntensors);
  for (const auto i : c10::irange(ntensors)) {
    vec_res.push_back(at::native::empty_cuda(
        {},
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt(),
        options.memory_format_opt()));
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};

  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf,
      kBFloat16,
      kBool,
      tensor_lists[0][0].scalar_type(),
      "foreach_tensor_max_cuda_scalar_type",
      [&]() {
        multi_tensor_apply<1>(
            tensor_lists,
            LpMaxFunctor<scalar_t>(),
            output_per_tensor.mutable_data_ptr<scalar_t>(),
            max_chunks_per_tensor);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const at::cuda::OptionalCUDAGuard device_guard(
            device_of(output_per_tensor));
        auto stream = at::cuda::getCurrentCUDAStream();

        const size_t num_kernels = ceil_div(ntensors, MAX_TENSORS_PER_KERNEL);
        for (const auto i : c10::irange(num_kernels)) {
          const size_t num_tensors_this_kernel =
              (i < num_kernels - 1 || ntensors % MAX_TENSORS_PER_KERNEL == 0)
              ? MAX_TENSORS_PER_KERNEL
              : (ntensors % MAX_TENSORS_PER_KERNEL);

          TensorListAddresses addr_struct;
          for (const auto j : c10::irange(num_tensors_this_kernel)) {
            addr_struct.addresses[j] = vec_res[i * MAX_TENSORS_PER_KERNEL + j]
                                           .mutable_data_ptr<scalar_t>();
          }

          lpmax_cleanup<scalar_t><<<num_tensors_this_kernel, 512, 0, stream>>>(
              output_per_tensor.const_data_ptr<scalar_t>() +
                  i * MAX_TENSORS_PER_KERNEL * max_chunks_per_tensor,
              addr_struct,
              max_chunks_per_tensor);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // correctly assign values to only non-empty slots, as the empty slots should
  // get skipped
  std::vector<Tensor> result;
  result.reserve(ntensors);
  int i = 0;
  for (const auto& t : tensors) {
    if (t.numel() != 0) {
      result.emplace_back(vec_res[i]);
      i++;
    } else {
      result.emplace_back(at::native::empty_cuda(
          {},
          optTypeMetaToScalarType(options.dtype_opt()),
          options.layout_opt(),
          options.device_opt(),
          options.pinned_memory_opt(),
          options.memory_format_opt()));
    }
  }
  return result;
}

#define AT_DISPATCH_OUT_DTYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                      \
          at::ScalarType::Double, out_t, __VA_ARGS__)       \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                  \
              at::ScalarType::Float, out_t, __VA_ARGS__)    \
              AT_PRIVATE_CASE_TYPE_USING_HINT(              \
                  at::ScalarType::Half, out_t, __VA_ARGS__) \
                  AT_PRIVATE_CASE_TYPE_USING_HINT(          \
                      at::ScalarType::BFloat16, out_t, __VA_ARGS__))

std::pair<std::vector<Tensor>, Tensor> _foreach_tensor_norm_cuda_impl(
    TensorList tensors,
    const double& p,
    std::optional<ScalarType> dtype,
    const bool calculate_global_norm,
    const bool calculate_norm_per_tensor) {
  TORCH_INTERNAL_ASSERT(
      calculate_global_norm || calculate_norm_per_tensor,
      "either calculate_global_norm or calculate_norm_per_tensor should be true but calculate_global_norm: ",
      calculate_global_norm,
      ", and calculate_norm_per_tensor: ",
      calculate_norm_per_tensor);
  check_foreach_norm_dtype(
      dtype, tensors[0].scalar_type(), "_foreach_tensor_norm_cuda");

  const size_t ntensors = tensors.size();
  int max_chunks_per_tensor = -1;

  for (const auto t : c10::irange(ntensors)) {
    int max_chunks_this_tensor =
        (tensors[t].numel() + kChunkSize - 1) / kChunkSize;
    if (max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }
  const auto options = tensors[0].options();
  const ScalarType output_dtype =
      dtype.has_value() ? dtype.value() : tensors[0].scalar_type();
  const ScalarType opmath_output_dtype = toOpMathType(output_dtype);
  const auto opmath_t_options = options.dtype(opmath_output_dtype);

  Tensor output_per_tensor;
  if (calculate_norm_per_tensor) {
    output_per_tensor = at::zeros(
        {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
        opmath_t_options);
  }
  Tensor global_output;
  if (calculate_global_norm) {
    global_output = at::zeros({320}, opmath_t_options);
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};

  const auto res_option = options.dtype(output_dtype);
  std::vector<at::Tensor> vec_res;
  if (calculate_norm_per_tensor) {
    vec_res.reserve(ntensors);
    for (const auto i : c10::irange(ntensors)) {
      vec_res.push_back(at::native::empty_cuda(
          {},
          optTypeMetaToScalarType(res_option.dtype_opt()),
          res_option.layout_opt(),
          res_option.device_opt(),
          res_option.pinned_memory_opt(),
          res_option.memory_format_opt()));
    }
  }
  Tensor global_norm;
  if (calculate_global_norm) {
    global_norm = at::native::empty_cuda(
        {},
        optTypeMetaToScalarType(res_option.dtype_opt()),
        res_option.layout_opt(),
        res_option.device_opt(),
        res_option.pinned_memory_opt(),
        res_option.memory_format_opt());
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      c10::kBFloat16,
      tensor_lists[0][0].scalar_type(),
      "foreach_tensor_norm_cuda_impl_scalar_type",
      [&]() {
        AT_DISPATCH_OUT_DTYPES(
            output_dtype, "foreach_tensor_norm_cuda_impl_out_dtype", [&]() {
              using out_opmath_t = typename at::opmath_type<out_t>;
              if (p == static_cast<double>(1)) {
                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::L1, out_t>(),
                    calculate_global_norm
                        ? global_output.mutable_data_ptr<out_opmath_t>()
                        : nullptr,
                    calculate_norm_per_tensor
                        ? output_per_tensor.mutable_data_ptr<out_opmath_t>()
                        : nullptr,
                    max_chunks_per_tensor,
                    calculate_global_norm,
                    calculate_norm_per_tensor);
              } else if (p == static_cast<double>(2)) {
                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::L2, out_t>(),
                    calculate_global_norm
                        ? global_output.mutable_data_ptr<out_opmath_t>()
                        : nullptr,
                    calculate_norm_per_tensor
                        ? output_per_tensor.mutable_data_ptr<out_opmath_t>()
                        : nullptr,
                    max_chunks_per_tensor,
                    calculate_global_norm,
                    calculate_norm_per_tensor);
              } else if (p == std::numeric_limits<double>::infinity()) {
                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::LInf, out_t>(),
                    calculate_global_norm
                        ? global_output.mutable_data_ptr<out_opmath_t>()
                        : nullptr,
                    calculate_norm_per_tensor
                        ? output_per_tensor.mutable_data_ptr<out_opmath_t>()
                        : nullptr,
                    max_chunks_per_tensor,
                    calculate_global_norm,
                    calculate_norm_per_tensor);
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              const at::cuda::OptionalCUDAGuard device_guard(
                  device_of(output_per_tensor));
              auto stream = at::cuda::getCurrentCUDAStream();

              const size_t num_kernels =
                  ceil_div(ntensors, MAX_TENSORS_PER_KERNEL);
              for (const auto i : c10::irange(num_kernels)) {
                const size_t num_tensors_this_kernel =
                    (i < num_kernels - 1 ||
                     ntensors % MAX_TENSORS_PER_KERNEL == 0)
                    ? MAX_TENSORS_PER_KERNEL
                    : (ntensors % MAX_TENSORS_PER_KERNEL);

                TensorListAddresses addr_struct;
                if (calculate_norm_per_tensor) {
                  for (const auto j : c10::irange(num_tensors_this_kernel)) {
                    addr_struct.addresses[j] =
                        vec_res[i * MAX_TENSORS_PER_KERNEL + j]
                            .mutable_data_ptr<out_t>();
                  }
                }

                if (p == static_cast<double>(1)) {
                  lpnorm_cleanup<scalar_t, NormType::L1, out_t>
                      <<<num_tensors_this_kernel, 512, 0, stream>>>(
                          calculate_global_norm
                              ? global_norm.mutable_data_ptr<out_t>()
                              : nullptr,
                          calculate_global_norm
                              ? global_output.const_data_ptr<out_opmath_t>()
                              : nullptr,
                          calculate_norm_per_tensor
                              ? output_per_tensor
                                      .const_data_ptr<out_opmath_t>() +
                                  i * MAX_TENSORS_PER_KERNEL *
                                      max_chunks_per_tensor
                              : nullptr,
                          addr_struct,
                          max_chunks_per_tensor,
                          calculate_global_norm,
                          calculate_norm_per_tensor);
                } else if (p == static_cast<double>(2)) {
                  lpnorm_cleanup<scalar_t, NormType::L2, out_t>
                      <<<num_tensors_this_kernel, 512, 0, stream>>>(
                          calculate_global_norm
                              ? global_norm.mutable_data_ptr<out_t>()
                              : nullptr,
                          calculate_global_norm
                              ? global_output.const_data_ptr<out_opmath_t>()
                              : nullptr,
                          calculate_norm_per_tensor
                              ? output_per_tensor
                                      .const_data_ptr<out_opmath_t>() +
                                  i * MAX_TENSORS_PER_KERNEL *
                                      max_chunks_per_tensor
                              : nullptr,
                          addr_struct,
                          max_chunks_per_tensor,
                          calculate_global_norm,
                          calculate_norm_per_tensor);
                } else if (p == std::numeric_limits<double>::infinity()) {
                  lpnorm_cleanup<scalar_t, NormType::LInf, out_t>
                      <<<num_tensors_this_kernel, 512, 0, stream>>>(
                          calculate_global_norm
                              ? global_norm.mutable_data_ptr<out_t>()
                              : nullptr,
                          calculate_global_norm
                              ? global_output.const_data_ptr<out_opmath_t>()
                              : nullptr,
                          calculate_norm_per_tensor
                              ? output_per_tensor
                                      .const_data_ptr<out_opmath_t>() +
                                  i * MAX_TENSORS_PER_KERNEL *
                                      max_chunks_per_tensor
                              : nullptr,
                          addr_struct,
                          max_chunks_per_tensor,
                          calculate_global_norm,
                          calculate_norm_per_tensor);
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            });
      });

  // correctly assign values to only non-empty slots, as the empty slots should
  // get skipped
  std::vector<Tensor> result;
  if (calculate_norm_per_tensor) {
    result.reserve(ntensors);
    int i = 0;
    for (const auto& t : tensors) {
      if (t.numel() != 0) {
        result.emplace_back(vec_res[i]);
        i++;
      } else {
        result.emplace_back(at::zeros({}, res_option));
      }
    }
  }
  return std::make_pair(result, global_norm);
}

#undef AT_DISPATCH_OUT_DTYPES

namespace {
double ord_to_double(const Scalar& ord) {
  if (ord.isIntegral(false)) {
    return ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    return ord.to<double>();
  } else {
    TORCH_CHECK(
        false,
        "foreach implementation of norm expects ord to be integer or float");
  }
}

bool use_slowpath(TensorList tensors, const double& p) {
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  return (
      !can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2) ||
        p == std::numeric_limits<double>::infinity()));
}
} // namespace

std::vector<Tensor> foreach_tensor_norm_cuda(
    TensorList tensors,
    const Scalar& ord,
    std::optional<ScalarType> dtype) {
  const auto p = ord_to_double(ord);
  check_foreach_api_restrictions(tensors);
  if (use_slowpath(tensors, p)) {
    return foreach_tensor_norm_slow(tensors, ord, dtype);
  }
  return _foreach_tensor_norm_cuda_impl(
             tensors,
             p,
             dtype,
             /* calculate_global_norm */ false,
             /* calculate_norm_per_tensor */ true)
      .first;
}

Tensor foreach_tensor_global_norm_cuda(
    TensorList tensors,
    const Scalar& ord,
    std::optional<ScalarType> dtype) {
  const auto p = ord_to_double(ord);
  check_foreach_api_restrictions(tensors);
  if (use_slowpath(tensors, p)) {
    return foreach_tensor_global_norm_slow(tensors, ord, dtype);
  }
  return _foreach_tensor_norm_cuda_impl(
             tensors,
             p,
             dtype,
             /* calculate_global_norm */ true,
             /* calculate_norm_per_tensor */ false)
      .second;
}

} // namespace at::native
