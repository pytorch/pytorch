#pragma once
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

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else

#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// _foreach_norm supports only L1, L2, and inf norm
enum class NormType { L1, L2, LInf };

// NOTE: This is a simple variant of TensorListMetadata in MultiTensorApply.cuh
// as we only need to track addresses for the lpnorm_cleanup function below.
// Why is this struct necessary? For the same reason the TensorListMetadata
// struct is necessary--which is to ferry static metadata to the CUDA kernel
// while complying with the 4kb size constraint. Since we only need to track
// addresses, we introduce this struct to be able to fit more Tensor pointers at
// a time, currently 400 empirically, compared to the much smaller values in
// depth_to_max_tensors. This way, we can launch fewer kernels for better
// performance.
//
// IF YOU USE THIS STRUCT, PLEASE ADD A ONE-OFF TEST IN test_foreach.py AS THIS
// IS CURRENTLY ONLY TESTED FOR _foreach_norm.
const size_t MAX_TENSORS_PER_KERNEL = 400;
struct TensorListAddresses {
  const void* addresses[MAX_TENSORS_PER_KERNEL];
};

namespace {
template <
    typename T,
    NormType norm_type,
    typename out_t,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpNormFunctor {
  using out_opmath_t = typename at::opmath_type<out_t>;
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<depth>& tl,
      out_opmath_t* global_output_ptr,
      out_opmath_t* output_per_tensor_ptr,
      const int max_chunks_per_tensor,
      const bool calculate_global_norm,
      const bool calculate_norm_per_tensor) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* x = (T*)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    __shared__ out_opmath_t s_vals[512];
    out_opmath_t vals[kILP];
    T r_x[kILP];
    for (int64_t i = 0; i < kILP; i++) {
      vals[i] = out_opmath_t(0);
      r_x[i] = T(0);
    }

    if (n % kILP == 0 && (chunk_size & kILP) == 0 && is_aligned(x)) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          const auto next = static_cast<out_opmath_t>(r_x[ii]);
          if constexpr (norm_type == NormType::LInf) {
            vals[ii] = max_propagate_nan(vals[ii], ::abs(next));
          } else {
            vals[ii] += norm_type == NormType::L1 ? ::abs(next) : next * next;
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            const auto next = static_cast<out_opmath_t>(x[i]);
            if constexpr (norm_type == NormType::LInf) {
              vals[ii] = max_propagate_nan(vals[ii], ::abs(next));
            } else {
              vals[ii] += norm_type == NormType::L1 ? ::abs(next) : next * next;
            }
          }
        }
      }
    }

    auto val = out_opmath_t(0);
    for (int i = 0; i < kILP; i++) {
      if constexpr (norm_type == NormType::LInf) {
        val = max_propagate_nan(val, vals[i]);
      } else {
        val += vals[i];
      }
    }
    auto final_val = norm_type == NormType::L1 || norm_type == NormType::L2
        ? at::native::cuda_utils::BlockReduceSum(val, s_vals)
        : at::native::cuda_utils::BlockReduceMax(val, s_vals);

    if (threadIdx.x == 0) {
      if (calculate_global_norm) {
        if (norm_type == NormType::LInf) {
          global_output_ptr[blockIdx.x] = max_propagate_nan(final_val, global_output_ptr[blockIdx.x]);
        } else {
          global_output_ptr[blockIdx.x] += final_val;
        }
      }
      if (calculate_norm_per_tensor) {
        output_per_tensor_ptr
            [(tl.start_tensor_this_launch + tensor_loc) *
                 max_chunks_per_tensor +
             chunk_idx] = final_val;
      }
    }
  }
};

template <
    typename T,
    NormType norm_type,
    typename out_t,
    typename out_opmath_t = at::opmath_type<out_t>>
__global__ void lpnorm_cleanup(
    out_t* global_norm,
    const out_opmath_t* global_output,
    const out_opmath_t* output_per_tensor,
    TensorListAddresses addr_struct,
    int max_chunks_per_tensor,
    const bool calculate_global_norm,
    const bool calculate_norm_per_tensor) {
  __shared__ out_opmath_t vals[512];

  if (calculate_global_norm) {
    if (blockIdx.x == 0) {
      out_opmath_t val{0};
      if (threadIdx.x < 320) {
        val = global_output[threadIdx.x];
      }
      out_opmath_t final_val =
          norm_type == NormType::L1 || norm_type == NormType::L2
          ? at::native::cuda_utils::BlockReduceSum<out_opmath_t>(val, vals)
          : at::native::cuda_utils::BlockReduceMax(val, vals);

      if (threadIdx.x == 0) {
        *global_norm = static_cast<out_t>(
            norm_type == NormType::L1 || norm_type == NormType::LInf
                ? final_val
                : ::sqrt(final_val));
      }
    }
  }

  if (calculate_norm_per_tensor) {
    const out_opmath_t* output_this_tensor =
        output_per_tensor + blockIdx.x * max_chunks_per_tensor;
    out_opmath_t val = 0;
    for (size_t i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) {
      if constexpr (norm_type == NormType::LInf) {
        val = max_propagate_nan(val, output_this_tensor[i]);
      } else {
        val += output_this_tensor[i];
      }
    }
    out_opmath_t final_val =
        norm_type == NormType::L1 || norm_type == NormType::L2
        ? at::native::cuda_utils::BlockReduceSum<out_opmath_t>(val, vals)
        : at::native::cuda_utils::BlockReduceMax(val, vals);
    if (threadIdx.x == 0) {
      *(out_t*)addr_struct.addresses[blockIdx.x] =
          norm_type == NormType::L1 || norm_type == NormType::LInf
          ? final_val
          : ::sqrt(final_val);
    }
  }
}

inline void check_foreach_norm_dtype(
    std::optional<ScalarType> opt_dtype,
    ScalarType self_dtype,
    const char* const name) {
  if (opt_dtype.has_value()) {
    auto dtype = opt_dtype.value();
    TORCH_CHECK(
        isFloatingType(dtype) || isComplexType(dtype),
        name,
        ": dtype should"
        " be floating point or complex, but got ",
        dtype);
    TORCH_CHECK(
        isComplexType(self_dtype) == isComplexType(dtype),
        name,
        ": dtype should be ",
        isComplexType(self_dtype) ? "complex" : "real",
        " for ",
        isComplexType(self_dtype) ? "complex" : "real",
        " inputs, but got ",
        dtype);
    TORCH_CHECK(
        promoteTypes(self_dtype, dtype) == dtype,
        name,
        ": the dtype of the input ",
        "(",
        self_dtype,
        ") should be convertible ",
        "without narrowing to the specified dtype (",
        dtype,
        ")");
  }
}
} // namespace

// note(mkozuki): Why excluding Int and Complex from fast path
// - Int: at::norm does not support.
// - Complex: __shfl_down_sync does not support complex and foreach does not
// support functions whose inputs dtypes and output dtype are different.
std::pair<std::vector<Tensor>, Tensor> _foreach_tensor_norm_cuda_impl(
    TensorList tensors,
    const Scalar& ord,
    std::optional<ScalarType> dtype,
    const bool calculate_global_norm,
    const bool calculate_norm_per_tensor);

} // namespace at::native
