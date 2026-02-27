#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>
#include <limits>
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
#include <ATen/ops/_foreach_max_native.h>
#include <ATen/ops/_foreach_norm_native.h>
#include <ATen/ops/_foreach_powsum_native.h>

#include <ATen/ops/empty_native.h>
#include <ATen/ops/full.h>
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

template <
    typename T,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpMaxFunctor {
  __device__ __forceinline__ void operator()(
      int64_t chunk_size,
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

  // Initialize output_per_tensor with lowest value
  Tensor output_per_tensor;

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
        // Initialize intermediate buffer with lowest()
        output_per_tensor = at::full(
            {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
            std::numeric_limits<scalar_t>::lowest(),
            options);

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
      int64_t chunk_size,
      TensorListMetadata<depth>& tl,
      out_opmath_t* output_per_tensor_ptr,
      const int max_chunks_per_tensor) {
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
      output_per_tensor_ptr
          [(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
           chunk_idx] = final_val;
    }
  }
};

template <
    typename T,
    NormType norm_type,
    typename out_t,
    bool apply_root = true,
    typename out_opmath_t = at::opmath_type<out_t>>
__global__ void lpnorm_cleanup(
    const out_opmath_t* output_per_tensor,
    TensorListAddresses addr_struct,
    int max_chunks_per_tensor) {
  __shared__ out_opmath_t vals[512];

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
    out_opmath_t result = final_val;
    if constexpr (apply_root && norm_type == NormType::L2) {
      result = ::sqrt(final_val);
    }
    *(out_t*)addr_struct.addresses[blockIdx.x] = result;
  }
}

namespace {
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
} // anonymous namespace

template <typename T, NormType norm_type, typename out_t, int reduce_dim>
struct LpNormDimFunctor {
  using out_opmath_t = typename at::opmath_type<out_t>;

  __device__ __forceinline__ out_opmath_t warp_reduce(out_opmath_t val) {
    constexpr int HALF_WARP_SIZE = 16;

#if !defined(USE_ROCM)
    constexpr unsigned int FULL_WARP_MASK = 0xffffffff; // use all 32 bits
    for (int offset = HALF_WARP_SIZE; offset > 0; offset /= 2) {
      if constexpr (norm_type == NormType::LInf) {
        val = max_propagate_nan(
            val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
      } else {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
      }
    }
#else
    // unsigned int FULL_WARP_MASK = 0xffffffff does not work on ROCM
    constexpr int WARP_SIZE = 32;
    for (int offset = HALF_WARP_SIZE; offset > 0; offset /= 2) {
      if constexpr (norm_type == NormType::LInf) {
        val = max_propagate_nan(val, __shfl_down(val, offset, WARP_SIZE));
      } else {
        val += __shfl_down(val, offset, WARP_SIZE);
      }
    }
#endif
    return val;
  }

  __device__ __forceinline__ void lp_row_norm(TensorListDimMetadata<2>& tl) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    const auto num_rows = tl.num_rows[tensor_loc];
    const auto num_cols = tl.num_cols[tensor_loc];

    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;

    T* data = (T*)tl.addresses[0][tensor_loc];

    constexpr int ROWS_PER_CHUNK = 16; // blockDim.y

    const int chunk_row_start = chunk_idx * ROWS_PER_CHUNK;

    const int row = chunk_row_start + warp_id;

    if (row >= num_rows)
      return;

    out_opmath_t val = out_opmath_t(0);
    T* row_ptr = data + row * num_cols;

    if (num_cols % kILP == 0 && is_aligned(row_ptr)) {
      // fast path using Instruction-Level Parallelism (ILP)
      out_opmath_t vals[kILP] = {0};
      T r_x[kILP];

      for (int64_t i_start = lane; i_start * kILP < num_cols;
           i_start += blockDim.x) {
        load_store(r_x, row_ptr, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          const auto elem = static_cast<out_opmath_t>(r_x[ii]);
          if constexpr (norm_type == NormType::LInf) {
            vals[ii] = max_propagate_nan(vals[ii], ::abs(elem));
          } else {
            vals[ii] += norm_type == NormType::L1 ? ::abs(elem) : elem * elem;
          }
        }
      }

      for (int ii = 0; ii < kILP; ii++) {
        if constexpr (norm_type == NormType::LInf) {
          val = max_propagate_nan(val, vals[ii]);
        } else {
          val += vals[ii];
        }
      }
    } else {
      for (int64_t i = lane; i < num_cols; i += 32) {
        const auto elem = static_cast<out_opmath_t>(row_ptr[i]);
        if constexpr (norm_type == NormType::LInf) {
          val = max_propagate_nan(val, ::abs(elem));
        } else {
          val += norm_type == NormType::L1 ? ::abs(elem) : elem * elem;
        }
      }
    }

    val = warp_reduce(val);

    if (lane == 0) {
      if constexpr (norm_type == NormType::L2) {
        val = ::sqrt(val);
      }
      out_t* output_ptr = (out_t*)tl.addresses[1][tensor_loc];
      output_ptr[row] = static_cast<out_t>(val);
    }
  }

  __device__ __forceinline__ void lp_col_norm(TensorListDimMetadata<2>& tl) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    const auto num_rows = tl.num_rows[tensor_loc];
    const auto num_cols = tl.num_cols[tensor_loc];
    const auto prod_of_other_dim = tl.prod_of_other_dim[tensor_loc];

    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;

    T* data = (T*)tl.addresses[0][tensor_loc];

    constexpr int COLS_PER_CHUNK = 16; // blockDim.y

    // can think of N-dimensional tensors as a large 2-dimensional tensor
    // (matrix) stacked horizontally next to each other. This col is the col of
    // that larger matrix. Note: the index for the start of that column needs to
    // be adjusted due to how the memory is laid out (see col_start_index).
    const int chunk_col_start = chunk_idx * COLS_PER_CHUNK;
    const int col = chunk_col_start + warp_id;
    if (col >= num_cols * prod_of_other_dim)
      return;
    int64_t col_start_index =
        num_rows * num_cols * (col / num_cols) + (col % num_cols);

    out_opmath_t val = out_opmath_t(0);

    for (int64_t i = lane; i < num_rows; i += 32) {
      const auto elem =
          static_cast<out_opmath_t>(data[i * num_cols + col_start_index]);
      if constexpr (norm_type == NormType::LInf) {
        val = max_propagate_nan(val, ::abs(elem));
      } else {
        val += norm_type == NormType::L1 ? ::abs(elem) : elem * elem;
      }
    }

    val = warp_reduce(val);

    if (lane == 0) {
      if constexpr (norm_type == NormType::L2) {
        val = ::sqrt(val);
      }
      out_t* output_ptr = (out_t*)tl.addresses[1][tensor_loc];
      output_ptr[col] = static_cast<out_t>(val);
    }
  }

  __device__ __forceinline__ void operator()(
      int64_t /*unused_chunk_size*/,
      TensorListDimMetadata<2>& tl) {
    if constexpr (reduce_dim == 1) {
      lp_row_norm(tl);
    } else {
      lp_col_norm(tl);
    }
  }
};

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

// Traits struct for dispatch function names - MSVC requires this pattern
// instead of constexpr ternary for string literals used in AT_DISPATCH macros
template <bool apply_root>
struct ForeachNormDispatchName;

template <>
struct ForeachNormDispatchName<true> {
  static constexpr const char* value = "foreach_tensor_norm_cuda";
};

template <>
struct ForeachNormDispatchName<false> {
  static constexpr const char* value = "foreach_tensor_powsum_cuda";
};

std::vector<Tensor> foreach_tensor_norm_cuda(
    TensorList tensors,
    at::IntArrayRef dim,
    bool keepdim,
    const Scalar& ord,
    std::optional<ScalarType> dtype) {
  const auto p = [&]() -> double {
    if (ord.isIntegral(false)) {
      return ord.to<int64_t>();
    } else if (ord.isFloatingPoint()) {
      return ord.to<double>();
    } else {
      TORCH_CHECK(
          false, "foreach_tensor_norm_cuda expects ord to be integer or float");
    }
  }();
  check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  auto tensors_dim = tensors[0].dim();
  const bool has_same_dim = std::all_of(
      tensors.begin() + 1, tensors.end(), [tensors_dim](const auto& t) {
        return t.dim() == tensors_dim;
      });

  const bool all_contiguous =
      std::all_of(tensors.begin(), tensors.end(), [](const auto& t) {
        return t.is_contiguous();
      });
  if (!can_use_fast_route(tensors) || !has_same_dim || !all_contiguous ||
      has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2) ||
        p == std::numeric_limits<double>::infinity())) {
    return foreach_tensor_norm_slow(tensors, dim, keepdim, ord, dtype);
  }
  check_foreach_norm_dtype(
      dtype, tensors[0].scalar_type(), "_foreach_tensor_norm_cuda");

  const size_t ntensors = tensors.size();

  const auto options = tensors[0].options();
  const ScalarType output_dtype =
      dtype.has_value() ? dtype.value() : tensors[0].scalar_type();

  std::vector<at::Tensor> output_tensors;
  output_tensors.reserve(ntensors);
  const auto res_option = options.dtype(output_dtype);
  int64_t reduce_dim = c10::maybe_wrap_dim(dim[0], tensors_dim);
  for (const auto t : c10::irange(ntensors)) {
    auto output_tensor_size = tensors[t].sizes().vec();
    if (keepdim) {
      output_tensor_size[reduce_dim] = 1;
    } else {
      output_tensor_size.erase(output_tensor_size.begin() + reduce_dim);
    }

    output_tensors.push_back(at::native::empty_cuda(
        output_tensor_size,
        optTypeMetaToScalarType(res_option.dtype_opt()),
        res_option.layout_opt(),
        res_option.device_opt(),
        res_option.pinned_memory_opt(),
        res_option.memory_format_opt()));
  }

  std::vector<at::Tensor> input_tensors;
  if (tensors_dim == 2 || reduce_dim != tensors_dim - 1) {
    // If tensor is already 2D matrix
    // Or if not a row norm, in which case reshaping doesn't help since tensors
    // are not contiguous across reduce dimension, then use as is.
    input_tensors = tensors.vec();
  } else {
    // view tensor as 2D matrix to compute row norm
    // Note, this works even if output_tensors are multidimensional since
    // tensors are contiguous and kernel views tensor as contiguous 1D-array.
    input_tensors.reserve(ntensors);
    for (const auto t : c10::irange(ntensors)) {
      if (tensors[t].size(-1) != 0)
        input_tensors.push_back(tensors[t].view({-1, tensors[t].size(-1)}));
    }
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{
      std::move(input_tensors), output_tensors};

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      c10::kBFloat16,
      tensor_lists[0][0].scalar_type(),
      "foreach_tensor_norm_cuda_scalar_type",
      [&]() {
        AT_DISPATCH_OUT_DTYPES(
            output_dtype, "foreach_tensor_norm_cuda_out_dtype", [&]() {
              if (reduce_dim == tensors_dim - 1) { // row norm
                reduce_dim = 1;
                if (p == static_cast<double>(1)) {
                  multi_tensor_apply_dim(
                      tensor_lists,
                      reduce_dim,
                      LpNormDimFunctor<scalar_t, NormType::L1, out_t, 1>());
                } else if (p == static_cast<double>(2)) {
                  multi_tensor_apply_dim(
                      tensor_lists,
                      reduce_dim,
                      LpNormDimFunctor<scalar_t, NormType::L2, out_t, 1>());
                } else {
                  multi_tensor_apply_dim(
                      tensor_lists,
                      reduce_dim,
                      LpNormDimFunctor<scalar_t, NormType::LInf, out_t, 1>());
                }
              } else { // col norm
                if (p == static_cast<double>(1)) {
                  multi_tensor_apply_dim(
                      tensor_lists,
                      reduce_dim,
                      LpNormDimFunctor<scalar_t, NormType::L1, out_t, 0>());
                } else if (p == static_cast<double>(2)) {
                  multi_tensor_apply_dim(
                      tensor_lists,
                      reduce_dim,
                      LpNormDimFunctor<scalar_t, NormType::L2, out_t, 0>());
                } else {
                  multi_tensor_apply_dim(
                      tensor_lists,
                      reduce_dim,
                      LpNormDimFunctor<scalar_t, NormType::LInf, out_t, 0>());
                }
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return output_tensors;
}

// Internal implementation for foreach_tensor_norm_cuda and
// foreach_tensor_powsum_cuda apply_root: if true, applies sqrt for L2 norm; if
// false, returns raw sum support_infinity: if true, includes L-infinity norm
// handling
template <bool apply_root, bool support_infinity>
std::vector<Tensor> foreach_tensor_norm_cuda_internal(
    TensorList tensors,
    double p,
    std::optional<ScalarType> dtype) {
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
  const ScalarType output_per_tensor_dtype = toOpMathType(output_dtype);
  auto output_per_tensor = at::zeros(
      {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
      options.dtype(output_per_tensor_dtype));

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(ntensors);
  const auto res_option = options.dtype(output_dtype);
  for (const auto i : c10::irange(ntensors)) {
    vec_res.push_back(at::native::empty_cuda(
        {},
        optTypeMetaToScalarType(res_option.dtype_opt()),
        res_option.layout_opt(),
        res_option.device_opt(),
        res_option.pinned_memory_opt(),
        res_option.memory_format_opt()));
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      c10::kBFloat16,
      tensor_lists[0][0].scalar_type(),
      ForeachNormDispatchName<apply_root>::value,
      [&]() {
        AT_DISPATCH_OUT_DTYPES(
            output_dtype, ForeachNormDispatchName<apply_root>::value, [&]() {
              using out_opmath_t = typename at::opmath_type<out_t>;
              if (p == static_cast<double>(1)) {
                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::L1, out_t>(),
                    output_per_tensor.template mutable_data_ptr<out_opmath_t>(),
                    max_chunks_per_tensor);
              } else if (p == static_cast<double>(2)) {
                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::L2, out_t>(),
                    output_per_tensor.template mutable_data_ptr<out_opmath_t>(),
                    max_chunks_per_tensor);
              } else if constexpr (support_infinity) {
                if (p == std::numeric_limits<double>::infinity()) {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<scalar_t, NormType::LInf, out_t>(),
                      output_per_tensor
                          .template mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                }
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
                for (const auto j : c10::irange(num_tensors_this_kernel)) {
                  addr_struct.addresses[j] =
                      vec_res[i * MAX_TENSORS_PER_KERNEL + j]
                          .template mutable_data_ptr<out_t>();
                }

                if (p == static_cast<double>(1)) {
                  lpnorm_cleanup<scalar_t, NormType::L1, out_t, apply_root>
                      <<<num_tensors_this_kernel, 512, 0, stream>>>(
                          output_per_tensor
                                  .template const_data_ptr<out_opmath_t>() +
                              i * MAX_TENSORS_PER_KERNEL *
                                  max_chunks_per_tensor,
                          addr_struct,
                          max_chunks_per_tensor);
                } else if (p == static_cast<double>(2)) {
                  lpnorm_cleanup<scalar_t, NormType::L2, out_t, apply_root>
                      <<<num_tensors_this_kernel, 512, 0, stream>>>(
                          output_per_tensor
                                  .template const_data_ptr<out_opmath_t>() +
                              i * MAX_TENSORS_PER_KERNEL *
                                  max_chunks_per_tensor,
                          addr_struct,
                          max_chunks_per_tensor);
                } else if constexpr (support_infinity) {
                  if (p == std::numeric_limits<double>::infinity()) {
                    lpnorm_cleanup<scalar_t, NormType::LInf, out_t, apply_root>
                        <<<num_tensors_this_kernel, 512, 0, stream>>>(
                            output_per_tensor
                                    .template const_data_ptr<out_opmath_t>() +
                                i * MAX_TENSORS_PER_KERNEL *
                                    max_chunks_per_tensor,
                            addr_struct,
                            max_chunks_per_tensor);
                  }
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            });
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
      result.emplace_back(at::zeros({}, res_option));
    }
  }
  return result;
}

// note(mkozuki): Why excluding Int and Complex from fast path
// - Int: at::norm does not support.
// - Complex: __shfl_down_sync does not support complex and foreach does not
// support functions whose inputs dtypes and output dtype are different.
std::vector<Tensor> foreach_tensor_norm_cuda(
    TensorList tensors,
    const Scalar& ord,
    std::optional<ScalarType> dtype) {
  const auto p = [&]() -> double {
    if (ord.isIntegral(false)) {
      return ord.to<int64_t>();
    } else if (ord.isFloatingPoint()) {
      return ord.to<double>();
    } else {
      TORCH_CHECK(
          false, "foreach_tensor_norm_cuda expects ord to be integer or float");
    }
  }();
  check_foreach_api_restrictions(tensors);
  // If the tensor is empty and norm == infty, we cannot compute the norm
  // because the operation does not have an identity. Also populate the
  // has_int_or_complex flag.
  bool has_int_or_complex = false;
  if (p == std::numeric_limits<double>::infinity()) {
    for (const auto& t : tensors) {
      TORCH_SYM_CHECK(
          t.sym_numel().sym_gt(0),
          "_foreach_norm cannot compute the infinity norm on an empty tensor because the operation does not have an identity");
      const auto scalar_type = t.scalar_type();
      if (at::isIntegralType(scalar_type, /*includeBool*/ true) ||
          at::isComplexType(scalar_type)) {
        has_int_or_complex = true;
      }
    }
  } else {
    has_int_or_complex =
        std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
          const auto scalar_type = t.scalar_type();
          return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
              at::isComplexType(scalar_type);
        });
  }
  if (!can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2) ||
        p == std::numeric_limits<double>::infinity())) {
    return foreach_tensor_norm_slow(tensors, ord, dtype);
  }
  check_foreach_norm_dtype(
      dtype, tensors[0].scalar_type(), "_foreach_tensor_norm_cuda");

  return foreach_tensor_norm_cuda_internal<
      /*apply_root=*/true,
      /*support_infinity=*/true>(tensors, p, dtype);
}

// _foreach_powsum: like _foreach_norm but returns sum(|x|^p) without the root
// Fast path only for p=1 and p=2; other values fall back to slow path
std::vector<Tensor> foreach_tensor_powsum_cuda(
    TensorList tensors,
    const Scalar& ord,
    std::optional<ScalarType> dtype) {
  const auto p = [&]() -> double {
    if (ord.isIntegral(false)) {
      return ord.to<int64_t>();
    } else if (ord.isFloatingPoint()) {
      return ord.to<double>();
    } else {
      TORCH_CHECK(
          false,
          "foreach_tensor_powsum_cuda expects ord to be integer or float");
    }
  }();
  check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  // Only use fast path for p=1 or p=2
  if (!can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2))) {
    return foreach_tensor_powsum_slow(tensors, ord, dtype);
  }
  check_foreach_norm_dtype(
      dtype, tensors[0].scalar_type(), "_foreach_tensor_powsum_cuda");

  return foreach_tensor_norm_cuda_internal<
      /*apply_root=*/false,
      /*support_infinity=*/false>(tensors, p, dtype);
}

#undef AT_DISPATCH_OUT_DTYPES

} // namespace at::native
