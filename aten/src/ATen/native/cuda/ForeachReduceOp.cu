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
#include <ATen/ops/_foreach_norm_native.h>

#include <ATen/ops/empty.h>
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
    NormType norm_type,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpNormFunctor {
  using opmath_t = typename at::opmath_type<T>;
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorListMetadata<depth>& tl,
      opmath_t* output_per_tensor,
      const int max_chunks_per_tensor) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];
    auto n = tl.numel_for_tensor[tensor_loc];

    T* x = (T*)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    __shared__ opmath_t s_vals[512];
    opmath_t vals[kILP];
    T r_x[kILP];
    for (int64_t i = 0; i < kILP; i++) {
      vals[i] = opmath_t(0);
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
          opmath_t next = static_cast<opmath_t>(r_x[ii]);
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
            opmath_t next = static_cast<opmath_t>(x[i]);
            if constexpr (norm_type == NormType::LInf) {
              vals[ii] = max_propagate_nan(vals[ii], ::abs(next));
            } else {
              vals[ii] += norm_type == NormType::L1 ? ::abs(next) : next * next;
            }
          }
        }
      }
    }

    auto val = opmath_t(0);
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
      output_per_tensor
          [(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor +
           chunk_idx] = final_val;
    }
  }
};

template <
    typename T,
    NormType norm_type,
    typename opmath_t = at::opmath_type<T>>
__global__ void lpnorm_cleanup(
    const opmath_t* output_per_tensor,
    TensorListAddresses addr_struct,
    int max_chunks_per_tensor) {
  __shared__ opmath_t vals[512];

  const opmath_t* output_this_tensor =
      output_per_tensor + blockIdx.x * max_chunks_per_tensor;
  opmath_t val = 0;
  for (size_t i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) {
    if constexpr (norm_type == NormType::LInf) {
      val = max_propagate_nan(val, output_this_tensor[i]);
    } else {
      val += output_this_tensor[i];
    }
  }
  opmath_t final_val = norm_type == NormType::L1 || norm_type == NormType::L2
      ? at::native::cuda_utils::BlockReduceSum<opmath_t>(val, vals)
      : at::native::cuda_utils::BlockReduceMax(val, vals);
  if (threadIdx.x == 0) {
    *(T*)addr_struct.addresses[blockIdx.x] =
        norm_type == NormType::L1 || norm_type == NormType::LInf
        ? final_val
        : ::sqrt(final_val);
  }
}

// note(mkozuki): Why excluding Int and Complex from fast path
// - Int: at::norm does not support.
// - Complex: __shfl_down_sync does not support complex and foreach does not
// support functions whose inputs dtypes and output dtype are different.
std::vector<Tensor> foreach_tensor_norm_cuda(
    TensorList tensors,
    const Scalar& ord) {
  double p;
  if (ord.isIntegral(false)) {
    p = ord.to<int64_t>();
  } else if (ord.isFloatingPoint()) {
    p = ord.to<double>();
  } else {
    TORCH_CHECK(
        false, "foreach_tensor_norm_cuda expects ord to be integer or float");
  }
  check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  if (!can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2) ||
        p == std::numeric_limits<double>::infinity())) {
    return foreach_tensor_norm_slow(tensors, ord);
  }

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
      {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
      options.dtype(toOpMathType(tensors[0].scalar_type())));

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(ntensors);
  for (const auto i : c10::irange(ntensors)) {
    vec_res.push_back(at::empty({}, options));
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};
  if (p == static_cast<double>(1)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_tensor_norm_cuda",
        [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          multi_tensor_apply<1>(
              tensor_lists,
              LpNormFunctor<scalar_t, NormType::L1>(),
              output_per_tensor.mutable_data_ptr<opmath_t>(),
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

            lpnorm_cleanup<scalar_t, NormType::L1>
                <<<num_tensors_this_kernel, 512, 0, stream>>>(
                    output_per_tensor.const_data_ptr<opmath_t>() +
                        i * MAX_TENSORS_PER_KERNEL * max_chunks_per_tensor,
                    addr_struct,
                    max_chunks_per_tensor);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  } else if (p == static_cast<double>(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_tensor_norm_cuda",
        [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          multi_tensor_apply<1>(
              tensor_lists,
              LpNormFunctor<scalar_t, NormType::L2>(),
              output_per_tensor.mutable_data_ptr<opmath_t>(),
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

            lpnorm_cleanup<scalar_t, NormType::L2>
                <<<num_tensors_this_kernel, 512, 0, stream>>>(
                    output_per_tensor.const_data_ptr<opmath_t>() +
                        i * MAX_TENSORS_PER_KERNEL * max_chunks_per_tensor,
                    addr_struct,
                    max_chunks_per_tensor);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  } else if (p == std::numeric_limits<double>::infinity()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_tensor_norm_cuda",
        [&]() {
          using opmath_t = typename at::opmath_type<scalar_t>;
          multi_tensor_apply<1>(
              tensor_lists,
              LpNormFunctor<scalar_t, NormType::LInf>(),
              output_per_tensor.mutable_data_ptr<opmath_t>(),
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

            lpnorm_cleanup<scalar_t, NormType::LInf>
                <<<num_tensors_this_kernel, 512, 0, stream>>>(
                    output_per_tensor.const_data_ptr<opmath_t>() +
                        i * MAX_TENSORS_PER_KERNEL * max_chunks_per_tensor,
                    addr_struct,
                    max_chunks_per_tensor);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  } else {
    TORCH_CHECK(
        false,
        "foreach_tensor_norm_cuda fast path got unexpected ord value: ",
        p);
  }

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
      result.emplace_back(at::zeros({}, options));
    }
  }
  return result;
}

} // namespace at::native
