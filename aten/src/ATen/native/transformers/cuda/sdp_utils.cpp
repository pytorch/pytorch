#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <c10/core/SymInt.h>
#include <c10/util/string_view.h>
#include <cmath>
#include <functional>
#include <iostream>

/**
* Note [SDPA Runtime Dispatch]
* SDPA relies on a runtime dispatch mechanism to select the appropriate
* kernel. This file contains exposes this through the `select_sdp_backend`
* The basic structure of this function is to call `priority_order` to get a
* list of backends to try, and then iterate through them until one succeeds.
* Each backend defines a use_<backend> function that returns true if the
* backend can be run with the given SDP parameters. The use_<backend> function
* will iterate over a list of "filters" that check for specific properties of
* the SDP parameters. If all filters pass, the backend can be used and use_<backend>
* returns true. If any filter fails, then use_<backend> returns false.
*
* In order to aid in debugging, each filter takes sdp_params and a debug flag.
* If the debug flag is set, the filter will print a warning message if it fails.
* The behavior of select_sdp_backend is to return the first backend that
* succeeds. If no backend is viable then it will run each use_<backend> function
* with debug=true and return SDPBackend::error.
*/

namespace sdp {
namespace {
std::array<SDPBackend, num_backends> priority_order(sdp_params params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::cudnn_mha,
      SDPBackend::flash_attention,
      SDPBackend::efficient_attention,
      SDPBackend::math};

  constexpr std::array<SDPBackend, num_backends> efficient_first{
      SDPBackend::efficient_attention,
      SDPBackend::flash_attention,
      SDPBackend::math};
  // Logic is taken from xformers
  // FlashAttention parallelizes across "batch_size * num_heads"
  // MemEff parallelizes across "batch_size * num_heads * num_queries" and can
  // be more efficient. batch_size, q_len, num_heads, k = inp.query.shape

  if (has_for_nested_inputs(params)) {
    return efficient_first;
  }
  if (params.query.dim() != 4) {
    return default_order;
  }
  const auto batch_size{params.query.sym_size(0)},
      num_heads{params.query.sym_size(1)},
      query_lengths{params.query.sym_size(2)},
      head_dim{params.query.sym_size(3)};
  if (batch_size > 0) {
    const auto threads_flash = batch_size * num_heads;
    const auto threads_cutlass =
        threads_flash * (query_lengths / c10::SymInt(64));
    bool more_threads_cutlass = (threads_cutlass / 2) >= threads_flash;
    bool small_threads_flash = threads_flash < 60;
    bool large_head_dim = head_dim.max(params.key.sym_size(3)) == 128;

    // The training heuristic is taken from
    // https://github.com/pytorch/pytorch/pull/99644 Revisit when updated
    // cutlass kernel is upstreamed.
    if (input_requires_grad(params)) {
      if (6 * threads_flash > query_lengths)
        return efficient_first;
    } else if ((small_threads_flash && more_threads_cutlass) || large_head_dim)
      return efficient_first;
  }
  return default_order;
}

bool check_head_dim_size(sdp_params params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (!(query_size_last == key_size_last &&
        query_size_last == value_size_last && query_size_last % 8 == 0 &&
        query_size_last <= 128 && value_size_last % 8 == 0 &&
        value_size_last <= 128)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 128.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

bool use_tensor_cores(sdp_params params, cudaDeviceProp* dprops, bool is_half) {
  if (dprops->major >= 8) {
    return true;
  }
  if (dprops->major >= 7) {
    return is_half;
  }
  return false;
}
int64_t minimum_gemm_alignment(sdp_params params) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_half = (params.query.dtype() == at::kHalf) ||
      (params.query.dtype() == at::kBFloat16);
  bool use_tc = use_tensor_cores(params, dprops, is_half);
  int64_t matmul_alignment_mn = 1;
  if (dprops->major >= 8) {
    matmul_alignment_mn = 4;
  }
  int64_t bits_per_scalar = is_half ? 16 : 32;
  if (use_tc) {
    matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar);
  }
  return matmul_alignment_mn;
}

bool check_head_dim_size_mem_efficient(sdp_params params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  const int64_t alignment = minimum_gemm_alignment(params);
  if (!(query_size_last == params.key.sym_size(-1) &&
        query_size_last % alignment == 0 && query_size_last > 0 &&
        value_size_last % alignment == 0 && value_size_last > 0)) {
    if (debug) {
      TORCH_WARN(
          "Mem efficient attention requires last dimension of inputs to be divisible by ",
          alignment,
          ". ",
          "Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

template <int Major, int Minor>
struct SMVersion {
  static constexpr int major = Major;
  static constexpr int minor = Minor;
  constexpr SMVersion() = default;
};

/**
 * Checks if the current CUDA device architecture is inclusively within the specified range.
 *
 * @param lower_bound The lower bound of the CUDA device architecture range.
 * @param upper_bound The upper bound of the CUDA device architecture range.
 * @param params The parameters for the current operation.
 * @return True if the current CUDA device architecture is within the specified range, false otherwise.
 */
template <typename lower_bound, typename upper_bound>
bool check_sm_version(cudaDeviceProp * dprops) {
  bool is_gte_lower_bound = dprops->major > lower_bound::major ||
      (dprops->major == lower_bound::major &&
       dprops->minor >= lower_bound::minor);
  bool is_lte_upper_bound = dprops->major < upper_bound::major ||
      (dprops->major == upper_bound::major &&
       dprops->minor <= upper_bound::minor);
  return is_gte_lower_bound && is_lte_upper_bound;
}

bool check_gpu_sm75_or_greater(sdp_params params, bool debug) {
  // Check that the gpu is capable of running flash attention
  using sm75 = SMVersion<7, 5>;
  using sm90 = SMVersion<9, 0>;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm75, sm90>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention only supports gpu architectures in the range [sm75, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

bool check_mem_efficient_hardware_support(sdp_params params, bool debug) {
  // Mem Efficient attention supports hardware in the range [sm_50, sm_90]
  using sm50 = SMVersion<5, 0>;
  using sm90 = SMVersion<9, 0>;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm50, sm90>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Mem Efficient Attention only supports gpu architectures in the range [sm50, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

bool check_requires_grad_and_head_dim_gt64_and_sm_ge86_lt90(
    sdp_params params,
    bool debug) {
  // Flash Attention will raise an error in the backward pass if the head_dim
  // size is greater than 64 And the device is between in the range [sm86, sm89]
  using sm86 = SMVersion<8, 6>;
  using sm89 = SMVersion<8, 9>;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm86_or_sm89 = check_sm_version<sm86, sm89>(dprops);
  bool is_head_dim_gt64 = params.query.sym_size(-1) > 64;
  if (input_requires_grad(params) && is_sm86_or_sm89 && is_head_dim_gt64) {
    if (debug) {
      TORCH_WARN(
          "Flash attention currently doesn't support training with head_dim greater than 64 on gpu architectures in the range[sm86, sm89].",
          "Attempting to run with head_dim: ",
          params.query.sym_size(-1), " on a sm ", dprops->major, ".",
          dprops->minor, " gpu.");
    }
    return false;
  }
  return true;
}

bool use_cudnn_mha(sdp_params kernel_params, bool print_debug) {
  static bool flag = c10::utils::check_env("TORCH_CUDNN_MHA_ENABLED") == true;
  return flag;
}

bool use_flash_attention(sdp_params params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_CHECK(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif

  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto constraints = array_of<bool (*)(sdp_params, bool)>(
      check_runtime_disabled_flash,
      check_tensor_shapes,
      check_batch_size_and_num_heads,
      check_for_attn_mask,
      check_head_dim_size,
      check_gpu_sm75_or_greater,
      check_requires_grad_and_head_dim_gt64_and_sm_ge86_lt90,
      check_for_seq_len_0_nested_tensor,
      check_nonzero_sequence_lengths,
      check_last_dim_stride_equals_1);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    constexpr auto sm80_flash_dtypes =
        array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
    return check_tensor_dtype(params, sm80_flash_dtypes, debug);
  } else {
    constexpr auto default_flash_dtypes = array_of<at::ScalarType>(at::kHalf);
    return check_tensor_dtype(params, default_flash_dtypes, debug);
  }
}

bool use_mem_efficient_attention(sdp_params params, bool debug) {
#ifndef USE_MEM_EFF_ATTENTION
  TORCH_CHECK(!debug, "Torch was not compiled with memory efficient attention.");
  return false;
#endif
  // Constraints specific to mem efficient attention
  constexpr auto default_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);
  constexpr auto sm50_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat);

  //  Define gate functions that determine if a flash kernel can be ran
  constexpr auto constraints = array_of<bool (*)(sdp_params, bool)>(
      check_runtime_disabled_mem_efficient,
      check_mem_efficient_hardware_support,
      check_requires_grad_and_nested,
      check_tensor_shapes,
      check_batch_size_and_num_heads,
      check_head_dim_size_mem_efficient,
      check_for_seq_len_0_nested_tensor,
      check_nonzero_sequence_lengths,
      check_last_dim_stride_equals_1);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major == 5) {
    return check_tensor_dtype(params, sm50_mem_efficient_dtypes, debug);
  }
  return check_tensor_dtype(params, default_mem_efficient_dtypes, debug);
}
} // namespace

SDPBackend select_sdp_backend(sdp_params kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_mha:
        if (use_cudnn_mha(kernel_params, print_debug)) {
              return SDPBackend::cudnn_mha;
        }
      case SDPBackend::flash_attention:
        if (use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (use_mem_efficient_attention(kernel_params, print_debug)) {
          return SDPBackend::efficient_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention or use_mem_efficient did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Memory efficient kernel not used because:");
  use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  use_flash_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  return SDPBackend::error;
}

bool check_for_seq_len_1_nested_tensor(sdp_params params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

} // namespace sdp
