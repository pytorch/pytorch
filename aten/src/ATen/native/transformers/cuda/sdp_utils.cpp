#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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
#include <c10/util/CallOnce.h>

#include <c10/core/SymInt.h>
#include <c10/util/string_view.h>
#include <cmath>
#include <functional>

#if USE_ROCM
#include <aotriton/flash.h>
#endif

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
// flash_attention V2 is universally faster than efficient_attention and Math
std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::cudnn_attention,
      SDPBackend::flash_attention,
      SDPBackend::efficient_attention,
      SDPBackend::math};
  return default_order;
}

bool use_tensor_cores(sdp_params const& params, cudaDeviceProp* dprops, bool is_half) {
  if (dprops->major >= 8) {
    return true;
  }
  if (dprops->major >= 7) {
    return is_half;
  }
  return false;
}
int64_t minimum_gemm_alignment(sdp_params const& params) {
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

bool check_head_dim_size_flash(sdp_params const& params, bool debug) {
  // All head_dim sizes must be equal and less than 256
  const auto max_size = c10::SymInt(256);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension and to be less than or equal to 256.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_head_dim_size_flash_nested(sdp_params const& params, bool debug) {
  const auto max_size = c10::SymInt(256);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last % 8 == 0) &&
        (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "For NestedTensor inputs, Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 256.",
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

bool check_head_dim_size_mem_efficient(sdp_params const& params, bool debug) {
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

bool check_flash_attention_hardware_support(sdp_params const& params, bool debug) {
  // Check that the gpu is capable of running flash attention
  using sm80 = SMVersion<8, 0>;
  using sm90 = SMVersion<9, 0>;
#if USE_ROCM
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (hipSuccess != aotriton::v2::flash::check_gpu(stream)) {
      auto dprops = at::cuda::getCurrentDeviceProperties();
      if (debug) {
          TORCH_WARN(
                  "Flash attention was not compiled for current AMD GPU architecture. Attempting to run on architecture ", dprops->gcnArchName);
      }
      return false;
  }
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm80, sm90>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention only supports gpu architectures in the range [sm80, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
#endif
  return true;
}

bool check_mem_efficient_hardware_support(sdp_params const& params, bool debug) {
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

bool check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89(
    sdp_params const& params,
    bool debug) {
  // Flash Attention will raise an error in the backward pass if the head_dim
  // size is greater than 192 And the device is between in the range [sm86, sm89]
  using sm86 = SMVersion<8, 6>;
  using sm89 = SMVersion<8, 9>;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm86_or_sm89 = check_sm_version<sm86, sm89>(dprops);
  bool is_head_dim_gt192 = params.query.sym_size(-1) > 192;
  bool is_head_dim_lte224 = params.query.sym_size(-1) <= 224;
  bool is_dropout = params.dropout > 0.0;
  //  head_dim size  in (192, 224] is not supported on sm86 and sm89
  bool cond1 = is_head_dim_gt192 && is_head_dim_lte224;
  // head_dim size > 224 and is_dropout is not supported on sm86 and sm89
  bool cond2 = params.query.sym_size(-1) > 224 && is_dropout;
  if (input_requires_grad(params) && is_sm86_or_sm89 && (cond1 || cond2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention currently doesn't support training with head_dim ∈ (192, 224] or "
          "(head_dim ∈ (224, 256] and dropout > 0.0) on gpu architectures in the range[sm86, sm89].",
          "Attempting to run with dropout set to: ", params.dropout,
          "and head_dim: ",
          params.query.sym_size(-1), " on a sm ", dprops->major, ".",
          dprops->minor, " gpu.");
    }
    return false;
  }
  return true;
}

bool check_flash_causal_non_square_seqlens(sdp_params const& params, bool debug) {
  // FlashAttention 2 updated the default mask meaning for causal in this PR:
  // 9e5e8bc91e it is now aligned to lower_right which would be a BC break
  // for non-square masks. We will not support non-square masks for causal w/ FAV2
  if (params.is_causal &&
      !params.query.is_nested() && !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ", params.query.sym_size(-2), " seqlen_k: ",
          params.key.sym_size(-2), ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}

bool check_all_tensors_on_device(sdp_params const& params, bool debug) {
  // Check that all tensors are on the GPU device
  // This should be handled by the stub dispatch, but whe call can_use_*_attention
  // directly from python we need to ensure that the tensors are on cuda
  if (params.query.device().type() != at::DeviceType::CUDA) {
    if (debug) {
      TORCH_WARN(
          "All tensors need to be on cuda device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  return true;
}

bool check_cudnn_tensor_shapes(sdp_params const& params, bool debug) {
  const auto num_heads{params.query.sym_size(1)},
      query_lengths{params.query.sym_size(2)},
      head_dim{params.query.sym_size(3)};
  const bool ok = query_lengths % 64 == 0 && head_dim % 64 == 0;
  if (!ok) {
    if (debug) {
      TORCH_WARN(
          "CuDNN requires sequence length and head dim to be divisible by 64. Got sequence length: ",
          query_lengths,
          ", head dim: ",
          head_dim,
          ".");
    }
    return false;
  }
  return true;
}

bool check_cudnn_layout(sdp_params const& params, bool debug) {
  const int64_t h = params.query.size(1);
  const int64_t s_q = params.query.size(2);
  const int64_t d = params.query.size(3);
  const int64_t s_k = params.key.size(2);
  const int64_t s_v = params.value.size(2);
  // corresponds to cuDNN's "packed QKV" layout
  const bool query_layout_ok = (params.query.stride(0) == s_q * 3 * h * d) &&
                                 (params.query.stride(1) == d) &&
                                 (params.query.stride(2) == 3 * h * d) &&
                                 (params.query.stride(3) == 1);
  const bool key_layout_ok = (params.key.stride(0) == s_k * 3 * h * d) &&
                               (params.key.stride(1) == d) &&
                               (params.key.stride(2) == 3 * h * d) &&
                               (params.key.stride(3) == 1);
  const bool value_layout_ok = (params.value.stride(0) == s_v * 3 * h * d) &&
                                 (params.value.stride(1) == d) &&
                                 (params.value.stride(2) == 3 * h * d) &&
                                 (params.value.stride(3) == 1);
  if (debug) {
    if (!query_layout_ok) { TORCH_WARN("Query tensor was not in cuDNN-supported packed QKV layout", params.query.strides()); }
    if (!key_layout_ok) { TORCH_WARN("Key tensor was not in cuDNN-supported packed QKV layout"); }
    if (!value_layout_ok) { TORCH_WARN("Value tensor was not in cuDNN-supported packed QKV layout"); }
  }
  return query_layout_ok && key_layout_ok && value_layout_ok;
}

bool check_cudnn_hardware_support(sdp_params const& params, bool debug) {
  using sm80 = SMVersion<8, 0>;
  using sm90 = SMVersion<9, 0>;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm80, sm90>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "cuDNN MHA only supports gpu architectures in the range [sm80, sm90]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

bool check_is_causal(sdp_params const& params, bool debug) {
  // Check that the input is causal
  if (!params.is_causal) {
    if (debug) {
      TORCH_WARN("CuDNN requires is_causal=True.");
    }
    return false;
  }
  return true;
}

bool check_for_nested_inputs(sdp_params const& params, bool debug) {
  // Check that the input is nested
  if (has_for_nested_inputs(params)) {
    if (debug) {
      TORCH_WARN("CuDNN currently does not support nested inputs.");
    }
    return false;
  }
  return true;
}

bool check_dtypes_low_precision(sdp_params const& params, bool debug) {
  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    constexpr auto sm80_dtypes =
        array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
    return check_tensor_dtype(params, sm80_dtypes, debug);
  } else {
    constexpr auto default_dtypes = array_of<at::ScalarType>(at::kHalf);
    return check_tensor_dtype(params, default_dtypes, debug);
  }
}

bool check_runtime_enabled_cudnn(sdp_params const& params, bool debug) {
  static c10::once_flag supported_flag;
  static bool supported = false;
  c10::call_once(supported_flag, []() {
    supported = (c10::utils::check_env("TORCH_CUDNN_SDPA_ENABLED") == true);
  });
  if (!supported) {
    if (debug) {
      TORCH_WARN(
          "The CuDNN backend needs to be enabled by setting the enviornment variable`TORCH_CUDNN_SDPA_ENABLED=1`");
    }
    return false;
  }
  return true;
}

bool check_cudnn_requires_grad(sdp_params const& params, bool debug) {
  // Check that the input is causal
  if (input_requires_grad(params)) {
    if (debug) {
      TORCH_WARN("CuDNN does not currently support inputs with requires_grad=True.");
    }
    return false;
  }
  return true;
}

} // namespace

bool can_use_cudnn_attention(const sdp_params& params, bool debug) {

  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints =
      array_of<bool (*)(sdp_params const&, bool)>(
          check_runtime_enabled_cudnn,
          check_cudnn_hardware_support);
          // check_all_tensors_on_device,
          // check_cudnn_tensor_shapes,
          // check_cudnn_layout,
          // check_is_causal,
          // check_for_nested_inputs,
          // check_cudnn_requires_grad,
          // check_dtypes_low_precision
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return true;
}

bool can_use_flash_attention(sdp_params const& params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_WARN_ONCE(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif

  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints = array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_flash,
      check_all_tensors_on_device,
      check_tensor_shapes,
      check_for_attn_mask,
      check_head_dim_size_flash,
      check_flash_attention_hardware_support,
      check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89,
      check_flash_causal_non_square_seqlens,
      check_dtypes_low_precision);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_nested,
        check_head_dim_size_flash_nested,
        check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
}

bool can_use_mem_efficient_attention(sdp_params const& params, bool debug) {
#ifndef USE_MEM_EFF_ATTENTION
  TORCH_WARN_ONCE(!debug, "Torch was not compiled with memory efficient attention.");
  return false;
#endif
  // Constraints specific to mem efficient attention
  constexpr auto greater_than_or_equal_sm80_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);
  constexpr auto less_than_sm80_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kFloat);

  //  Define gate functions that determine if a mem efficient kernel can be ran
  constexpr auto general_constraints = array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_mem_efficient,
      check_all_tensors_on_device,
      check_mem_efficient_hardware_support,
      check_tensor_shapes,
      check_head_dim_size_mem_efficient);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        check_requires_grad_and_nested,
        check_batch_size_nested,
        check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    return check_tensor_dtype(params, greater_than_or_equal_sm80_mem_efficient_dtypes, debug);
  }
  return check_tensor_dtype(params, less_than_sm80_mem_efficient_dtypes, debug);
}

SDPBackend select_sdp_backend(sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP() && !ctx.userEnabledCuDNNSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_attention:
        if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) {
              TORCH_WARN("USING CUDNN SDPA");
              return SDPBackend::cudnn_attention;
        }
        break;
      case SDPBackend::flash_attention:
        if (sdp::can_use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) {
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
  sdp::can_use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  sdp::can_use_flash_attention(kernel_params, print_debug);
  TORCH_WARN("CuDNN attention kernel not used because:");
  sdp::can_use_cudnn_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return SDPBackend::error;
}

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug) {
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
