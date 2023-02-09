#pragma once

#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <c10/util/Exception.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#include <functional>
#include <unordered_set>
#include <vector>
#include <cmath>

namespace sdp {

template <typename To, typename From>
To bit_cast(From f) {
  static_assert(sizeof(To) == sizeof(From));
  To t;
  std::memcpy(&t, &f, sizeof(f));
  return t;
}

struct sdp_params {
  const at::Tensor& query;
  const at::Tensor& key;
  const at::Tensor& value;
  bool has_attn_mask;
  double dropout;
  bool is_causal;
};

inline std::array<SDPBackend, num_backends> priority_order(sdp_params params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::efficient_attention,
      SDPBackend::math};
  // Logic is taken from xformers
  // FlashAttention parallelizes across "batch_size * num_heads"
  // MemEff parallelizes across "batch_size * num_heads * num_queries" and can
  // be more efficient. batch_size, q_len, num_heads, k = inp.query.shape
  if (params.query.is_nested()) {
    // See check_for_nested_inputs for details
    return {
        SDPBackend::efficient_attention,
        SDPBackend::flash_attention,
        SDPBackend::math};
  }
  const auto sizes = params.query.sizes();
  if (params.query.dim() != 4) {
    return default_order;
  }
  const auto batch_size{sizes[0]}, num_heads{sizes[1]}, query_lengths{sizes[2]},
      head_dim{sizes[3]};
  if (batch_size > 0) {
    const int64_t threads_flash = batch_size * num_heads;
    const int64_t threads_cutlass =
        threads_flash * (int64_t)std::floor(query_lengths / 64);
    bool more_threads_cutlass =
        (int64_t)std::floor(threads_cutlass / 2) >= threads_flash;
    bool small_threads_flash = threads_flash < 60;
    bool large_head_dim = std::max(head_dim, params.key.sizes()[3]) == 128;
    if ((small_threads_flash && more_threads_cutlass) || large_head_dim) {
      return {
          SDPBackend::efficient_attention,
          SDPBackend::flash_attention,
          SDPBackend::math};
    }
  }
  return default_order;
}

template <typename dtype_vector>
inline bool check_tensor_dtype(
    sdp_params params,
    dtype_vector allowed_dtypes,
    bool debug) {
  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), query_dtype) !=
         allowed_dtypes.end()))) {
    if (debug) {
      TORCH_WARN(
        "Expected query, key and value to all be of dtype: {",
        c10::Join(", ", allowed_dtypes), "}. Got ",
        "Query dtype: ",
        params.query.dtype(),
        ", Key dtype: ",
        params.key.dtype(),
        ", and Value dtype: ",
        params.value.dtype(),
        " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_for_non_zero_dropout(sdp_params params, bool debug) {
  if (params.dropout != 0.0) {
    if (debug) {
      TORCH_WARN("Mem_efficient does not support non_zero dropout. Dropout_p: ", params.dropout);
    }
    return false;
  }
  return true;
}

inline bool check_for_seq_len_1_nested_tensor(sdp_params params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }
  // we are only checking query but should probably check all of them
  const auto nt_q_tensor_impl = at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_size_tensor();
  auto num_head_dims = nt_q_tensor_impl->opt_size(1);
  if (!num_head_dims.has_value() ) {
    // num_head_dims is ragged
    if (debug) {
      TORCH_WARN("Memory efficient attention does not support ragged num_head_dims");
    }
    return false;
  }

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN("Memory efficient attention does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

inline bool check_for_nested_inputs(sdp_params params, bool debug){
  if (params.query.is_nested() || params.key.is_nested() || params.value.is_nested()) {
    if (debug) {
      TORCH_WARN("We are not enabling nested Tensors for Flash Attention because of cuda memory errors.");
    }
    return false;
  }
  return true;
}

inline bool check_requires_grad(sdp_params params, bool debug) {
  bool any_tensors_are_subclass =
      at::areAnyTensorSubclassLike({params.query, params.key, params.value});
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  if ((any_inputs_require_grad && gradmode_enabled) || any_tensors_are_subclass) {
    if (debug) {
      TORCH_WARN("Flash Attention does not currently support training.");
    }
    return false;
  }
  return true;
}

inline bool check_requires_grad_and_nested(sdp_params params, bool debug) {
  // If we fail both checks then we return false
  if (!check_for_nested_inputs(params, false) && !check_requires_grad(params,false)){
    if (debug){
      TORCH_WARN("Memory efficient attention currently doesn't support training with NT inputs.");
    }
    return false;
  }
  return true;
}

inline bool check_for_attn_mask(sdp_params params, bool debug) {
  if (params.has_attn_mask) {
    if (debug) {
      TORCH_WARN("Both fused kernels do not support non-null attn_mask.");
    }
    return false;
  }
  return true;
}

inline bool check_tensor_shapes(sdp_params params, bool debug) {
  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
        (query_dim == 4 ))) {
    if (debug) {
      TORCH_WARN(
        "Both fused kernels requires query, key and value to be 4 dimensional, but got Query dim: ",
        query_dim,
        ", Key dim: ",
        params.key.dim(),
        ", Value dim: ",
        params.value.dim(),
        " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_equal_batch_size_and_num_heads(sdp_params params, bool debug) {
  // This is expected to be called after check_tensor_shapes ensuring that the size()
  // calls won't error since the inputs are all 4 dimensional
  bool same_batch_size = params.query.size(0) == params.key.size(0) &&
      params.query.size(0) == params.value.size(0);
  // We pass through for NestedTensors since this is checked in a later filter
  bool same_num_heads = params.query.is_nested()
      ? true
      : params.query.size(1) == params.key.size(1) &&
          params.query.size(1) == params.value.size(1);

  if (!(same_batch_size && same_num_heads)) {
    if (debug) {
      TORCH_WARN(
        "Both fused kernels requires query, key and value to have the same batch_size and num_heads. Query.sizes(): ",
        params.query.sizes(),
        ", Key sizes(): ",
        params.key.sizes(),
        ", Value sizes(): ",
        params.value.sizes(),
        " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_head_dim_size(sdp_params params, bool debug) {
  const int64_t query_size_last = params.query.size(-1);
  const int64_t key_size_last = params.key.size(-1);
  const int64_t value_size_last = params.value.size(-1);
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
          params.key.size(-1),
          ", Value.size(-1): ",
          params.value.size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool use_tensor_cores(
    sdp_params params,
    cudaDeviceProp* dprops,
    bool is_half) {
  if (dprops->major >= 8) {
    return true;
  }
  if (dprops->major >= 7) {
    return is_half;
  }
  return false;
}
inline int64_t minimum_gemm_alignment(sdp_params params) {
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

inline bool check_head_dim_size_mem_efficient(sdp_params params, bool debug) {
  const int64_t query_size_last = params.query.size(-1);
  const int64_t value_size_last = params.value.size(-1);
  const int64_t alignment = minimum_gemm_alignment(params);
  if (!(query_size_last == params.key.size(-1) &&
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
        params.key.size(-1),
        ", Value.size(-1): ",
        params.value.size(-1),
        " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_flash(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of flash
  // sdp kernels
  if (!at::globalContext().userEnabledFlashSDP()) {
    if (debug) {
      TORCH_WARN("Flash attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_mem_efficient(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of mem_efficient
  // sdp kernels
  if (!at::globalContext().userEnabledMemEfficientSDP()) {
    if (debug) {
      TORCH_WARN("Memory Efficient attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

inline bool check_gpu_sm75_or_greater(sdp_params params, bool debug) {
  // Check that the gpu is capable of running flash attention
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  if (!(is_sm8x || is_sm75)) {
    if (debug) {
      TORCH_WARN(
        "Flash attention only supports sm75 and sm8x gpu architectures. Attempting to run on a sm ",
        dprops->major,
        ".",
        dprops->minor,
        " gpu.");
    }
    return false;
  }
  return true;
}

inline bool check_gpu_sm50_or_greater(sdp_params params, bool debug) {
  // Check that the gpu is capable of running flash attention
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm50 = dprops->major >= 5;
  if (!(is_sm50)) {
    if (debug) {
      TORCH_WARN(
        "Mem Efficient Attention only supports sm5x or greater gpu architectures. Attempting to run on a sm ",
        dprops->major,
        ".",
        dprops->minor,
        " gpu.");
    }
    return false;
  }
  return true;
}

inline bool check_gpu_sm86_head_dim_128(sdp_params params, bool debug) {
  // Memory Efficient Attention is throwing a cuda illegal memory error
  // on sm86 when head_dim is 128.
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm86 = (dprops->major == 8) && (dprops->minor == 6);
  if (is_sm86 && (params.query.size(-1) == 128)) {
    if (debug) {
      TORCH_WARN(
        "Memory Efficient Attention does not currently support head_dim == 128 on sm86",
        "because it is throwing a cuda illegal memory error on sm86 when head_dim is 128.");
    }
    return false;
  }
  return true;
}

inline bool check_use_deterministic_algorithms(sdp_params params, bool debug) {
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Memory Efficient attention is a non-deterministic algorithm. ",
          "To explicitly disable Memory Efficient attention call torch.use_deterministic_algorithms(True, warn_only=False).");
      // Warn the user but don't disable the kernel.
      return true;
    } else {
      if (debug) {
        TORCH_WARN(
            "Memory Efficient attention is a non-deterministic algorithm and torch.use_deterministic_algorithms(True) has been set.");
      }
      return false;
    }
  }
  // Determinism is not set so we can use the kernel.
  return true;
}

inline bool use_flash_attention(sdp_params params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_CHECK(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif
  //  Define gate functions that determine if a flash kernel can be ran
  constexpr std::array<bool(*)(sdp_params, bool), 8> constraints {{
      check_runtime_disabled_flash,
      check_tensor_shapes,
      check_equal_batch_size_and_num_heads,
      check_for_attn_mask,
      check_head_dim_size,
      check_gpu_sm75_or_greater,
      check_for_nested_inputs,
      check_for_seq_len_1_nested_tensor}};
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    static const std::array<at::ScalarType, 2> sm80_flash_dtypes{at::kHalf, at::kBFloat16};
    return check_tensor_dtype(params, sm80_flash_dtypes, debug);
  } else {
    static const std::array<at::ScalarType, 1> default_flash_dtypes{at::kHalf};
    return check_tensor_dtype(params, default_flash_dtypes, debug);
  }
}

inline bool use_mem_efficient_attention(sdp_params params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_CHECK(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif
  // Constraints specific to flash attention
  static const std::vector<caffe2::ScalarType> flash_dtypes{
      at::kHalf, at::kFloat, at::kBFloat16};

  //  Define gate functions that determine if a flash kernel can be ran
  constexpr std::array<bool(*)(sdp_params, bool), 11> constraints{{
      check_gpu_sm50_or_greater,
      check_runtime_disabled_mem_efficient,
      check_requires_grad_and_nested,
      check_tensor_shapes,
      check_equal_batch_size_and_num_heads,
      check_for_attn_mask,
      check_head_dim_size_mem_efficient,
      check_gpu_sm86_head_dim_128,
      check_for_seq_len_1_nested_tensor,
      check_for_non_zero_dropout,
      check_use_deterministic_algorithms}};
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  if (!check_tensor_dtype(params, flash_dtypes, debug)) {
    return false;
  }
  return true;
}

inline SDPBackend select_sdp_backend(sdp_params kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() && !ctx.userEnabledMemEfficientSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
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

} // namespace sdp
