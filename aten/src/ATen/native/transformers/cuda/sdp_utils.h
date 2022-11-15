#pragma once

#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <ATen/NestedTensorImpl.h>

#include <functional>
#include <unordered_set>
#include <vector>

namespace sdp {

struct sdp_params {
  const at::Tensor& query;
  const at::Tensor& key;
  const at::Tensor& value;
  bool has_attn_mask;
  double dropout;
  bool need_attn_weights;
  bool is_causal;
};

enum class SDPBackend { flash_attention, efficient_attention, math, error };

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
    TORCH_CHECK(
        !debug,
        "Expected query, key and value to be of dtype float16 or bfloat16 but got Query dtype: ",
        params.query.dtype(),
        ", Key dtype: ",
        params.key.dtype(),
        ", and Value dtype: ",
        params.value.dtype(),
        " instead.");
    return false;
  }
  return true;
}

inline bool check_for_attn_weights(sdp_params params, bool debug) {
  // This can be returned form flash attention but care is needed
  // to convert from flash_attn format to attn_weights
  if (params.need_attn_weights) {
    TORCH_CHECK(!debug, "Flash Attention does not support need attn weights");
    return false;
  }
  return true;
}

inline bool check_for_non_zero_dropout(sdp_params params, bool debug) {
  if (params.dropout != 0.0) {
    TORCH_CHECK(!debug, "Mem_efficient does not support non_zero dropout. Dropout_p: ", params.dropout);
    return false;
  }
  return true;
}

inline bool check_for_seq_len_1_nested_tensor(sdp_params params, bool debug) {
  if (!params.query.is_nested()) {
    return true;
  }
  const at::Tensor& sizes = at::native::get_nested_tensor_impl(params.query)->get_nested_size_tensor();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      TORCH_CHECK(
          !debug, "Flash Attention does not support sequence_length <= 1");
      return false;
    }
  }

  return true;
}

inline bool check_for_attn_mask(sdp_params params, bool debug) {
  if (params.has_attn_mask) {
    TORCH_CHECK(!debug, "Flash Attention does not support attention mask.");
    return false;
  }
  return true;
}

inline bool check_tensor_shapes(sdp_params params, bool debug) {
  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
        (query_dim == 4 ))) {
    TORCH_CHECK(
        !debug,
        "Flash attention requires query, key and value to be 4 dimensional, but got Query dim: ",
        query_dim,
        ", Key dim: ",
        params.key.dim(),
        ", Value dim: ",
        params.value.dim(),
        " instead.");
    return false;
  }
  return true;
}

inline bool check_head_dim_size(sdp_params params, bool debug) {
  const int64_t query_size_last = params.query.size(-1);
  if (!(query_size_last == params.key.size(-1) &&
        query_size_last == params.value.size(-1) && query_size_last % 8 == 0 &&
        query_size_last <= 128)) {
    TORCH_CHECK(
        !debug,
        "Flash attention requires last dimension of inputs to be a multiple of 8 and less than or equal to 128.",
        "Got Query.size(-1): ",
        query_size_last,
        ", Key.size(-1): ",
        params.key.size(-1),
        ", Value.size(-1): ",
        params.value.size(-1),
        " instead.");
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_flash(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of flash
  // sdp kernels
  if (!at::globalContext().userEnabledFlashSDP()) {
    TORCH_CHECK(!debug, "Flash attention has been runtime disabled.");
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_mem_efficient(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of mem_efficient
  // sdp kernels
  if (!at::globalContext().userEnabledMemEfficientSDP()) {
    TORCH_CHECK(!debug, "Memory Efficient attention has been runtime disabled.");
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
    TORCH_CHECK(
        !debug,
        "Flash attention only supports sm75 and sm8x gpu architectures. Attempting to run on a sm ",
        dprops->major,
        ".",
        dprops->minor,
        " gpu.");
    return false;
  }
  return true;
}

inline bool check_gpu_sm50_or_greater(sdp_params params, bool debug) {
  // Check that the gpu is capable of running flash attention
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm50 = dprops->major >= 5;
  if (!(is_sm50)) {
    TORCH_CHECK(
        !debug,
        "Mem Efficient Attention only supports sm5x or greater gpu architectures. Attempting to run on a sm ",
        dprops->major,
        ".",
        dprops->minor,
        " gpu.");
    return false;
  }
  return true;
}

inline bool use_flash_attention(sdp_params params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_CHECK(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif
  //  Define gate functions that determine if a flash kernel can be ran
  constexpr std::array<bool(*)(sdp_params, bool), 7> constraints {{
      check_runtime_disabled_flash,
      check_tensor_shapes,
      check_for_attn_weights,
      check_for_attn_mask,
      check_head_dim_size,
      check_gpu_sm75_or_greater,
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
  std::vector<std::function<bool(sdp_params, bool)>> constraints{
      check_gpu_sm50_or_greater,
      check_runtime_disabled_mem_efficient,
      check_for_attn_weights,
      check_tensor_shapes,
      check_for_attn_mask,
      check_for_seq_len_1_nested_tensor,
      check_for_non_zero_dropout};
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
  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  if (use_flash_attention(kernel_params, print_debug)) {
    return SDPBackend::flash_attention;
  }
  if (use_mem_efficient_attention(kernel_params, print_debug)) {
    return SDPBackend::efficient_attention;
  }
  if (ctx.userEnabledMathSDP()) {
    return SDPBackend::math;
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention or use_mem_efficient did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run use_flash_attention with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  use_mem_efficient_attention(kernel_params, print_debug);
  use_flash_attention(kernel_params, print_debug);
  return SDPBackend::error;
}

} // namespace sdp
