#pragma once

#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <functional>
#include <unordered_set>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

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

enum class SDPBackend { flash_attention, mem_eff_attention, math, error};

//  Define gate functions that determine if a fused kernel can be ran
inline bool check_gradients(sdp_params params, bool debug) {
  if (params.query.requires_grad() || params.key.requires_grad() ||
      params.value.requires_grad()) {
    TORCH_CHECK(
        debug,
        "Query, Key, or Value require's gradient. Query: ",
        params.query.requires_grad(),
        " Key: ",
        params.key.requires_grad(),
        " Value: ",
        params.value.requires_grad());
    return false;
  }
  return true;
}

inline bool compiled_with_fused_kernels(sdp_params params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_CHECK(debug, "Torch was not compiled with flash attention");
  return false;
#endif
  return true;
}

inline bool check_tensor_dtype(
    sdp_params params,
    std::vector<caffe2::ScalarType> allowed_dtypes,
    bool debug) {
  if (std::find(
          allowed_dtypes.begin(), allowed_dtypes.end(), params.query.dtype()) ==
      allowed_dtypes.end()) {
    TORCH_CHECK(
        debug,
        "Query is not in the allowed dtypes. Query dtype: ",
        params.query.dtype());
    return false;
  }
  if (std::find(
          allowed_dtypes.begin(), allowed_dtypes.end(), params.key.dtype()) ==
      allowed_dtypes.end()) {
    TORCH_CHECK(
        debug,
        "Key is not in the allowed dtypes. Key dtype: ",
        params.key.dtype());
    return false;
  }
  if (std::find(
          allowed_dtypes.begin(), allowed_dtypes.end(), params.value.dtype()) ==
      allowed_dtypes.end()) {
    TORCH_CHECK(
        debug,
        "Value is not in the allowed dtypes. Value dtype: ",
        params.value.dtype());
    return false;
  }
  return true;
}

inline bool check_for_attn_weights(sdp_params params, bool debug) {
  // This can be returned form flash attention but care is needed
  // to convert from flash_attn format to attn_weights
  if (params.need_attn_weights) {
    TORCH_CHECK(debug, "Flash Attention does not support need attn weights");
    return false;
  }
  return true;
}

inline bool check_for_attn_mask(sdp_params params, bool debug) {
  if (params.has_attn_mask) {
    TORCH_CHECK(debug, "Flash Attention does not support attention mask.");
    return false;
  }
  return true;
}

inline bool check_tensor_shapes(sdp_params params, bool debug) {
  if (params.query.dim() != 4) {
    TORCH_CHECK(
        debug,
        "Query is not a 4 dimensional tensor. Query dim: ",
        params.query.dim());
    return false;
  }
  if (params.key.dim() != 4) {
    TORCH_CHECK(
        debug,
        "Key is not a 4 dimensional tensor. Key dim: ",
        params.key.dim());
    return false;
  }
  if (params.value.dim() != 4) {
    TORCH_CHECK(
        debug,
        "Value is not a 4 dimensional tensor. Value dim: ",
        params.value.dim());
    return false;
  }
  return true;
}

// inline bool check_packed_projection_shape(sdp_params params, bool debug) {
//   if (params.query.dim() == 5 && params.query.size(2) != 3) {
//     TORCH_CHECK(
//         debug,
//         "Query is 5 dimensional(input is packed) but the 2nd dimension's shape does not equal 3. Query.size(2): ",
//         params.query.size(2));
//     return false;
//   }
//   if (params.key.dim() == 5 && params.key.size(2) != 3) {
//     TORCH_CHECK(
//         debug,
//         "Key is 5 dimensional(input is packed) but the 2nd dimension's shape does not equal 3. Key.size(2): ",
//         params.key.size(2));
//     return false;
//   }
//   if (params.value.dim() == 5 && params.value.size(2) != 3) {
//     TORCH_CHECK(
//         debug,
//         "Value is 5 dimensional(input is packed) but the 2nd dimension's shape does not equal 3. Value.size(2): ",
//         params.value.size(2));
//     return false;
//   }
//   return true;
// }

inline bool check_head_dim_size(
    sdp_params params,
    const std::unordered_set<int64_t> allowed_sizes,
    bool debug) {
  if (allowed_sizes.find(params.query.size(-1)) == allowed_sizes.end()) {
    TORCH_CHECK(
        debug,
        "Query's last dimension is not in the set of supported sizes. Query.size(-1): ",
        params.query.size(-1));
    return false;
  }
  if (allowed_sizes.find(params.key.size(-1)) == allowed_sizes.end()) {
    TORCH_CHECK(
        debug,
        "Key's last dimension is not in the set of supported sizes. Key.size(-1): ",
        params.key.size(-1));
    return false;
  }
  if (allowed_sizes.find(params.value.size(-1)) == allowed_sizes.end()) {
    TORCH_CHECK(
        debug,
        "Value's last dimension is not in the set of supported sizes. Value.size(-1): ",
        params.value.size(-1));
    return false;
  }
  return true;
}

inline bool check_runtime_disabled(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of fused sdp kernels
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledFusedSDP()) {
    TORCH_CHECK(debug, "Flash attention has been runtime disabled.");
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
        debug,
        "Flash attention only supports sm75 and sm8x gpu architectures. Attempting to run on a sm ",
        dprops->major,
        ".",
        dprops->minor,
        " gpu.");
    return false;
  }
  return true;
}

inline bool use_flash_attention(sdp_params params, bool debug) {
  // Constraints specific to flash attention
  static const std::unordered_set<int64_t> flash_embed_dim_sizes{
      16, 32, 64, 128};
  static const std::vector<caffe2::ScalarType> flash_dtypes{
      at::kHalf, at::kBFloat16};

  //
  std::vector<std::function<bool(sdp_params, bool)>> constraints{
      check_runtime_disabled,
      check_tensor_shapes,
      // check_packed_projection_shape,
      check_for_attn_weights,
      check_for_attn_mask,
      compiled_with_fused_kernels,
      check_gpu_sm75_or_greater};
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  // Check flash specific functions
  if (!check_head_dim_size(params, flash_embed_dim_sizes, debug)) {
    return false;
  }
  if (!check_tensor_dtype(params, flash_dtypes, debug)) {
    return false;
  }
  return true;
}

inline SDPBackend select_sdp_backend(sdp_params kernel_params) {
  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  if (use_flash_attention(kernel_params, !print_debug)) {
    return SDPBackend::flash_attention;
  }
  auto& ctx = at::globalContext();
  if (ctx.userEnabledMathSDP()){
       return SDPBackend::math;
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run use_flash_attention with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  use_flash_attention(kernel_params, !print_debug);
  return SDPBackend::error;
}

} // namespace sdp
