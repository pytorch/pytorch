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

enum class SDPBackend {flash_attention, math, error};

#define CHECK_DTYPE(tensor, types)                                    \
  if (std::find(types.begin(), types.end(), params.tensor.dtype()) == \
      types.end()) {                                                  \
    TORCH_CHECK(                                                      \
        debug,                                                        \
        #tensor,                                                      \
        " is not in the allowed dtypes. ",                            \
        #tensor,                                                      \
        " dtype: ",                                                   \
        params.tensor.dtype());                                       \
    return false;                                                     \
  }
template <typename dtype_vector>
inline bool check_tensor_dtype(
    sdp_params params,
    dtype_vector allowed_dtypes,
    bool debug) {
  CHECK_DTYPE(query, allowed_dtypes)
  CHECK_DTYPE(key, allowed_dtypes)
  CHECK_DTYPE(value, allowed_dtypes)
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

#define CHECK_DIM(tensor, n_dims)      \
  if (params.tensor.dim() != n_dims) { \
    TORCH_CHECK(                       \
        debug,                         \
        #tensor,                       \
        " is not a ",                  \
        #n_dims,                       \
        " dimensional tensor. ",       \
        #tensor,                       \
        " dim: ",                      \
        params.tensor.dim());          \
    return false;                      \
  }

inline bool check_tensor_shapes(sdp_params params, bool debug) {
  CHECK_DIM(query, 4)
  CHECK_DIM(key, 4)
  CHECK_DIM(value, 4)
  return true;
}

#define CHECK_HEAD_SIZE(tensor)                                              \
  if ((params.tensor.size(-1) % 8 != 0) && (params.tensor.size(-1) > 128)) { \
    TORCH_CHECK(                                                             \
        debug,                                                               \
        #tensor,                                                             \
        "'s last dimensions is not a multiple of 8 and last then 128. ",     \
        #tensor,                                                             \
        ".size(-1) = ",                                                      \
        params.tensor.size(-1));                                             \
    return false;                                                            \
  }

inline bool check_head_dim_size(
    sdp_params params,
    bool debug) {
  CHECK_HEAD_SIZE(query)
  CHECK_HEAD_SIZE(key)
  CHECK_HEAD_SIZE(value)
  return true;
}

inline bool check_runtime_disabled(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of flash sdp kernels
  if (!at::globalContext().userEnabledFlashSDP()) {
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
  #ifndef USE_FLASH_ATTENTION
    TORCH_CHECK(debug, "Torch was not compiled with flash attention.");
    return false;
  #endif
  // Constraints specific to flash attention
  static const std::vector<caffe2::ScalarType> flash_dtypes{at::kHalf, at::kBFloat16};

  //  Define gate functions that determine if a flash kernel can be ran
  std::vector<std::function<bool(sdp_params, bool)>> constraints{
      check_runtime_disabled,
      check_tensor_shapes,
      check_for_attn_weights,
      check_for_attn_mask,
      check_head_dim_size,
      check_gpu_sm75_or_greater};
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
  // 2. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP()){
       return SDPBackend::error;
  }
  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  if (use_flash_attention(kernel_params, !print_debug)) {
    return SDPBackend::flash_attention;
  }
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
