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

template <class D>
inline bool check_tensor_dtype_fn(
    caffe2::TypeMeta tensor_dtype,
    D ref_dtype) {
  return tensor_dtype == ref_dtype;
}

template <class D, class... E>
inline bool check_tensor_dtype_fn(
    caffe2::TypeMeta tensor_dtype,
    D ref_dtype,
    E... other_ref_dtype) {
  return check_tensor_dtype_fn(tensor_dtype, ref_dtype)
    || check_tensor_dtype_fn(tensor_dtype, other_ref_dtype...);
}

inline bool check_gpu_sm75_or_greater(bool debug) {
  // Check that the gpu is capable of running flash attention
  auto dprops = at::cuda::getCurrentDeviceProperties();
  auto major = dprops->major;
  auto minor = dprops->minor;
  bool is_sm75 = major == 7 && minor == 5;
  bool is_sm8x = major == 8 && minor >= 0;
  if (debug) {
    TORCH_CHECK(check_gpu_sm75_or_greater(false),
        "Flash attention only supports sm75 and sm8x gpu architectures. Attempting to run on a sm ",
        major,
        ".",
        minor,
        " gpu.");
  }
  return (is_sm75 || is_sm8x);
}

inline bool use_flash_attention(sdp_params params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  TORCH_CHECK(!debug, "Torch was not compiled with flash attention.");
  return false;
#endif

  if (!at::globalContext().userEnabledFlashSDP()) {
    TORCH_CHECK(!debug, "Flash attention has been runtime disabled.");
    return false;
  }

  if (!check_gpu_sm75_or_greater(debug)) {
    return false;
  }

  if (!(params.query.dim() == params.key.dim() &&
        params.query.dim() == params.value.dim() &&
        params.query.dim() == 4)) {
    TORCH_CHECK(!debug, "Flash attention requires query, key and value to be 4 dimensional, but got ",
        params.query.dim(), ", ", params.key.dim(), ", ", params.value.dim(), " instead.");
    return false;
  }

  // This can be returned from flash attention but care is needed
  // to convert from flash_attn format to attn_weights
  if (params.need_attn_weights) {
    TORCH_CHECK(!debug, "Flash Attention does not support returning attn weights.");
    return false;
  }

  if (params.has_attn_mask) {
    TORCH_CHECK(!debug, "Flash Attention does not support attention mask.");
    return false;
  }

  if (!(params.query.size(-1) == params.key.size(-1) &&
        params.query.size(-1) == params.value.size(-1) &&
        params.query.size(-1) % 8 == 0 &&
        params.query.size(-1) <= 128)) {
    TORCH_CHECK(!debug, "Flash attention requires last dimension of inputs to be a multiple of 8 and less than or equal to 128.
        Got " params.query.size(-1), ", ", params.key.size(-1), ", ", params.value.size(-1), " instead.");
    return false;
  }

  if (!(
      check_tensor_dtype_fn<at::kHalf, at::kBFloat16>(params.query.dtype()) &&
      check_tensor_dtype_fn<at::kHalf, at::kBFloat16>(params.key.dtype()) &&
      check_tensor_dtype_fn<at::kHalf, at::kBFloat16>(params.value.dtype()))) {
    TORCH_CHECK(!debug, "Expected query, key and value to be of dtype float16 or bfloat16 but got ", params.query.dtype(),
        ", "
        params.key.dtype(), ", ", params.value.dtype(), " instead.");
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
