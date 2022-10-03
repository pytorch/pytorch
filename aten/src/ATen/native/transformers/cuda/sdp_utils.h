#pragma once

#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <ATen/cuda/CUDAContext.h>
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

enum class SDPBackend { flash_attention, math, error };

inline bool check_gpu_sm75_or_greater(bool debug) {
  // Check that the gpu is capable of running flash attention
  auto dprops = at::cuda::getCurrentDeviceProperties();
  auto major = dprops->major;
  auto minor = dprops->minor;
  bool is_sm75 = major == 7 && minor == 5;
  bool is_sm8x = major == 8 && minor >= 0;
  if (debug) {
    TORCH_CHECK(
        check_gpu_sm75_or_greater(false),
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

  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() &&
        query_dim == params.value.dim() &&
        query_dim == 4)) {
    TORCH_CHECK(
        !debug,
        "Flash attention requires query, key and value to be 4 dimensional, but got ",
        query_dim,
        ", ",
        params.key.dim(),
        ", ",
        params.value.dim(),
        " instead.");
    return false;
  }

  // TODO: This can be returned from flash attention but care is needed
  // to convert from flash_attn format to attn_weights
  if (params.need_attn_weights) {
    TORCH_CHECK(
        !debug, "Flash Attention does not support returning attn weights.");
    return false;
  }

  if (params.has_attn_mask) {
    TORCH_CHECK(!debug, "Flash Attention does not support attention mask.");
    return false;
  }

  auto query_size_last = params.query.size(-1);
  if (!(query_size_last == params.key.size(-1) &&
        query_size_last == params.value.size(-1) &&
        query_size_last % 8 == 0 &&
        query_size_last <= 128)) {
    TORCH_CHECK(
        !debug,
        "Flash attention requires last dimension of inputs to be a multiple of 8 and less than or equal to 128.",
        "Got ",
        query_size_last,
        ", ",
        params.key.size(-1),
        ", ",
        params.value.size(-1),
        " instead.");
    return false;
  }

  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (query_dtype == at::kHalf || query_dtype == at::kBFloat16))) {
    TORCH_CHECK(
        !debug,
        "Expected query, key and value to be of dtype float16 or bfloat16 but got ",
        params.query.dtype(),
        ", ",
        params.key.dtype(),
        ", and ",
        params.value.dtype(),
        " instead.");
    return false;
  }
  return true;
}
} // namespace sdp
