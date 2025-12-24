#pragma once

#include <ATen/Context.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/flash_attn/utils.h>
#include <ATen/xpu/XPUContext.h>

namespace sdp {

C10_EXPORT bool is_flash_attention_available();
C10_EXPORT bool can_use_flash_attention(sdp_params const& params, bool debug);
C10_EXPORT bool check_flash_attention_hardware_support(
    sdp_params const& params,
    bool debug);

} // namespace sdp
