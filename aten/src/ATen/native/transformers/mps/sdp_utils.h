#pragma once

#include <ATen/Context.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace at::native::mps {

inline bool prefill_attention_supports_head_dim(int64_t head_dim) {
  switch (head_dim) {
    case 32:
    case 64:
    case 72:
    case 80:
    case 96:
    case 128:
    case 256:
      return true;
    default:
      return false;
  }
}

bool can_use_flash_attention(const sdp::sdp_params& params, bool debug);
bool can_use_mem_efficient_attention(const sdp::sdp_params& params, bool debug);
sdp::SDPBackend select_sdp_backend(const sdp::sdp_params& params);

} // namespace at::native::mps
