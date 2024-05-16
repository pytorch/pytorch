#pragma once

#include <ATen/Context.h>
#include <c10/macros/Macros.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/macros/Export.h>

namespace sdp {

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug);
SDPBackend select_sdp_backend(sdp_params const& kernel_params);
C10_EXPORT bool can_use_flash_attention(sdp_params const& params, bool debug);
C10_EXPORT bool can_use_mem_efficient_attention(sdp_params const& params, bool debug);
C10_EXPORT bool can_use_cudnn_attention(sdp_params const& params, bool debug);

} // namespace sdp
