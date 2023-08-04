#pragma once

#include <ATen/Context.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sdp {
struct sdp_params {
  const at::Tensor& query;
  const at::Tensor& key;
  const at::Tensor& value;
  const c10::optional<at::Tensor> attn_mask;
  double dropout;
  bool is_causal;
};

bool check_for_seq_len_1_nested_tensor(sdp_params params, bool debug);
SDPBackend select_sdp_backend(sdp_params kernel_params);

} // namespace sdp
