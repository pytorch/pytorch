#pragma once
#include <c10/metal/common.h>

template <typename index_t>
struct CTCLossParams {
  index_t BLANK;
  index_t max_input_length;
  index_t max_target_length;
  index_t batch_size;
  index_t tg_target_stride;
  index_t log_probs_time_stride;
  index_t log_probs_batch_stride;
  index_t log_probs_token_stride;
  index_t log_alpha_batch_stride;
  index_t log_alpha_time_stride;
  index_t log_alpha_target_stride;
};

template <typename index_t>
struct CTCLossBackwardCollectParams {
  index_t BLANK;
  index_t max_input_length;
  index_t max_target_length;
  index_t num_labels;
  index_t tg_target_stride;
  index_t log_probs_time_stride;
  index_t log_probs_batch_stride;
  index_t log_probs_token_stride;
  index_t log_alpha_beta_batch_stride;
  index_t log_alpha_beta_time_stride;
  index_t log_alpha_beta_target_stride;
  index_t grad_time_stride;
  index_t grad_batch_stride;
  index_t grad_token_stride;
  index_t grad_out_batch_stride;
  bool zero_infinity;
};

// Shared by LossOps.metal (Metal kernels) and LossOps.mm (dispatch).
// The binary layout must stay identical on both sides.
struct NLLParams {
  int64_t ignore_index; // int64 so any Python-level value compares exactly
  uint32_t N;
  uint32_t C;
  uint32_t reduction;
  uint32_t has_weight;
};
