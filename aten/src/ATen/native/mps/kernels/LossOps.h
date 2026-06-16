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

// Shared by LossOps.metal and LossOps.mm; layout must stay identical on both.
struct FusedLossParams {
  uint32_t numel; // filled by fused_loss_reduce
  uint32_t has_weight; // filled by fused_loss_reduce
  uint32_t aligned; // filled by fused_loss_reduce: all operands 4-elem aligned
  uint32_t reduction; // ATen: 1=Mean, 2=Sum (set by caller)
  float p0; // op scalar: beta/delta, clamp-lo; norm for fused_loss_bwd
  float p1; // op scalar: clamp-hi
  uint32_t flag; // op flag: is_huber, ...; scalar-grad for fused_loss_bwd
};
