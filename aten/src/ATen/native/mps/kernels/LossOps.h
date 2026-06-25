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
