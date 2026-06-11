#pragma once
#include <c10/metal/common.h>

struct CrossEntropyParams {
  uint32_t vocab_size;
  uint32_t batch_size;
  int32_t ignore_index;
  float label_smoothing;
};
