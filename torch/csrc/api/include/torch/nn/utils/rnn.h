#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace utils {
namespace rnn {

inline Tensor pad_sequence(std::vector<Tensor>& sequences, bool batch_first=false, float padding_value=0) {
  auto max_size = sequences[0].sizes();
  auto trailing_dims = std::vector<int64_t>(max_size.begin() + 1, max_size.end());

  auto max_len = (*std::max_element(
    sequences.begin(),
    sequences.end(),
    [](Tensor& a, Tensor& b) {return a.sizes()[0] < b.sizes()[0];}
  )).sizes()[0];

  std::vector<int64_t> out_dims;
  out_dims.reserve(sequences.size() + max_len + trailing_dims.size());

  if (batch_first) {
    out_dims.insert(out_dims.end(), sequences.size());
    out_dims.insert(out_dims.end(), max_len);
  } else {
    out_dims.insert(out_dims.end(), max_len);
    out_dims.insert(out_dims.end(), sequences.size());
  }

  out_dims.insert(out_dims.end(), trailing_dims.begin(), trailing_dims.end());

  auto out_tensor = torch::full(out_dims, padding_value, sequences[0].dtype());
  for(int i = 0; i < sequences.size(); i++) {
    int length = sequences[i].sizes()[0];
    if (batch_first) {
      out_tensor.select(0, i).narrow(0, 0, length).copy_(sequences[i]);
    } else {
      out_tensor.select(1, i).narrow(0, 0, length).copy_(sequences[i]);
    }
  }
  return out_tensor;
}

} // namespace rnn
} // namespace utils
} // namespace nn
} // namespace torch
