#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/Config.h"

namespace at { namespace native {

Tensor pad_sequence(TensorList tensors, bool batch_first, Scalar padding_value) {
  // assuming trailing dimensions and type of all the Tensors
  // in sequences are same and fetching those from tensors[0]
  if (tensors.size() == 0) {
    AT_ERROR("pad_sequence: input tensor sequence must have at least one tensor.");
  }
  auto max_size = tensors[0].sizes();
  std::vector<int64_t> output_size(max_size.size() + 1);
  std::copy(max_size.begin() + 1, max_size.end(), output_size.begin() + 2);

  // Find the max sequence length
  int64_t max_len = 0;
  for (auto & t : tensors) {
    if (t.size(0) > max_len) {
      max_len = t.size(0);
    }
  }

  if (batch_first) {
    output_size[0] = tensors.size();
    output_size[1] = max_len;
  } else {
    output_size[0] = max_len;
    output_size[1] = tensors.size();
  }
  
  auto result = tensors[0].type().tensor(output_size).fill_(padding_value);
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    auto i = it - tensors.begin();
    auto seq_len = it->size(0);
    Tensor view;
    if (batch_first) {
      // result[i, :seq_len, ...]
      view = result.narrow(0, i, 1).narrow(1, 0, seq_len).squeeze(0);
    } else {
      // result[:seq_len, i, ...]
      view = result.narrow(0, 0, seq_len).narrow(1, i, 1).squeeze(1);
    }
    view.copy_(*it);
  }
  return result;
}

}}
