#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace utils {
namespace rnn {

/// Pad a list of variable length Tensors with ``padding_value``
///
/// ``pad_sequence`` stacks a list of Tensors along a new dimension,
/// and pads them to equal length. For example, if the input is list of
/// sequences with size ``L x *`` and if batch_first is false, and ``T x B x *``
/// otherwise.
///
/// `B` is batch size. It is equal to the number of elements in ``sequences``.
/// `T` is length of the longest sequence.
/// `L` is length of the sequence.
/// `*` is any number of trailing dimensions, including none.
///
/// Note:
///     This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
///     where `T` is the length of the longest sequence. This function assumes
///     trailing dimensions and type of all the Tensors in sequences are same.
///
/// Arguments:
///     sequences (torch::ArrayRef<Tensor>): list of variable length sequences.
///     batch_first (bool, optional): output will be in ``B x T x *`` if true, or in
///         ``T x B x *`` otherwise
///     padding_value (double, optional): value for padded elements. Default: 0.
///
/// Returns:
///     Tensor of size ``T x B x *`` if `batch_first` is ``false``.
///     Tensor of size ``B x T x *`` otherwise
inline Tensor pad_sequence(
    ArrayRef<Tensor> sequences,
    bool batch_first = false,
    double padding_value = 0) {
  // assuming trailing dimensions and type of all the Tensors
  // in sequences are same and fetching those from sequences[0]
  auto max_size = sequences[0].sizes();
  auto trailing_dims = max_size.slice(1);
  auto max_len = std::max_element(
    sequences.begin(),
    sequences.end(),
    [](const Tensor& a, const Tensor& b) {
      return a.size(0) < b.size(0);
    }
  )->size(0);

  std::vector<int64_t> out_dims;
  if (batch_first) {
    out_dims = {(int64_t)sequences.size(), max_len};
  } else {
    out_dims = {max_len, (int64_t)sequences.size()};
  }
  out_dims.insert(out_dims.end(), trailing_dims.begin(), trailing_dims.end());

  auto out_tensor = torch::full({out_dims}, padding_value, sequences[0].options());
  for (size_t i = 0; i < sequences.size(); i++) {
    auto tensor = sequences[i];
    int64_t length = tensor.size(0);
    // use index notation to prevent duplicate references to the tensor
    if (batch_first) {
      out_tensor.select(0, i).narrow(0, 0, length).copy_(tensor);
    } else {
      out_tensor.narrow(0, 0, length).select(1, i).copy_(tensor);
    }
  }
  return out_tensor;
}

} // namespace rnn
} // namespace utils
} // namespace nn
} // namespace torch
