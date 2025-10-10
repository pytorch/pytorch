#pragma once

#include <c10/util/irange.h>
#include <torch/types.h>

#include <utility>

namespace torch::nn::utils::rnn {

inline Tensor invert_permutation(const Tensor& permutation) {
  if (!permutation.defined()) {
    return torch::Tensor();
  }
  Tensor output =
      torch::empty_like(permutation, torch::MemoryFormat::Contiguous);
  output.scatter_(
      0,
      permutation,
      torch::arange(0, permutation.numel(), permutation.device()));
  return output;
}

/// Holds the data and list of `batch_sizes` of a packed sequence.
///
/// All RNN modules accept packed sequences as inputs.
///
/// Note:
///     Instances of this class should never be created manually. They are meant
///     to be instantiated by functions like `pack_padded_sequence`.
///
///     Batch sizes represent the number elements at each sequence step in
///     the batch, not the varying sequence lengths passed to
///     `pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
///     the :class:`PackedSequence` would contain data ``axbc`` with
///     ``batch_sizes=[2,1,1]``.
///
/// Attributes:
///     data (Tensor): Tensor containing packed sequence
///     batch_sizes (Tensor): Tensor of integers holding
///         information about the batch size at each sequence step
///     sorted_indices (Tensor, optional): Tensor of integers holding how this
///         :class:`PackedSequence` is constructed from sequences.
///     unsorted_indices (Tensor, optional): Tensor of integers holding how this
///         to recover the original sequences with correct order.
///
/// .. note::
///     `data` can be on arbitrary device and of arbitrary dtype.
///     `sorted_indices` and `unsorted_indices` must be ``torch::kInt64``
///     tensors on the same device as `data`.
///
///     However, `batch_sizes` should always be a CPU ``torch::kInt64`` tensor.
///
///     This invariant is maintained throughout `PackedSequence` class,
///     and all functions that construct a `PackedSequence` in libtorch
///     (i.e., they only pass in tensors conforming to this constraint).
class PackedSequence {
 public:
  explicit PackedSequence(
      Tensor data,
      Tensor batch_sizes,
      Tensor sorted_indices = {},
      Tensor unsorted_indices = {}) {
    // NB: if unsorted_indices is provided, it should be the inverse permutation
    // to sorted_indices. Don't assert it here because the PackedSequence ctor
    // should only be used internally.
    if (!unsorted_indices.defined()) {
      unsorted_indices = invert_permutation(sorted_indices);
    }
    TORCH_CHECK(
        batch_sizes.device().type() == kCPU,
        "batch_sizes should always be on CPU. "
        "Instances of PackedSequence should never be created manually. "
        "They should be instantiated by functions like pack_sequence "
        "and pack_padded_sequences in nn::utils::rnn. "
        "https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence");
    data_ = std::move(data);
    batch_sizes_ = std::move(batch_sizes);
    sorted_indices_ = std::move(sorted_indices);
    unsorted_indices_ = std::move(unsorted_indices);
  }

  const Tensor& data() const {
    return data_;
  }

  const Tensor& batch_sizes() const {
    return batch_sizes_;
  }

  const Tensor& sorted_indices() const {
    return sorted_indices_;
  }

  const Tensor& unsorted_indices() const {
    return unsorted_indices_;
  }

  PackedSequence pin_memory() const {
    // Why not convert `batch_sizes`?
    // See NOTE [ device and dtype of a PackedSequence ]
    return PackedSequence(
        data_.pin_memory(),
        batch_sizes_,
        sorted_indices_.defined() ? sorted_indices_.pin_memory() : Tensor(),
        unsorted_indices_.defined() ? unsorted_indices_.pin_memory()
                                    : Tensor());
  }

  PackedSequence to(TensorOptions options) const {
    // Performs dtype and/or device conversion on `data_`.
    //
    // If the ``data_`` Tensor already has the correct `torch::Dtype`
    // and `torch::Device`, then ``self`` is returned.
    // Otherwise, returns a copy with the desired configuration.

    // Why not convert `batch_sizes`?
    // See NOTE [ device and dtype of a PackedSequence ]
    Tensor data = data_.to(options);
    if (data.is_same(data_)) {
      return *this;
    } else {
      // Does not forward device or dtype args, device is set from data.device()
      Tensor sorted_indices = sorted_indices_.defined()
          ? sorted_indices_.to(
                options.device(data.device()).dtype(sorted_indices_.dtype()))
          : Tensor();
      Tensor unsorted_indices = unsorted_indices_.defined()
          ? unsorted_indices_.to(
                options.device(data.device()).dtype(unsorted_indices_.dtype()))
          : Tensor();
      return PackedSequence(
          std::move(data),
          batch_sizes_,
          std::move(sorted_indices),
          std::move(unsorted_indices));
    }
  }

  PackedSequence cuda() const {
    return to(kCUDA);
  }

  PackedSequence cpu() const {
    return to(kCPU);
  }

  /// Returns true if `data_` stored on a gpu
  bool is_cuda() const {
    return data_.is_cuda();
  }

  /// Returns true if `data_` stored on in pinned memory
  bool is_pinned() const {
    return data_.is_pinned();
  }

 private:
  Tensor data_;
  Tensor batch_sizes_;
  Tensor sorted_indices_;
  Tensor unsorted_indices_;
};

/// Packs a Tensor containing padded sequences of variable length.
///
/// `input` can be of size ``T x B x *`` where `T` is the length of the
/// longest sequence (equal to ``lengths[0]``), ``B`` is the batch size, and
/// ``*`` is any number of dimensions (including 0). If ``batch_first`` is
/// ``true``, ``B x T x *`` `input` is expected.
///
/// For unsorted sequences, use `enforce_sorted = false`. If `enforce_sorted` is
/// ``true``, the sequences should be sorted by length in a decreasing order,
/// i.e.
/// ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
/// shortest one.
///
/// Note:
///     This function accepts any input that has at least two dimensions. You
///     can apply it to pack the labels, and use the output of the RNN with
///     them to compute the loss directly. A Tensor can be retrieved from
///     a `PackedSequence` object by calling its ``.data()`` function.
///
/// Arguments:
///     input (Tensor): padded batch of variable length sequences.
///     lengths (Tensor): list of sequences lengths of each batch element.
///     batch_first (bool, optional): if ``true``, the input is expected in ``B
///     x T x *``
///         format. Default: ``false``.
///     enforce_sorted (bool, optional): if ``true``, the input is expected to
///         contain sequences sorted by length in a decreasing order. If
///         ``false``, this condition is not checked. Default: ``true``.
///
/// Returns:
///     a `PackedSequence` object
inline PackedSequence pack_padded_sequence(
    Tensor input,
    Tensor lengths,
    bool batch_first = false,
    bool enforce_sorted = true) {
  lengths = lengths.to(kInt64);
  Tensor sorted_indices;
  if (enforce_sorted) {
    sorted_indices = Tensor();
  } else {
    std::tie(lengths, sorted_indices) =
        torch::sort(lengths, /*dim=*/-1, /*descending=*/true);
    sorted_indices = sorted_indices.to(input.device());
    int64_t batch_dim = batch_first ? 0 : 1;
    input = input.index_select(batch_dim, sorted_indices);
  }

  auto [data, batch_sizes] =
      torch::_pack_padded_sequence(input, lengths, batch_first);
  return PackedSequence(
      std::move(data), std::move(batch_sizes), std::move(sorted_indices), {});
}

/// Pads a packed batch of variable length sequences.
///
/// It is an inverse operation to `pack_padded_sequence`.
///
/// The returned Tensor's data will be of size ``T x B x *``, where `T` is the
/// length of the longest sequence and `B` is the batch size. If ``batch_first``
/// is true, the data will be transposed into ``B x T x *`` format.
///
/// Batch elements will be ordered decreasingly by their length.
///
/// Arguments:
///     sequence (PackedSequence): batch to pad
///     batch_first (bool, optional): if ``true``, the output will be in ``B x T
///     x *``
///         format.
///     padding_value (double, optional): values for padded elements.
///     total_length (int64_t, optional): if specified, the output will be
///     padded to
///         have length `total_length`. This method will throw error
///         if `total_length` is less than the max sequence length in
///         `sequence`.
///
/// Returns:
///     Tuple of Tensor containing the padded sequence, and a Tensor
///     containing the list of lengths of each sequence in the batch.
inline std::tuple<Tensor, Tensor> pad_packed_sequence(
    const PackedSequence& sequence,
    bool batch_first = false,
    double padding_value = 0.0,
    std::optional<int64_t> total_length = std::nullopt) {
  int64_t max_seq_length = sequence.batch_sizes().size(0);
  if (total_length.has_value()) {
    int64_t total_length_val = total_length.value();
    TORCH_CHECK(
        total_length_val >= max_seq_length,
        "Expected total_length to be at least the length "
        "of the longest sequence in input, but got "
        "total_length=",
        total_length_val,
        " and max sequence length being ",
        max_seq_length);
    max_seq_length = total_length_val;
  }
  auto [padded_output, lengths] = torch::_pad_packed_sequence(
      sequence.data(),
      sequence.batch_sizes(),
      batch_first,
      padding_value,
      max_seq_length);
  const Tensor& unsorted_indices = sequence.unsorted_indices();
  if (unsorted_indices.defined()) {
    int64_t batch_dim = batch_first ? 0 : 1;
    return std::make_tuple(
        padded_output.index_select(batch_dim, unsorted_indices),
        lengths.index({unsorted_indices.cpu()}));
  }
  return std::make_tuple(padded_output, lengths);
}

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
///     batch_first (bool, optional): output will be in ``B x T x *`` if true,
///     or in
///         ``T x B x *`` otherwise
///     padding_value (double, optional): value for padded elements. Default: 0.
///     padding_side (str, optional): the side to pad the sequences on. Default:
///         "right".
///
/// Returns:
///     Tensor of size ``T x B x *`` if `batch_first` is ``false``.
///     Tensor of size ``B x T x *`` otherwise
inline Tensor pad_sequence(
    ArrayRef<Tensor> sequences,
    bool batch_first = false,
    double padding_value = 0,
    std::string_view padding_side = "right") {
  return at::pad_sequence(sequences, batch_first, padding_value, padding_side);
}

/// Packs a list of variable length Tensors
///
/// ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
/// the length of a sequence and `*` is any number of trailing dimensions,
/// including zero.
///
/// For unsorted sequences, use `enforce_sorted = false`. If ``enforce_sorted``
/// is ``true``, the sequences should be sorted in the order of decreasing
/// length.
///
///
/// Arguments:
///     sequences (torch::ArrayRef<Tensor>): A list of sequences of decreasing
///     length. enforce_sorted (bool, optional): if ``true``, checks that the
///     input
///         contains sequences sorted by length in a decreasing order. If
///         ``false``, this condition is not checked. Default: ``true``.
///
/// Returns:
///     a `PackedSequence` object
inline PackedSequence pack_sequence(
    ArrayRef<Tensor> sequences,
    bool enforce_sorted = true) {
  Tensor lengths = torch::empty({(int64_t)sequences.size()}, kInt64);
  for (const auto i : c10::irange(sequences.size())) {
    lengths[static_cast<int64_t>(i)] = sequences[i].size(0);
  }
  return pack_padded_sequence(
      at::pad_sequence(sequences),
      std::move(lengths),
      /*batch_first=*/false,
      /*enforce_sorted=*/enforce_sorted);
}

} // namespace torch::nn::utils::rnn
