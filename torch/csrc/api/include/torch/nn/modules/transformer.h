#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/options/transformer.h>
#include <torch/nn/pimpl.h>

#include <torch/types.h>

#include <ostream>

namespace torch::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transformer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A transformer model. User is able to modify the attributes as needed. The
/// architecture is based on the paper "Attention Is All You Need". Ashish
/// Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
/// Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need.
/// In Advances in Neural Information Processing Systems, pages 6000-6010.
///
/// See https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html to
/// learn about the exact behavior of this transformer model
///
/// See the documentation for `torch::nn::Transformer` class to learn what
/// constructor arguments are supported for this encoder layer model
///
/// Example:
/// ```
/// Transformer trans(TransformerOptions(512, 8));
/// ```
class TORCH_API TransformerImpl : public Cloneable<TransformerImpl> {
 public:
  explicit TransformerImpl(TransformerOptions options_);

  /// forward function for Transformer Module
  /// Args:
  ///   src: the sequence to the encoder (required).
  ///   tgt: the sequence to the decoder (required).
  ///   src_mask: the additive mask for the src sequence (optional).
  ///   tgt_mask: the additive mask for the tgt sequence (optional).
  ///   memory_mask: the additive mask for the encoder output (optional).
  ///   src_key_padding_mask: the ByteTensor mask for src keys per batch
  ///   (optional). tgt_key_padding_mask: the ByteTensor mask for tgt keys per
  ///   batch (optional). memory_key_padding_mask: the ByteTensor mask for
  ///   memory keys per batch (optional).
  ///
  /// Shape:
  ///   src: `(S, N, E)`
  ///   tgt: `(T, N, E)`
  ///   src_mask: `(S, S)`
  ///   tgt_mask: `(T, T)`
  ///   memory_mask: `(T, S)`
  ///   src_key_padding_mask: `(N, S)`
  ///   tgt_key_padding_mask: `(N, T)`
  ///   memory_key_padding_mask: `(N, S)`
  ///
  ///   Note:
  ///     [src/tgt/memory]_mask ensures that position i is allowed to attend the
  ///     unmasked positions. If a ByteTensor is provided, the non-zero
  ///     positions are not allowed to attend while the zero positions will be
  ///     unchanged. If a BoolTensor is provided, positions with `True` are not
  ///     allowed to attend while `False` values will be unchanged. If a
  ///     FloatTensor is provided, it will be added to the attention weight.
  ///
  ///     [src/tgt/memory]_key_padding_mask provides specified elements in the
  ///     key to be ignored by the attention. If a ByteTensor is provided, the
  ///     non-zero positions will be ignored while the zero positions will be
  ///     unchanged. If a BoolTensor is provided, the positions with the value
  ///     of `True` will be ignored while the position with the value of `False`
  ///     will be unchanged.
  ///
  ///   output: `(T, N, E)`
  ///
  ///   Note:
  ///     Due to the multi-head attention architecture in the transformer model,
  ///     the output sequence length of a transformer is same as the input
  ///     sequence (i.e. target) length of the decode.
  ///
  ///   where
  ///   S is the source sequence length,
  ///   T is the target sequence length,
  ///   N is the batch size,
  ///   E is the feature number.
  Tensor forward(
      const Tensor& src,
      const Tensor& tgt,
      const Tensor& src_mask = {},
      const Tensor& tgt_mask = {},
      const Tensor& memory_mask = {},
      const Tensor& src_key_padding_mask = {},
      const Tensor& tgt_key_padding_mask = {},
      const Tensor& memory_key_padding_mask = {});

  void reset() override;

  void reset_parameters();

  /// Generate a square mask for the sequence.
  /// The masked positions are filled with `-inf` in float type.
  /// Unmasked positions are filled with `0.0` in float type.
  /// Note:
  ///   1. This function will always return a CPU tensor.
  ///   2. This function requires the platform support IEEE754, since `-inf` is
  ///   guaranteed to
  ///      be valid only when IEEE754 is supported. If the platform doesn't
  ///      support IEEE754, this function will fill the mask with the smallest
  ///      float number instead of `-inf`, a one time warning will pop up as
  ///      well.
  static Tensor generate_square_subsequent_mask(int64_t sz);

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {2, AnyValue(Tensor())},
      {3, AnyValue(Tensor())},
      {4, AnyValue(Tensor())},
      {5, AnyValue(Tensor())},
      {6, AnyValue(Tensor())},
      {7, AnyValue(Tensor())})

 public:
  /// options with which this `Transformer` was constructed
  TransformerOptions options;

  /// encoder module
  AnyModule encoder;

  /// decoder module
  AnyModule decoder;
};

/// A `ModuleHolder` subclass for `TransformerImpl`.
/// See the documentation for `TransformerImpl` class to learn what
/// methods it provides, and examples of how to use `Transformer` with
/// `torch::nn::TransformerOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Transformer);

} // namespace torch::nn
