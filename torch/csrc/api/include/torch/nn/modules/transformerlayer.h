#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/options/transformerlayer.h>
#include <torch/nn/pimpl.h>

#include <torch/types.h>

#include <ostream>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerEncoderLayer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerEncoderLayer module.
/// See
/// https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoderLayer.html
/// to learn abouut the exact behavior of this encoder layer model
///
/// See the documentation for `torch::nn::TransformerEncoderLayer` class to
/// learn what constructor arguments are supported for this encoder layer model
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512,
/// 8).dropout(0.1));
/// ```
// NOLINTNEXTLINE(bugprone-exception-escape)
class TORCH_API TransformerEncoderLayerImpl
    : public Cloneable<TransformerEncoderLayerImpl> {
 public:
  TransformerEncoderLayerImpl(int64_t d_model, int64_t nhead)
      : TransformerEncoderLayerImpl(
            TransformerEncoderLayerOptions(d_model, nhead)) {}
  explicit TransformerEncoderLayerImpl(
      const TransformerEncoderLayerOptions& options_);

  Tensor forward(
      const Tensor& src,
      const Tensor& src_mask = {},
      const Tensor& src_key_padding_mask = {});

  void reset() override;

  void reset_parameters();

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})

 public:
  /// options with which this `TransformerEncoderLayer` was constructed
  TransformerEncoderLayerOptions options;

  /// self attention
  MultiheadAttention self_attn = nullptr;

  /// feedforward first linear layer
  Linear linear1 = nullptr;

  /// feedforward dropout layer
  Dropout dropout = nullptr;

  /// feedforward second linear layer
  Linear linear2 = nullptr;

  /// pre feedforward, normalization layer
  LayerNorm norm1 = nullptr;
  /// post feedfastward, normalization layer
  LayerNorm norm2 = nullptr;

  /// pre feedfastward, dropout layer
  Dropout dropout1 = nullptr;
  /// post feedfastward, dropout layer
  Dropout dropout2 = nullptr;
};

/// A `ModuleHolder` subclass for `TransformerEncoderLayerImpl``.
/// See the documentation for `TransformerEncoderLayerImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerEncoderLayer`
/// with `torch::nn::TransformerEncoderLayerOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(TransformerEncoderLayer);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerDecoderLayer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerDecoderLayer is made up of self-attn, multi-head-attn and
/// feedforward network. This standard decoder layer is based on the paper
/// "Attention Is All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
/// Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia
/// Polosukhin. 2017. Attention is all you need. In Advances in Neural
/// Information Processing Systems, pages 6000-6010. Users may modify or
/// implement in a different way during application. See
/// https://pytorch.org/docs/master/nn.html#transformer-layers to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::TransformerDecoderLayerOptions` class
/// to learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// TransformerDecoderLayer model(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.2));
/// ```
// NOLINTNEXTLINE(bugprone-exception-escape)
class TORCH_API TransformerDecoderLayerImpl
    : public Cloneable<TransformerDecoderLayerImpl> {
 public:
  TransformerDecoderLayerImpl(int64_t d_model, int64_t nhead)
      : TransformerDecoderLayerImpl(
            TransformerDecoderLayerOptions(d_model, nhead)) {}
  explicit TransformerDecoderLayerImpl(
      const TransformerDecoderLayerOptions& options_);

  void reset() override;

  void reset_parameters();

  /// Pass the inputs (and mask) through the decoder layer.
  /// Args:
  ///       tgt: the sequence to the decoder layer (required).
  ///       memory: the sequence from the last layer of the encoder (required).
  ///       tgt_mask: the mask for the tgt sequence (optional).
  ///       memory_mask: the mask for the memory sequence (optional).
  ///       tgt_key_padding_mask: the mask for the tgt keys per batch
  ///       (optional). memory_key_padding_mask: the mask for the memory keys
  ///       per batch (optional).
  Tensor forward(
      Tensor tgt,
      const Tensor& memory,
      const Tensor& tgt_mask = {},
      const Tensor& memory_mask = {},
      const Tensor& tgt_key_padding_mask = {},
      const Tensor& memory_key_padding_mask = {});

  /// The options used to configure this module.
  TransformerDecoderLayerOptions options;

  /// self attention
  MultiheadAttention self_attn{nullptr};

  /// Dropout, post self attention
  Dropout dropout1{nullptr};

  /// Normalization, post self attention
  LayerNorm norm1{nullptr};

  /// Multi-headed attention
  MultiheadAttention multihead_attn{nullptr};

  /// Dropout, post multi-headed attention
  Dropout dropout2{nullptr};

  /// Normalization, post multi-headed attention
  LayerNorm norm2{nullptr};

  /// Feed forward first linear layer
  Linear linear1{nullptr};

  /// Feed forward dropout layer
  Dropout dropout{nullptr};

  /// Feed forward second linear layer
  Linear linear2{nullptr};

  /// Dropout, post feed forward
  Dropout dropout3{nullptr};

  /// Normalization, post feed forward
  LayerNorm norm3{nullptr};

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {2, AnyValue(Tensor())},
      {3, AnyValue(Tensor())},
      {4, AnyValue(Tensor())},
      {5, AnyValue(Tensor())})

  /// Apply activation based on configuration
  Tensor activation(const Tensor& input);
};

/// A `ModuleHolder` subclass for `TransformerDecoderLayerImpl`.
/// See the documentation for `TransformerDecoderLayerImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerDecoderLayer`
/// with `torch::nn::TransformerDecoderLayerOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(TransformerDecoderLayer);

} // namespace nn
} // namespace torch
