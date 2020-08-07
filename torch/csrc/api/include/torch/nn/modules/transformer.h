#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/options/transformer.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/common.h>

#include <torch/types.h>

#include <ostream>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerEncoderLayer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerEncoderLayer module.
/// See https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoderLayer.html to
/// learn abouut the exact behavior of this encoder layer model
///
/// See the documentation for `torch::nn::TransformerEncoderLayer` class to learn what
/// constructor arguments are supported for this encoder layer model
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512, 8).dropout(0.1));
/// ```
class TORCH_API TransformerEncoderLayerImpl : public Cloneable<TransformerEncoderLayerImpl> {

  public:
    explicit TransformerEncoderLayerImpl(const TransformerEncoderLayerOptions& options_);

    Tensor forward(
      const Tensor& src,
      const Tensor& src_mask = {},
      const Tensor& src_key_padding_mask = {});

    void reset() override;

    void reset_parameters();

  protected:
    FORWARD_HAS_DEFAULT_ARGS(
      {1, AnyValue(Tensor())},
      {2, AnyValue(Tensor())})

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
    LayerNorm norm2 = nullptr;;

    /// pre feedfastward, dropout layer
    Dropout dropout1 = nullptr;
    /// post feedfastward, dropout layer
    Dropout dropout2 = nullptr;
};

/// A `ModuleHolder` subclass for `TransformerEncoderLayerImpl``.
/// See the documentation for `TransformerEncoderLayerImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerEncoderLayer` with
/// `torch::nn::TransformerEncoderLayerOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TransformerEncoderLayer);

} // namespace nn
} // namespace torch
