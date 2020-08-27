#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/options/transformercoder.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/modules/common.h>

#include <torch/types.h>

#include <ostream>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerEncoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerEncoder module.
/// See https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoder.html to
/// learn abouut the exact behavior of this encoder layer module.
///
/// See the documentation for `torch::nn::TransformerEncoder` class to learn what
/// constructor arguments are supported for this encoder module.
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512, 8).dropout(0.1));
//  TransformerEncoder encoder(TransformerEncoderOptions(encoderLayer, 6).norm(LayerNorm(LayerNormOptions({2}))));
/// ```
class TORCH_API TransformerEncoderImpl : public Cloneable<TransformerEncoderImpl> {

  public:
    explicit TransformerEncoderImpl(TransformerEncoderOptions options_);

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
    /// options with which this `TransformerEncoder` was constructed
    TransformerEncoderOptions options;

    /// module list that contains all the encoder layers
    ModuleList layers = nullptr;

    /// optional normalization module
    AnyModule norm;
};

/// A `ModuleHolder` subclass for `TransformerEncoderImpl``.
/// See the documentation for `TransformerEncoderImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerEncoder` with
/// `torch::nn::TransformerEncoderOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TransformerEncoder);

} // namespace nn
} // namespace torch
