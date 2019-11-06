#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/options/upsampling.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <ostream>

namespace torch {
namespace nn {

/// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D
/// (volumetric) data.
///
/// See https://pytorch.org/docs/stable/nn.html#Upsample to learn more
/// about the exact semantics of this module.
class TORCH_API UpsampleImpl : public Cloneable<UpsampleImpl> {
 public:
  explicit UpsampleImpl(const UpsampleOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `Upsample` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this `Module` was constructed.
  UpsampleOptions options;
};

/// A `ModuleHolder` subclass for `UpsampleImpl`.
/// See the documentation for `UpsampleImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(Upsample);

} // namespace nn
} // namespace torch
