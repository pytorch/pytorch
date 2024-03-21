#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/options/upsampling.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/Export.h>

#include <cstddef>
#include <ostream>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsample ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D
/// (volumetric) data.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.Upsample to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::UpsampleOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Upsample
/// model(UpsampleOptions().scale_factor({3}).mode(torch::kLinear).align_corners(false));
/// ```
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
/// See the documentation for `UpsampleImpl` class to learn what methods it
/// provides, and examples of how to use `Upsample` with
/// `torch::nn::UpsampleOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Upsample);

} // namespace nn
} // namespace torch
