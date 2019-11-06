#pragma once

#include <torch/nn/functional/activation.h>
#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor dropout(Tensor input, const DropoutOptions& options) {
  if (options.inplace()) {
    return torch::dropout_(input, options.p(), input->is_training());
  } else {
    return torch::dropout(input, options.p(), input->is_training()); 
  }
}

inline Tensor dropout2d(Tensor input, const Dropout2dOptions& options) {
  if (options.inplace()) {
    return torch::feature_dropout_(input, options.p(), input->is_training());
  } else {
    return torch::feature_dropout(input, options.p(), input->is_training()); 
  }
}

inline Tensor dropout3d(Tensor input, const Dropout3dOptions& options) {
  if (options.inplace()) {
    return torch::feature_dropout_(input, options.p(), input->is_training());
  } else {
    return torch::feature_dropout(input, options.p(), input->is_training()); 
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
