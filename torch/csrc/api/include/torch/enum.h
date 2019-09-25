#pragma once

#include <c10/util/variant.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#define TORCH_ENUM_DECLARE(name) \
namespace torch { \
namespace enumtype { \
  struct name {}; \
} \
TORCH_API extern const enumtype::name k##name; \
}

#define TORCH_ENUM_DEFINE(name) \
namespace torch { \
const enumtype::name k##name; \
}

TORCH_ENUM_DECLARE(Linear)
TORCH_ENUM_DECLARE(Conv1D)
TORCH_ENUM_DECLARE(Conv2D)
TORCH_ENUM_DECLARE(Conv3D)
TORCH_ENUM_DECLARE(ConvTranspose1D)
TORCH_ENUM_DECLARE(ConvTranspose2D)
TORCH_ENUM_DECLARE(ConvTranspose3D)
TORCH_ENUM_DECLARE(Sigmoid)
TORCH_ENUM_DECLARE(Tanh)
TORCH_ENUM_DECLARE(ReLU)
TORCH_ENUM_DECLARE(LeakyReLU)
TORCH_ENUM_DECLARE(FanIn)
TORCH_ENUM_DECLARE(FanOut)

namespace torch {

/// Variable of `Nonlinearity` type can take one of the following values:
/// - `torch::kLinear`
/// - `torch::kConv1D`
/// - `torch::kConv2D`
/// - `torch::kConv3D`
/// - `torch::kConvTranspose1D`
/// - `torch::kConvTranspose2D`
/// - `torch::kConvTranspose3D`
/// - `torch::kSigmoid`
/// - `torch::kTanh`
/// - `torch::kReLU`
/// - `torch::kLeakyReLU`
using Nonlinearity = c10::variant<
  enumtype::Linear,
  enumtype::Conv1D,
  enumtype::Conv2D,
  enumtype::Conv3D,
  enumtype::ConvTranspose1D,
  enumtype::ConvTranspose2D,
  enumtype::ConvTranspose3D,
  enumtype::Sigmoid,
  enumtype::Tanh,
  enumtype::ReLU,
  enumtype::LeakyReLU
>;

/// Variable of `FanMode` type can take one of the following values:
/// - `torch::kFanIn`
/// - `torch::kFanOut`
using FanMode = c10::variant<
  enumtype::FanIn,
  enumtype::FanOut
>;

} // namespace torch
