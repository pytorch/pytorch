#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/variant.h>

namespace torch {

namespace enumtype {
  // Nonlinearity
  struct Linear {};
  struct Conv1D {};
  struct Conv2D {};
  struct Conv3D {};
  struct ConvTranspose1D {};
  struct ConvTranspose2D {};
  struct ConvTranspose3D {};
  struct Sigmoid {};
  struct Tanh {};
  struct ReLU {};
  struct LeakyReLU {};

  // FanMode
  struct FanIn {};
  struct FanOut {};
} // namespace enumtype

TORCH_API extern const enumtype::Linear kLinear;
TORCH_API extern const enumtype::Conv1D kConv1D;
TORCH_API extern const enumtype::Conv2D kConv2D;
TORCH_API extern const enumtype::Conv3D kConv3D;
TORCH_API extern const enumtype::ConvTranspose1D kConvTranspose1D;
TORCH_API extern const enumtype::ConvTranspose2D kConvTranspose2D;
TORCH_API extern const enumtype::ConvTranspose3D kConvTranspose3D;
TORCH_API extern const enumtype::Sigmoid kSigmoid;
TORCH_API extern const enumtype::Tanh kTanh;
TORCH_API extern const enumtype::ReLU kReLU;
TORCH_API extern const enumtype::LeakyReLU kLeakyReLU;
TORCH_API extern const enumtype::FanIn kFanIn;
TORCH_API extern const enumtype::FanOut kFanOut;

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
