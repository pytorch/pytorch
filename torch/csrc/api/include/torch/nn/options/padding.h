#pragma once

#include <c10/util/variant.h>
#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a pad functional.
struct TORCH_API PadOptions {
  PadOptions(std::vector<int64_t> pad);

  /// m-elements tuple, where m/2 <= input dimensions and m is even.
  TORCH_ARG(std::vector<int64_t>, pad);

  /// "constant", "reflect", "replicate" or "circular". Default: "constant"
  typedef c10::variant<
    enumtype::kConstant,
    enumtype::kReflect,
    enumtype::kReplicate,
    enumtype::kCircular> mode_t;
  TORCH_ARG(mode_t, mode) = torch::kConstant;

  /// fill value for "constant" padding. Default: 0
  TORCH_ARG(double, value) = 0;
};

} // namespace nn
} // namespace torch
