#pragma once

#include <c10/util/variant.h>
#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional ReflectionPad module.
template <size_t D>
struct TORCH_API ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// If it is `int`, uses the same padding in all boundaries.
  /// If it is a 2-`tuple` (for ReflectionPad1d), uses (padding_left, padding_right).
  /// If it is a 4-`tuple` (for ReflectionPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  /// If it is a 6-`tuple` (for ReflectionPad3d), uses (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).

  TORCH_ARG(ExpandingArray<D*2>, padding);
};

/// `ReflectionPadOptions` specialized for the `ReflectionPad1d` module.
///
/// Example:
/// ```
/// ReflectionPad1d model(ReflectionPad1dOptions({3, 1}));
/// ```
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

/// `ReflectionPadOptions` specialized for the `ReflectionPad2d` module.
///
/// Example:
/// ```
/// ReflectionPad2d model(ReflectionPad2dOptions({1, 1, 2, 0}));
/// ```
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

/// `ReflectionPadOptions` specialized for the `ReflectionPad3d` module.
///
/// Example:
/// ```
/// ReflectionPad3d model(ReflectionPad3dOptions({1, 1, 2, 0, 1, 1}));
/// ```
using ReflectionPad3dOptions = ReflectionPadOptions<3>;

// ============================================================================

/// Options for a `D`-dimensional ReplicationPad module.
template <size_t D>
struct TORCH_API ReplicationPadOptions {
  ReplicationPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ReplicationPad1d), uses (padding_left, padding_right).
  /// - If it is a 4-`tuple` (for ReplicationPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ReplicationPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D*2>, padding);
};

/// `ReplicationPadOptions` specialized for the `ReplicationPad1d` module.
///
/// Example:
/// ```
/// ReplicationPad1d model(ReplicationPad1dOptions({3, 1}));
/// ```
using ReplicationPad1dOptions = ReplicationPadOptions<1>;

/// `ReplicationPadOptions` specialized for the `ReplicationPad2d` module.
///
/// Example:
/// ```
/// ReplicationPad2d model(ReplicationPad2dOptions({1, 1, 2, 0}));
/// ```
using ReplicationPad2dOptions = ReplicationPadOptions<2>;

/// `ReplicationPadOptions` specialized for the `ReplicationPad3d` module.
///
/// Example:
/// ```
/// ReplicationPad3d model(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}));
/// ```
using ReplicationPad3dOptions = ReplicationPadOptions<3>;

// ============================================================================

/// Options for the `ZeroPad2d` module.
///
/// Example:
/// ```
/// ZeroPad2d model(ZeroPad2dOptions({1, 1, 2, 0}));
/// ```
struct TORCH_API ZeroPad2dOptions {
  ZeroPad2dOptions(ExpandingArray<4> padding) : padding_(padding) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 4-`tuple` (for ZeroPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  TORCH_ARG(ExpandingArray<4>, padding);
};

// ============================================================================

/// Options for a `D`-dimensional ConstantPad module.
template <size_t D>
struct TORCH_API ConstantPadOptions {
  ConstantPadOptions(ExpandingArray<D*2> padding, double value) : padding_(padding), value_(value) {}

  /// The size of the padding.
  /// - If it is `int`, uses the same padding in all boundaries.
  /// - If it is a 2-`tuple` (for ConstantPad1d), uses (padding_left, padding_right).
  /// - If it is a 4-`tuple` (for ConstantPad2d), uses (padding_left, padding_right, padding_top, padding_bottom).
  /// - If it is a 6-`tuple` (for ConstantPad3d), uses
  ///   (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).
  TORCH_ARG(ExpandingArray<D*2>, padding);

  /// Fill value for constant padding.
  TORCH_ARG(double, value);
};

/// `ConstantPadOptions` specialized for the `ConstantPad1d` module.
///
/// Example:
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1}, 3.5));
/// ```
using ConstantPad1dOptions = ConstantPadOptions<1>;

/// `ConstantPadOptions` specialized for the `ConstantPad2d` module.
///
/// Example:
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
/// ```
using ConstantPad2dOptions = ConstantPadOptions<2>;

/// `ConstantPadOptions` specialized for the `ConstantPad3d` module.
///
/// Example:
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
/// ```
using ConstantPad3dOptions = ConstantPadOptions<3>;

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::pad`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::pad(input, F::PadFuncOptions({1, 2, 2, 1, 1, 2}).mode(torch::kReplicate));
/// ```
struct TORCH_API PadFuncOptions {
  typedef c10::variant<
    enumtype::kConstant,
    enumtype::kReflect,
    enumtype::kReplicate,
    enumtype::kCircular> mode_t;

  PadFuncOptions(std::vector<int64_t> pad);

  /// m-elements tuple, where m/2 <= input dimensions and m is even.
  TORCH_ARG(std::vector<int64_t>, pad);

  /// "constant", "reflect", "replicate" or "circular". Default: "constant"
  TORCH_ARG(mode_t, mode) = torch::kConstant;

  /// fill value for "constant" padding. Default: 0
  TORCH_ARG(double, value) = 0;
};

} // namespace functional

} // namespace nn
} // namespace torch
