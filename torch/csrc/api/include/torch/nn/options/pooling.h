#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional avgpool module.
template <size_t D>
struct AvgPoolOptions {
  AvgPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take an average over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size`
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;

  /// when True, will include the zero-padding in the averaging calculation
  TORCH_ARG(bool, count_include_pad) = true;

  /// if specified, it will be used as divisor, otherwise size of the pooling
  /// region will be used.

  TORCH_ARG(std::optional<int64_t>, divisor_override) = c10::nullopt;
};

/// `AvgPoolOptions` specialized for the `AvgPool1d` module.
///
/// Example:
/// ```
/// AvgPool1d model(AvgPool1dOptions(3).stride(2));
/// ```
using AvgPool1dOptions = AvgPoolOptions<1>;

/// `AvgPoolOptions` specialized for the `AvgPool2d` module.
///
/// Example:
/// ```
/// AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
/// ```
using AvgPool2dOptions = AvgPoolOptions<2>;

/// `AvgPoolOptions` specialized for the `AvgPool3d` module.
///
/// Example:
/// ```
/// AvgPool3d model(AvgPool3dOptions(5).stride(2));
/// ```
using AvgPool3dOptions = AvgPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::avg_pool1d`.
///
/// See the documentation for `torch::nn::AvgPool1dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));
/// ```
using AvgPool1dFuncOptions = AvgPool1dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::avg_pool2d`.
///
/// See the documentation for `torch::nn::AvgPool2dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));
/// ```
using AvgPool2dFuncOptions = AvgPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::avg_pool3d`.
///
/// See the documentation for `torch::nn::AvgPool3dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool3d(x, F::AvgPool3dFuncOptions(3).stride(2));
/// ```
using AvgPool3dFuncOptions = AvgPool3dOptions;
} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional maxpool module.
template <size_t D>
struct MaxPoolOptions {
  MaxPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take a max over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// a parameter that controls the stride of elements in the window
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;
};

/// `MaxPoolOptions` specialized for the `MaxPool1d` module.
///
/// Example:
/// ```
/// MaxPool1d model(MaxPool1dOptions(3).stride(2));
/// ```
using MaxPool1dOptions = MaxPoolOptions<1>;

/// `MaxPoolOptions` specialized for the `MaxPool2d` module.
///
/// Example:
/// ```
/// MaxPool2d model(MaxPool2dOptions({3, 2}).stride({2, 2}));
/// ```
using MaxPool2dOptions = MaxPoolOptions<2>;

/// `MaxPoolOptions` specialized for the `MaxPool3d` module.
///
/// Example:
/// ```
/// MaxPool3d model(MaxPool3dOptions(3).stride(2));
/// ```
using MaxPool3dOptions = MaxPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::max_pool1d` and
/// `torch::nn::functional::max_pool1d_with_indices`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool1d(x, F::MaxPool1dFuncOptions(3).stride(2));
/// ```
using MaxPool1dFuncOptions = MaxPool1dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::max_pool2d` and
/// `torch::nn::functional::max_pool2d_with_indices`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));
/// ```
using MaxPool2dFuncOptions = MaxPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::max_pool3d` and
/// `torch::nn::functional::max_pool3d_with_indices`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool3d(x, F::MaxPool3dFuncOptions(3).stride(2));
/// ```
using MaxPool3dFuncOptions = MaxPool3dOptions;
} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional adaptive maxpool module.
template <typename output_size_t>
struct AdaptiveMaxPoolOptions {
  AdaptiveMaxPoolOptions(output_size_t output_size)
      : output_size_(output_size) {}

  /// the target output size
  TORCH_ARG(output_size_t, output_size);
};

/// `AdaptiveMaxPoolOptions` specialized for the `AdaptiveMaxPool1d` module.
///
/// Example:
/// ```
/// AdaptiveMaxPool1d model(AdaptiveMaxPool1dOptions(3));
/// ```
using AdaptiveMaxPool1dOptions = AdaptiveMaxPoolOptions<ExpandingArray<1>>;

/// `AdaptiveMaxPoolOptions` specialized for the `AdaptiveMaxPool2d` module.
///
/// Example:
/// ```
/// AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
/// ```
using AdaptiveMaxPool2dOptions =
    AdaptiveMaxPoolOptions<ExpandingArrayWithOptionalElem<2>>;

/// `AdaptiveMaxPoolOptions` specialized for the `AdaptiveMaxPool3d` module.
///
/// Example:
/// ```
/// AdaptiveMaxPool3d model(AdaptiveMaxPool3dOptions(3));
/// ```
using AdaptiveMaxPool3dOptions =
    AdaptiveMaxPoolOptions<ExpandingArrayWithOptionalElem<3>>;

namespace functional {
/// Options for `torch::nn::functional::adaptive_max_pool1d` and
/// `torch::nn::functional::adaptive_max_pool1d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool1d(x, F::AdaptiveMaxPool1dFuncOptions(3));
/// ```
using AdaptiveMaxPool1dFuncOptions = AdaptiveMaxPool1dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::adaptive_max_pool2d` and
/// `torch::nn::functional::adaptive_max_pool2d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool2d(x, F::AdaptiveMaxPool2dFuncOptions(3));
/// ```
using AdaptiveMaxPool2dFuncOptions = AdaptiveMaxPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::adaptive_max_pool3d` and
/// `torch::nn::functional::adaptive_max_pool3d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool3d(x, F::AdaptiveMaxPool3dFuncOptions(3));
/// ```
using AdaptiveMaxPool3dFuncOptions = AdaptiveMaxPool3dOptions;
} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional adaptive avgpool module.
template <typename output_size_t>
struct AdaptiveAvgPoolOptions {
  AdaptiveAvgPoolOptions(output_size_t output_size)
      : output_size_(output_size) {}

  /// the target output size
  TORCH_ARG(output_size_t, output_size);
};

/// `AdaptiveAvgPoolOptions` specialized for the `AdaptiveAvgPool1d` module.
///
/// Example:
/// ```
/// AdaptiveAvgPool1d model(AdaptiveAvgPool1dOptions(5));
/// ```
using AdaptiveAvgPool1dOptions = AdaptiveAvgPoolOptions<ExpandingArray<1>>;

/// `AdaptiveAvgPoolOptions` specialized for the `AdaptiveAvgPool2d` module.
///
/// Example:
/// ```
/// AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
/// ```
using AdaptiveAvgPool2dOptions =
    AdaptiveAvgPoolOptions<ExpandingArrayWithOptionalElem<2>>;

/// `AdaptiveAvgPoolOptions` specialized for the `AdaptiveAvgPool3d` module.
///
/// Example:
/// ```
/// AdaptiveAvgPool3d model(AdaptiveAvgPool3dOptions(3));
/// ```
using AdaptiveAvgPool3dOptions =
    AdaptiveAvgPoolOptions<ExpandingArrayWithOptionalElem<3>>;

namespace functional {
/// Options for `torch::nn::functional::adaptive_avg_pool1d`.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool1dOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));
/// ```
using AdaptiveAvgPool1dFuncOptions = AdaptiveAvgPool1dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::adaptive_avg_pool2d`.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool2dOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions(3));
/// ```
using AdaptiveAvgPool2dFuncOptions = AdaptiveAvgPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::adaptive_avg_pool3d`.
///
/// See the documentation for `torch::nn::AdaptiveAvgPool3dOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool3d(x, F::AdaptiveAvgPool3dFuncOptions(3));
/// ```
using AdaptiveAvgPool3dFuncOptions = AdaptiveAvgPool3dOptions;
} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional maxunpool module.
template <size_t D>
struct MaxUnpoolOptions {
  MaxUnpoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take a max over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
};

/// `MaxUnpoolOptions` specialized for the `MaxUnpool1d` module.
///
/// Example:
/// ```
/// MaxUnpool1d model(MaxUnpool1dOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool1dOptions = MaxUnpoolOptions<1>;

/// `MaxUnpoolOptions` specialized for the `MaxUnpool2d` module.
///
/// Example:
/// ```
/// MaxUnpool2d model(MaxUnpool2dOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool2dOptions = MaxUnpoolOptions<2>;

/// `MaxUnpoolOptions` specialized for the `MaxUnpool3d` module.
///
/// Example:
/// ```
/// MaxUnpool3d model(MaxUnpool3dOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool3dOptions = MaxUnpoolOptions<3>;

// ============================================================================

namespace functional {

/// Options for a `D`-dimensional maxunpool functional.
template <size_t D>
struct MaxUnpoolFuncOptions {
  MaxUnpoolFuncOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take a max over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// the targeted output size
  TORCH_ARG(std::optional<std::vector<int64_t>>, output_size) = c10::nullopt;
};

/// `MaxUnpoolFuncOptions` specialized for
/// `torch::nn::functional::max_unpool1d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool1d(x, indices,
/// F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool1dFuncOptions = MaxUnpoolFuncOptions<1>;

/// `MaxUnpoolFuncOptions` specialized for
/// `torch::nn::functional::max_unpool2d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool2d(x, indices,
/// F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));
/// ```
using MaxUnpool2dFuncOptions = MaxUnpoolFuncOptions<2>;

/// `MaxUnpoolFuncOptions` specialized for
/// `torch::nn::functional::max_unpool3d`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));
/// ```
using MaxUnpool3dFuncOptions = MaxUnpoolFuncOptions<3>;

} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional fractional maxpool module.
template <size_t D>
struct FractionalMaxPoolOptions {
  FractionalMaxPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size) {}

  /// the size of the window to take a max over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the target output size of the image
  TORCH_ARG(std::optional<ExpandingArray<D>>, output_size) = c10::nullopt;

  /// If one wants to have an output size as a ratio of the input size, this
  /// option can be given. This has to be a number or tuple in the range (0, 1)
  using ExpandingArrayDouble = torch::ExpandingArray<D, double>;
  TORCH_ARG(std::optional<ExpandingArrayDouble>, output_ratio) = c10::nullopt;

  TORCH_ARG(torch::Tensor, _random_samples) = Tensor();
};

/// `FractionalMaxPoolOptions` specialized for the `FractionalMaxPool2d` module.
///
/// Example:
/// ```
/// FractionalMaxPool2d model(FractionalMaxPool2dOptions(5).output_size(1));
/// ```
using FractionalMaxPool2dOptions = FractionalMaxPoolOptions<2>;

/// `FractionalMaxPoolOptions` specialized for the `FractionalMaxPool3d` module.
///
/// Example:
/// ```
/// FractionalMaxPool3d model(FractionalMaxPool3dOptions(5).output_size(1));
/// ```
using FractionalMaxPool3dOptions = FractionalMaxPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::fractional_max_pool2d` and
/// `torch::nn::functional::fractional_max_pool2d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool2d(x,
/// F::FractionalMaxPool2dFuncOptions(3).output_size(2));
/// ```
using FractionalMaxPool2dFuncOptions = FractionalMaxPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::fractional_max_pool3d` and
/// `torch::nn::functional::fractional_max_pool3d_with_indices`
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool3d(x,
/// F::FractionalMaxPool3dFuncOptions(3).output_size(2));
/// ```
using FractionalMaxPool3dFuncOptions = FractionalMaxPool3dOptions;
} // namespace functional

// ============================================================================

/// Options for a `D`-dimensional lppool module.
template <size_t D>
struct LPPoolOptions {
  LPPoolOptions(double norm_type, ExpandingArray<D> kernel_size)
      : norm_type_(norm_type),
        kernel_size_(kernel_size),
        stride_(kernel_size) {}

  TORCH_ARG(double, norm_type);

  // the size of the window to take an average over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  // the stride of the window. Default value is `kernel_size`
  TORCH_ARG(ExpandingArray<D>, stride);

  // when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;
};

/// `LPPoolOptions` specialized for the `LPPool1d` module.
///
/// Example:
/// ```
/// LPPool1d model(LPPool1dOptions(1, 2).stride(5).ceil_mode(true));
/// ```
using LPPool1dOptions = LPPoolOptions<1>;

/// `LPPoolOptions` specialized for the `LPPool2d` module.
///
/// Example:
/// ```
/// LPPool2d model(LPPool2dOptions(1, std::vector<int64_t>({3, 4})).stride({5,
/// 6}).ceil_mode(true));
/// ```
using LPPool2dOptions = LPPoolOptions<2>;

/// `LPPoolOptions` specialized for the `LPPool3d` module.
///
/// Example:
/// ```
/// LPPool3d model(LPPool3dOptions(1, std::vector<int64_t>({3, 4, 5})).stride(
/// {5, 6, 7}).ceil_mode(true));
/// ```
using LPPool3dOptions = LPPoolOptions<3>;

namespace functional {
/// Options for `torch::nn::functional::lp_pool1d`.
///
/// See the documentation for `torch::nn::LPPool1dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool1d(x, F::LPPool1dFuncOptions(2, 3).stride(2));
/// ```
using LPPool1dFuncOptions = LPPool1dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::lp_pool2d`.
///
/// See the documentation for `torch::nn::LPPool2dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool2d(x, F::LPPool2dFuncOptions(2, {2, 3}).stride(2));
/// ```
using LPPool2dFuncOptions = LPPool2dOptions;
} // namespace functional

namespace functional {
/// Options for `torch::nn::functional::lp_pool3d`.
///
/// See the documentation for `torch::nn::LPPool3dOptions` class to learn what
/// arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool3d(x, F::LPPool3dFuncOptions(2, {2, 3, 4}).stride(2));
/// ```
using LPPool3dFuncOptions = LPPool3dOptions;
} // namespace functional

} // namespace nn
} // namespace torch
