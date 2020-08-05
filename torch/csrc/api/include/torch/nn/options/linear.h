#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <c10/util/variant.h>

namespace torch {
namespace nn {

/// Options for the `Linear` module.
///
/// Example:
/// ```
/// Linear model(LinearOptions(5, 2).bias(false));
/// ```
struct TORCH_API LinearOptions {
  LinearOptions(int64_t in_features, int64_t out_features);
  /// size of each input sample
  TORCH_ARG(int64_t, in_features);

  /// size of each output sample
  TORCH_ARG(int64_t, out_features);

  /// If set to false, the layer will not learn an additive bias. Default: true
  TORCH_ARG(bool, bias) = true;
};

// ============================================================================

/// Options for the `Flatten` module.
///
/// Example:
/// ```
/// Flatten model(FlattenOptions().start_dim(2).end_dim(4));
/// ```
struct TORCH_API FlattenOptions {
  /// first dim to flatten
  TORCH_ARG(int64_t, start_dim) = 1;
  /// last dim to flatten
  TORCH_ARG(int64_t, end_dim) = -1;
};

// ============================================================================

/// Options for the `Unflatten` module.
///
/// Example:
/// ```
/// Unflatten model(UnflattenOptions(0, {2, 2}));
/// Unflatten model(UnflattenOptions("B", {{"B1", 2}, {"B2", 2}}));
/// ```
struct TORCH_API UnflattenOptions {
  typedef std::vector<std::tuple<std::string, int64_t>> namedshape_t;
  typedef c10::variant<std::vector<int64_t>, namedshape_t> sizes_t;
  typedef c10::variant<int64_t, std::string> dim_t;
  
  UnflattenOptions(int64_t dim, std::vector<int64_t> unflattened_size);
  UnflattenOptions(std::string dim, namedshape_t unflattened_size);

  /// dim to unflatten
  TORCH_ARG(dim_t, dim);
  /// new shape of unflattened dim
  TORCH_ARG(sizes_t, unflattened_size);
};

// ============================================================================

/// Options for the `Bilinear` module.
///
/// Example:
/// ```
/// Bilinear model(BilinearOptions(3, 2, 4).bias(false));
/// ```
struct TORCH_API BilinearOptions {
  BilinearOptions(int64_t in1_features, int64_t in2_features, int64_t out_features);
  /// The number of features in input 1 (columns of the input1 matrix).
  TORCH_ARG(int64_t, in1_features);
  /// The number of features in input 2 (columns of the input2 matrix).
  TORCH_ARG(int64_t, in2_features);
  /// The number of output features to produce (columns of the output matrix).
  TORCH_ARG(int64_t, out_features);
  /// Whether to learn and add a bias after the bilinear transformation.
  TORCH_ARG(bool, bias) = true;
};

} // namespace nn
} // namespace torch
