#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/lambda.h>

#include <vector>

namespace torch {
namespace data {
namespace transforms {

/// A `Collation` is a transform that reduces a batch into a single value.
/// The result is a `BatchDataset` that has the type of the single value as its
/// `BatchType`.
template <typename T, typename BatchType = std::vector<T>>
using Collation = BatchTransform<BatchType, T>;

/// A `Collate` allows passing a custom function to reduce/collate a batch
/// into a single value. It's effectively the lambda version of `Collation`,
/// which you could subclass and override `operator()` to achieve the same.
///
/// \rst
/// .. code-block:: cpp
///   using namespace torch::data;
///
///   auto dataset = datasets::MNIST("path/to/mnist")
///     .map(transforms::Collate<Example<>>([](std::vector<Example<>> e) {
///       return std::move(e.front());
///     }));
/// \endrst
template <typename T, typename BatchType = std::vector<T>>
using Collate = BatchLambda<BatchType, T>;
} // namespace transforms
} // namespace data
} // namespace torch
