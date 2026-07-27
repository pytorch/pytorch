#pragma once

#include <torch/data/example.h>
#include <torch/types.h>

#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch::data::datasets {
template <typename S, typename T>
class MapDataset;
template <typename D, typename T>
MapDataset<D, T> map(D, T); // NOLINT
} // namespace torch::data::datasets

namespace torch::data::datasets {
namespace detail {
template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};
} // namespace detail

/// A dataset that can yield data only in batches.
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = ArrayRef<size_t>>
class BatchDataset {
 public:
  using SelfType = Self;
  using BatchType = Batch;
  using BatchRequestType = BatchRequest;
  constexpr static bool is_stateful = detail::is_optional<BatchType>::value;

  virtual ~BatchDataset() = default;

  /// Returns a batch of data given an index.
  virtual Batch get_batch(BatchRequest request) = 0;

  /// Returns the size of the dataset, or an empty std::optional if it is
  /// unsized.
  virtual std::optional<size_t> size() const = 0;

  /// Creates a `MapDataset` that applies the given `transform` to this dataset.
  template <typename TransformType>
  MapDataset<Self, TransformType> map(TransformType transform) & {
    return datasets::map(static_cast<Self&>(*this), std::move(transform));
  }

  /// Creates a `MapDataset` that applies the given `transform` to this dataset.
  template <typename TransformType>
  MapDataset<Self, TransformType> map(TransformType transform) && {
    return datasets::map(
        std::move(static_cast<Self&>(*this)), std::move(transform));
  }
};

/// A dataset that can yield data in batches, or as individual examples.
///
/// A `Dataset` is a `BatchDataset`, because it supports random access and
/// therefore batched access is implemented (by default) by calling the random
/// access indexing function for each index in the requested batch of indices.
/// This can be customized.
template <typename Self, typename SingleExample = Example<>>
class Dataset : public BatchDataset<Self, std::vector<SingleExample>> {
 public:
  using ExampleType = SingleExample;

  /// Returns the example at the given index.
  virtual ExampleType get(size_t index) = 0;

  /// Returns a batch of data.
  /// The default implementation calls `get()` for every requested index
  /// in the batch.
  std::vector<ExampleType> get_batch(ArrayRef<size_t> indices) override {
    std::vector<ExampleType> batch;
    batch.reserve(indices.size());
    for (const auto i : indices) {
      batch.push_back(get(i));
    }
    return batch;
  }
};

/// A `StreamDataset` represents a dataset that is a potentially infinite
/// stream. It takes as batch index only a number, which is the batch size, and
/// yields that many elements from the stream.
template <typename Self, typename Batch = std::vector<Example<>>>
using StreamDataset = BatchDataset<Self, Batch, /*BatchRequest=*/size_t>;
} // namespace torch::data::datasets
