#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/datasets/map.h>
#include <torch/data/example.h>

#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
template <typename S, typename T>
struct StatefulMapDataset;
template <typename D, typename T>
StatefulMapDataset<D, T> stateful_map(D, T); // NOLINT
} // namespace datasets
} // namespace data
} // namespace torch

namespace torch {
namespace data {
namespace datasets {

template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = ArrayRef<size_t>>
class StatefulBatchDataset
    : public BatchDataset<Self, optional<Batch>, BatchRequest> {
 public:
  virtual void reset() = 0;

  template <typename TransformType>
  StatefulMapDataset<Self, TransformType> map(TransformType transform) & {
    return stateful_map(static_cast<Self&>(*this), std::move(transform));
  }

  /// Creates a `MapDataset` that applies the given `transform` to this dataset.
  template <typename TransformType>
  StatefulMapDataset<Self, TransformType> map(TransformType transform) && {
    return stateful_map(
        std::move(static_cast<Self&>(*this)), std::move(transform));
  }
};

template <typename Self, typename SingleExample = Example<>>
class StatefulDataset
    : public StatefulBatchDataset<Self, std::vector<SingleExample>> {
 public:
  using ExampleType = optional<SingleExample>;

  virtual optional<SingleExample> get(size_t index) = 0;

  optional<std::vector<SingleExample>> get_batch(
      ArrayRef<size_t> indices) override {
    optional<std::vector<SingleExample>> batch;
    for (const auto i : indices) {
      if (ExampleType example = get(i)) {
        if (!batch) {
          batch = std::vector<SingleExample>();
          batch->reserve(indices.size());
        }
        batch->push_back(*example);
      } else {
        break;
      }
    }
    return batch;
  }
};

/// A `StatefulMapDataset` is a dataset that applies a transform to a source
/// dataset.
template <typename SourceDataset, typename AppliedTransform>
struct StatefulMapDataset
    : StatefulBatchDataset<
          StatefulMapDataset<SourceDataset, AppliedTransform>,
          typename AppliedTransform::OutputBatchType,
          typename SourceDataset::BatchRequestType> {
  using DatasetType = SourceDataset;
  using TransformType = AppliedTransform;
  using BatchRequestType = typename SourceDataset::BatchRequestType;
  using OutputBatchType = optional<typename TransformType::OutputBatchType>;

  StatefulMapDataset(DatasetType dataset, TransformType transform)
      : dataset(std::move(dataset)), transform(std::move(transform)) {}

  /// Gets a batch from the source dataset and applies the transform to it,
  /// returning the result.
  OutputBatchType get_batch(BatchRequestType indices) override {
    if (auto batch = dataset.get_batch(indices)) {
      return transform.apply_batch(std::move(*batch));
    }
    return nullopt;
  }

  /// Returns the size of the source dataset.
  optional<size_t> size() const noexcept {
    return dataset.size();
  }

  void reset() override {
    dataset.reset();
  }

  SourceDataset dataset;
  AppliedTransform transform;
};

/// Creates a `StatefulMapDataset` with the given dataset and transform.
template <typename DatasetType, typename TransformType>
StatefulMapDataset<DatasetType, TransformType> stateful_map(
    DatasetType dataset,
    TransformType transform) {
  static_assert(
      std::is_same<
          typename DatasetType::BatchType::value_type,
          typename TransformType::InputBatchType>::value,
      "BatchType type of dataset does not match input type of transform");
  return {std::move(dataset), std::move(transform)};
}

} // namespace datasets
} // namespace data
} // namespace torch
