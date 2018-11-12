#pragma once

#include <torch/data/datasets/base.h>
#include <torch/types.h>

#include <ATen/core/ArrayRef.h>

#include <cstddef>
#include <utility>

namespace torch {
namespace data {
namespace datasets {

/// A `MapDataset` is a dataset that applies a transform to a source dataset.
template <typename SourceDataset, typename AppliedTransform>
struct MapDataset : BatchDataset<
                        MapDataset<SourceDataset, AppliedTransform>,
                        typename AppliedTransform::OutputBatchType,
                        typename SourceDataset::BatchRequestType> {
  using DatasetType = SourceDataset;
  using TransformType = AppliedTransform;
  using BatchRequestType = typename SourceDataset::BatchRequestType;
  using OutputBatchType = typename TransformType::OutputBatchType;

  MapDataset(DatasetType dataset, TransformType transform)
      : dataset(std::move(dataset)), transform(std::move(transform)) {}

  /// Gets a batch from the source dataset and applies the transform to it,
  /// returning the result.
  OutputBatchType get_batch(BatchRequestType indices) override {
    return transform.apply_batch(dataset.get_batch(indices));
  }

  /// Returns the size of the source dataset.
  optional<size_t> size() const noexcept {
    return dataset.size();
  }

  SourceDataset dataset;
  AppliedTransform transform;
};

/// Creates a `MapDataset` with the given dataset and transform.
template <typename DatasetType, typename TransformType>
MapDataset<DatasetType, TransformType> map(
    DatasetType dataset,
    TransformType transform) {
  static_assert(
      std::is_same<
          typename DatasetType::BatchType,
          typename TransformType::InputBatchType>::value,
      "BatchType type of dataset does not match input type of transform");
  return {std::move(dataset), std::move(transform)};
}

} // namespace datasets
} // namespace data
} // namespace torch
