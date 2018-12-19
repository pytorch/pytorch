#pragma once

#include <torch/data/datasets/base.h>
#include <torch/types.h>

#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace torch {
namespace data {
namespace datasets {
namespace detail {
template <bool C, typename T>
using optional_if_t = typename std::conditional<C, torch::optional<T>, T>::type;
} // namespace detail

/// A `MapDataset` is a dataset that applies a transform to a source dataset.
template <typename SourceDataset, typename AppliedTransform>
struct MapDataset : BatchDataset<
                        MapDataset<SourceDataset, AppliedTransform>,
                        detail::optional_if_t<
                            SourceDataset::is_stateful,
                            typename AppliedTransform::OutputBatchType>,
                        typename SourceDataset::BatchRequestType> {
  using DatasetType = SourceDataset;
  using TransformType = AppliedTransform;
  using BatchRequestType = typename SourceDataset::BatchRequestType;
  using OutputBatchType = detail::optional_if_t<
      SourceDataset::is_stateful,
      typename AppliedTransform::OutputBatchType>;

  MapDataset(DatasetType dataset, TransformType transform)
      : dataset(std::move(dataset)), transform(std::move(transform)) {}

  /// Gets a batch from the source dataset and applies the transform to it,
  /// returning the result.
  OutputBatchType get_batch(BatchRequestType indices) override {
    return get_batch_impl(std::move(indices));
  }

  /// Returns the size of the source dataset.
  optional<size_t> size() const noexcept override {
    return dataset.size();
  }

  void reset() {
    dataset.reset();
  }

  SourceDataset dataset;
  AppliedTransform transform;

 private:
  template <
      typename D = SourceDataset,
      typename = torch::disable_if_t<D::is_stateful>>
  OutputBatchType get_batch_impl(BatchRequestType indices) {
    return transform.apply_batch(dataset.get_batch(indices));
  }

  template <typename D = SourceDataset>
  torch::enable_if_t<D::is_stateful, OutputBatchType> get_batch_impl(
      BatchRequestType indices) {
    if (auto batch = dataset.get_batch(indices)) {
      return transform.apply_batch(std::move(*batch));
    }
    return nullopt;
  }
};

/// Creates a `MapDataset` with the given dataset and transform.
template <typename DatasetType, typename TransformType>
MapDataset<DatasetType, TransformType> map(
    DatasetType dataset,
    TransformType transform) {
  static_assert(
      std::is_same<
          typename std::conditional<
              DatasetType::is_stateful,
              typename DatasetType::BatchType::value_type,
              typename DatasetType::BatchType>::type,
          typename TransformType::InputBatchType>::value,
      "BatchType type of dataset does not match input type of transform");
  return {std::move(dataset), std::move(transform)};
}

} // namespace datasets
} // namespace data
} // namespace torch
