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
class MapDataset : public BatchDataset<
                       MapDataset<SourceDataset, AppliedTransform>,
                       detail::optional_if_t<
                           SourceDataset::is_stateful,
                           typename AppliedTransform::OutputBatchType>,
                       typename SourceDataset::BatchRequestType> {
 public:
  using DatasetType = SourceDataset;
  using TransformType = AppliedTransform;
  using BatchRequestType = typename SourceDataset::BatchRequestType;
  using OutputBatchType = detail::optional_if_t<
      SourceDataset::is_stateful,
      typename AppliedTransform::OutputBatchType>;

  MapDataset(DatasetType dataset, TransformType transform)
      : dataset_(std::move(dataset)), transform_(std::move(transform)) {}

  /// Gets a batch from the source dataset and applies the transform to it,
  /// returning the result.
  OutputBatchType get_batch(BatchRequestType indices) override {
    return get_batch_impl(std::move(indices));
  }

  /// Returns the size of the source dataset.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  optional<size_t> size() const noexcept override {
    return dataset_.size();
  }

  /// Calls `reset()` on the underlying dataset.
  /// NOTE: Stateless datasets do not have a reset() method, so a call to this
  /// method will only compile for stateful datasets (which have a reset()
  /// method).
  void reset() {
    dataset_.reset();
  }

  /// Returns the underlying dataset.
  const SourceDataset& dataset() noexcept {
    return dataset_;
  }

  /// Returns the transform being applied.
  const AppliedTransform& transform() noexcept {
    return transform_;
  }

 private:
  /// The implementation of `get_batch()` for the stateless case, which simply
  /// applies the transform to the output of `get_batch()` from the dataset.
  template <
      typename D = SourceDataset,
      typename = torch::disable_if_t<D::is_stateful>>
  OutputBatchType get_batch_impl(BatchRequestType indices) {
    return transform_.apply_batch(dataset_.get_batch(std::move(indices)));
  }

  /// The implementation of `get_batch()` for the stateful case. Here, we follow
  /// the semantics of `Optional.map()` in many functional languages, which
  /// applies a transformation to the optional's content when the optional
  /// contains a value, and returns a new optional (of a different type)  if the
  /// original optional returned by `get_batch()` was empty.
  template <typename D = SourceDataset>
  torch::enable_if_t<D::is_stateful, OutputBatchType> get_batch_impl(
      BatchRequestType indices) {
    if (auto batch = dataset_.get_batch(std::move(indices))) {
      return transform_.apply_batch(std::move(*batch));
    }
    return nullopt;
  }

  /// The underlying dataset being transformed.
  SourceDataset dataset_;

  // The transformation that is applied to batches received from the dataset.
  AppliedTransform transform_;
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
