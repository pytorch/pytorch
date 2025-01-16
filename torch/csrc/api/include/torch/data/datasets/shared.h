#pragma once

#include <torch/data/datasets/base.h>

#include <memory>
#include <utility>

namespace torch::data::datasets {

/// A dataset that wraps another dataset in a shared pointer and implements the
/// `BatchDataset` API, delegating all calls to the shared instance. This is
/// useful when you want all worker threads in the dataloader to access the same
/// dataset instance. The dataset must take care of synchronization and
/// thread-safe access itself.
///
/// Use `torch::data::datasets::make_shared_dataset()` to create a new
/// `SharedBatchDataset` like you would a `std::shared_ptr`.
template <typename UnderlyingDataset>
class SharedBatchDataset : public BatchDataset<
                               SharedBatchDataset<UnderlyingDataset>,
                               typename UnderlyingDataset::BatchType,
                               typename UnderlyingDataset::BatchRequestType> {
 public:
  using BatchType = typename UnderlyingDataset::BatchType;
  using BatchRequestType = typename UnderlyingDataset::BatchRequestType;

  /// Constructs a new `SharedBatchDataset` from a `shared_ptr` to the
  /// `UnderlyingDataset`.
  /* implicit */ SharedBatchDataset(
      std::shared_ptr<UnderlyingDataset> shared_dataset)
      : dataset_(std::move(shared_dataset)) {}

  /// Calls `get_batch` on the underlying dataset.
  BatchType get_batch(BatchRequestType request) override {
    return dataset_->get_batch(std::move(request));
  }

  /// Returns the `size` from the underlying dataset.
  std::optional<size_t> size() const override {
    return dataset_->size();
  }

  /// Accesses the underlying dataset.
  UnderlyingDataset& operator*() {
    return *dataset_;
  }

  /// Accesses the underlying dataset.
  const UnderlyingDataset& operator*() const {
    return *dataset_;
  }

  /// Accesses the underlying dataset.
  UnderlyingDataset* operator->() {
    return dataset_.get();
  }

  /// Accesses the underlying dataset.
  const UnderlyingDataset* operator->() const {
    return dataset_.get();
  }

  /// Calls `reset()` on the underlying dataset.
  void reset() {
    dataset_->reset();
  }

 private:
  std::shared_ptr<UnderlyingDataset> dataset_;
};

/// Constructs a new `SharedBatchDataset` by creating a
/// `shared_ptr<UnderlyingDatase>`. All arguments are forwarded to
/// `make_shared<UnderlyingDataset>`.
template <typename UnderlyingDataset, typename... Args>
SharedBatchDataset<UnderlyingDataset> make_shared_dataset(Args&&... args) {
  return std::make_shared<UnderlyingDataset>(std::forward<Args>(args)...);
}
} // namespace torch::data::datasets
