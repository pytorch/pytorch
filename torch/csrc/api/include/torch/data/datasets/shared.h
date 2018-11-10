#pragma once

#include <torch/data/datasets/base.h>

#include <c10/util/Exception.h>

#include <memory>
#include <utility>

namespace torch {
namespace data {
namespace datasets {

template <typename UnderlyingDataset>
class SharedDataset : BatchDataset<
                          SharedDataset<UnderlyingDataset>,
                          typename UnderlyingDataset::BatchType,
                          typename UnderlyingDataset::BatchRequestType> {
 public:
  using BatchType = typename UnderlyingDataset::BatchType;
  using BatchRequestType = typename UnderlyingDataset::BatchRequestType;

  /* implicit */ SharedDataset(
      std::shared_ptr<UnderlyingDataset> shared_dataset)
      : dataset_(std::move(shared_dataset)) {}

  BatchType get_batch(BatchRequestType request) override {
    AT_ASSERT(dataset_ != nullptr);
    return dataset_->get_batch(std::move(request));
  }

  optional<size_t> size() const override {
    AT_ASSERT(dataset_ != nullptr);
    return dataset_->size();
  }

  UnderlyingDataset& operator*() {
    return *dataset_;
  }

  const UnderlyingDataset& operator*() const {
    return *dataset_;
  }

  UnderlyingDataset* operator->() {
    return dataset_.get();
  }

  const UnderlyingDataset* operator->() const {
    return dataset_.get();
  }

 private:
  std::shared_ptr<UnderlyingDataset> dataset_;
};

template <typename UnderlyingDataset, typename... Args>
SharedDataset<UnderlyingDataset> make_shared_dataset(Args&&... args) {
  return std::make_shared<UnderlyingDataset>(std::forward<Args>(args)...);
}
} // namespace datasets
} // namespace data
} // namespace torch
