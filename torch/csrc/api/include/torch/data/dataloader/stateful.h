#pragma once

#include <torch/data/dataloader/base.h>

#include <torch/data/worker_exception.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <thread>
#include <utility>

namespace torch {
namespace data {
template <typename Dataset>
class StatefulDataLoader : public DataLoaderBase<
                               Dataset,
                               typename Dataset::BatchType::value_type,
                               typename Dataset::BatchRequestType> {
 public:
  using super = DataLoaderBase<
      Dataset,
      typename Dataset::BatchType::value_type,
      typename Dataset::BatchRequestType>;
  using typename super::BatchRequestType;

  StatefulDataLoader(Dataset dataset, DataLoaderOptions options)
      : super(
            torch::make_unique<Dataset>(std::move(dataset)),
            std::move(options)) {
    for (size_t w = 0; w < this->options_.workers; ++w) {
      this->workers_.emplace_back(
          [this] { this->worker_thread(*this->main_thread_dataset_); });
    }
  }

  void reset(bool prefetch = true) override {
    this->main_thread_dataset_->reset();
    super::reset(prefetch);
  }

 private:
  optional<BatchRequestType> get_batch_request() override {
    return this->options_.batch_size;
  }
};
} // namespace data
} // namespace torch
