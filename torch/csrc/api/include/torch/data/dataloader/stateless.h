#pragma once

#include <torch/data/dataloader/base.h>
#include <torch/data/worker_exception.h>

#include <torch/csrc/utils/memory.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <thread>
#include <utility>

namespace torch {
namespace data {
template <typename Dataset, typename Sampler>
class StatelessDataLoader : public DataLoaderBase<
                                Dataset,
                                typename Dataset::BatchType,
                                typename Sampler::BatchRequestType> {
 public:
  using super = DataLoaderBase<
      Dataset,
      typename Dataset::BatchType,
      typename Sampler::BatchRequestType>;
  using typename super::BatchRequestType;

  StatelessDataLoader(
      Dataset dataset,
      DataLoaderOptions options,
      Sampler sampler)
      : super(
            options.workers_ ? std::unique_ptr<Dataset>(nullptr)
                             : torch::make_unique<Dataset>(std::move(dataset)),
            std::move(options)),
        sampler_(std::move(sampler)) {
    for (size_t w = 0; w < this->options_.workers; ++w) {
      // Here we copy the dataset into the worker thread closure. Each worker
      // has its own copy of the dataset. This means the dataset must be
      // trivially copiable, or else we don't expect more than one worker to
      // be in use.
      this->workers_.emplace_back(
          [this, dataset]() mutable { this->worker_thread(dataset); });
    }
  }

  void reset(bool prefetch = true) override {
    sampler_.reset();
    super::reset(prefetch);
  }

 private:
  optional<BatchRequestType> get_batch_request() override {
    auto indices = sampler_.next(this->options_.batch_size);
    if (!indices ||
        (indices->size() < this->options_.batch_size &&
         this->options_.drop_last)) {
      return nullopt;
    }
    AT_ASSERT(indices->size() > 0);
    return indices;
  }

  Sampler sampler_;
};
} // namespace data
} // namespace torch
