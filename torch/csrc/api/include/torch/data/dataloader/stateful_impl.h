#pragma once

#include <torch/data/dataloader/impl.h>

#include <torch/data/worker_exception.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <thread>
#include <utility>

namespace torch {
namespace data {
namespace detail {
namespace dataloader {
template <typename Dataset>
class StatefulImpl : public Impl<
                         Dataset,
                         typename Dataset::BatchType::value_type,
                         typename Dataset::BatchRequestType> {
 public:
  using super = Impl<
      Dataset,
      typename Dataset::BatchType::value_type,
      typename Dataset::BatchRequestType>;
  using super::options_;
  using super::pop_result;
  using super::push_job;
  using super::worker_thread;
  using super::workers_;
  using typename super::BatchType;
  using typename super::Result;

  StatefulImpl(Dataset dataset, DataLoaderOptions options)
      : super(std::move(options)), dataset_(std::move(dataset)) {
    for (size_t w = 0; w < options_.workers; ++w) {
      workers_.emplace_back([this] { this->worker_thread(this->dataset_); });
    }
  }

  void reset(bool prefetch = true) override {
    dataset_.reset();
    super::reset(prefetch);
  }

 private:
  optional<BatchType> next() {
    if (options_.workers > 0) {
      while (optional<Result> result = pop_result()) {
        if (result->exception) {
          throw WorkerException(result->exception);
        } else if (result->batch) {
          prefetch(1);
          return std::move(result->batch);
        }
      }
    } else {
      return dataset_.get_batch(options_.batch_size);
    }
    return at::nullopt;
  }

  void prefetch(size_t requested_jobs) override {
    for (size_t r = 0; r < requested_jobs; ++r) {
      push_job(options_.batch_size);
    }
  }

  Dataset dataset_;
};
} // namespace dataloader
} // namespace detail
} // namespace data
} // namespace torch
