#pragma once

#include <torch/data/impl.h>

namespace torch {
namespace data {
template <typename Dataset>
class StatefulImpl : public Impl<Dataset, typename Dataset::BatchRequestType> {
 public:
  using super = Impl<Dataset, typename Dataset::BatchRequestType>;
  using super::options_;
  using super::prefetch;
  using super::push_job;
  using super::sequencer_;
  using super::workers_;
  using typename super::Batch;
  using typename super::BatchRequestType;
  using typename super::Result;

  StatefulImpl(Dataset dataset, DataLoaderOptions options)
      : super(std::move(options)) {
    for (size_t w = 0; w < options_.workers; ++w) {
      // Here we copy the dataset into the worker thread closure. Each worker
      // has its own copy of the dataset. This means the dataset must be
      // trivially copiable, or else we don't expect more than one worker to
      // be in use.
      workers_.emplace_back([this, dataset]() mutable {
        this->worker_thread(std::move(dataset));
      });
    }
    if (options_.workers == 0) {
      main_thread_dataset_ = torch::make_unique<Dataset>(std::move(dataset));
    }
  }

  void reset(bool prefetch = true) override {
    // dataset reset
    super::reset(prefetch);
  }

 private:
  optional<Batch> next() {
    optional<Batch> batch;
    // if (options_.workers > 0) {
    //   optional<Result> result = sequencer_->next(
    //       [this] { return this->shuttle_.pop_result(this->options_.timeout);
    //       });
    //   if (result) {
    //     if (result->exception) {
    //       throw WorkerException(result->exception);
    //     } else {
    //       AT_ASSERT(result->batch.has_value());
    //       batch = std::move(result->batch);
    //       prefetch(1);
    //     }
    //   }
    // } else if (auto batch_request = get_batch_request()) {
    //   AT_ASSERT(main_thread_dataset_ != nullptr);
    //   batch = main_thread_dataset_->get_batch(std::move(*batch_request));
    // }
    return batch;
  }

  void prefetch(size_t requested_jobs) override {
    // while (requested_jobs-- > 0) {
    //   if (auto batch_request = get_batch_request()) {
    //     push_job(std::move(*batch_request));
    //   } else {
    //     break;
    //   }
    // }
  }

  /// The dataset for the main thread, only has a value if the number of
  /// worker threads was configured as zero, meaning the main thread has to do
  /// all the work (synchronously). NOTE: Really want this to be on the heap
  /// when empty, therefore `unique_ptr` and not `optional`.
  std::unique_ptr<Dataset> main_thread_dataset_;

}; // namespace data
} // namespace data
} // namespace torch
