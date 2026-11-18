#pragma once

#include <torch/data/dataloader/base.h>
#include <torch/data/worker_exception.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <thread>
#include <utility>

namespace torch::data {

/// A dataloader for stateless datasets.
///
/// This dataloader follows the traditional PyTorch dataloader design, whereby a
/// (possibly) stateful sampler produces *batch requests* for a stateless
/// dataset, which acts as a simple batch request to batch mapping. The batch
/// request will often be an array of indices, and if the dataset is a simple
/// image dataset, the dataset would produce the images at those indices.
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

  /// Constructs the `StatelessDataLoader` from a `dataset`, a `sampler` and
  /// some `options`.
  StatelessDataLoader(
      Dataset dataset,
      Sampler sampler,
      DataLoaderOptions options)
      : super(options), sampler_(std::move(sampler)) {
    for (const auto w : c10::irange(this->options_.workers)) {
      // Here we copy the dataset into the worker thread closure. Each worker
      // has its own copy of the dataset. This means the dataset must be
      // trivially copiable, or else we don't expect more than one worker to
      // be in use.
      (void)w; // Suppress unused variable warning
      this->workers_.emplace_back(
          [this, dataset]() mutable { this->worker_thread(dataset); });
    }
    if (this->options_.workers == 0) {
      this->main_thread_dataset_ =
          std::make_unique<Dataset>(std::move(dataset));
    }
  }

 private:
  /// Resets the internal state of the dataloader and the sampler.
  void reset() override {
    sampler_.reset();
    // Call the base class method last because it calls `prefetch()`
    super::reset();
  }

  /// Queries the sampler for the next batch request (possibly progressing its
  /// internal state).
  std::optional<BatchRequestType> get_batch_request() override {
    auto indices = sampler_.next(this->options_.batch_size);
    if (!indices ||
        (indices->size() < this->options_.batch_size &&
         this->options_.drop_last)) {
      return std::nullopt;
    }
    AT_ASSERT(indices->size() > 0);
    return indices;
  }

  /// The `Sampler` used to produce batch requests.
  Sampler sampler_;
};
} // namespace torch::data
