#pragma once

#include <c10/util/irange.h>
#include <torch/data/dataloader/base.h>

#include <cstddef>
#include <thread>
#include <utility>

namespace torch {
namespace data {

/// A dataloader for stateful datasets.
///
/// A dataloader for stateful datatasets differs from one for stateless
/// datasets one in that the dataset is shared among worker threads, and that
/// this dataset is itself responsible for producing batches rather than
/// depending on a sampler. The statefulness here actually refers to the
/// dataset. The StatefulDataLoader simply alters the data loading algorithm to
/// accommodate the stateful, shared nature of the dataset. Note that the
/// dataset must be thread safe if more than one worker thread is used.
///
/// A stateful dataloader is created by calling `make_data_loader` with a
/// stateful dataset.
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

  /// Constructs the `StatefulDataLoader` from a `dataset` and some `options`.
  StatefulDataLoader(Dataset dataset, DataLoaderOptions options)
      : super(options, std::make_unique<Dataset>(std::move(dataset))) {
    for ([[maybe_unused]] const auto _ : c10::irange(this->options_.workers)) {
      // As opposed to the stateless case, here all worker threads access the
      // same underlying dataset.
      this->workers_.emplace_back(
          [this] { this->worker_thread(*this->main_thread_dataset_); });
    }
  }

 private:
  /// Resets the internal state of the dataloader and the dataset.
  void reset() override {
    this->main_thread_dataset_->reset();
    // Call the base class method last because it calls `prefetch()`
    super::reset();
  }

  /// For stateful datasets, the batch request is always the batch size. The
  /// dataset is responsible for determining what goes into the batch next.
  std::optional<BatchRequestType> get_batch_request() override {
    return this->options_.batch_size;
  }
};
} // namespace data
} // namespace torch
