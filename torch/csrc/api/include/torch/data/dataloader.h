#pragma once

#include <torch/data/dataloader_options.h>
#include <torch/data/detail/data_shuttle.h>
#include <torch/data/detail/sequencers.h>
#include <torch/data/iterator.h>
#include <torch/data/samplers/random.h>
#include <torch/data/worker_exception.h>
#include <torch/types.h>

#include <torch/data/datasets/stateful.h>
#include <torch/data/impl.h>
#include <torch/data/stateful_impl.h>
#include <torch/data/stateless_impl.h>

#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace data {
template <typename Impl>
class DataLoader {
 public:
  using Batch = typename Impl::Batch;

  virtual ~DataLoader() {
    join();
  }

  /// Returns an iterator into the `DataLoader`. The lifetime of the iterator is
  /// bound to the `DataLoader`. In C++ standards language, the category of the
  /// iterator is `OutputIterator`. See
  /// https://en.cppreference.com/w/cpp/named_req/OutputIterator for what this
  /// means. In short: you may increment the iterator and dereference it, but
  /// cannot go back, or step forward more than one position at a time. When the
  /// `DataLoader` is exhausted, it will compare equal with the special
  /// "sentinel" iterator returned by `DataLoader::end()`. Most of the time, you
  /// should only use range-for loops to loop over the `DataLoader`, but
  /// standard algorithms like `std::copy(dataloader.begin(), dataloader.end(),
  /// output_iterator)`  are supported too.
  Iterator<Batch> begin() {
    return impl_.begin();
  }

  /// Returns a special "sentinel" iterator that compares equal with a
  /// non-sentinel iterator once the `DataLoader` is exhausted.
  Iterator<Batch> end() {
    return impl_.end();
  }

  /// Joins the `DataLoader`'s worker threads and drains internal queues.
  /// This function may only be invoked from the main thread (in which the
  /// `DataLoader` lives).
  void join() {
    return impl_.join();
  }

  /// Returns the options with which the `DataLoader` was configured.
  const FullDataLoaderOptions& options() const noexcept {
    return impl_.options();
  }

  template <typename... Args>
  explicit DataLoader(Args&&... args) : impl_(std::forward<Args>(args)...) {}

 private:
  Impl impl_;
}; // namespace data


// !!!!
//
// Add a boolean to BatchDataset/StatefulBatchDataset that is is_stateful and enable_if true or false
//
// !!!!

/// Creates a new `DataLoader`, inferring the necessary template types from
/// the given arguments.
template <typename... Ts, typename Sampler>
std::unique_ptr<
  DataLoader<
    StatelessImpl<
      datasets::BatchDataset<Ts...>, Sampler
    >
  >
>
make_data_loader(
    datasets::BatchDataset<Ts...>&& dataset,
    DataLoaderOptions options,
    Sampler sampler) {
  return torch::make_unique<
  DataLoader<
    StatelessImpl<
      datasets::BatchDataset<Ts...>, Sampler
    >
  >
  >(
      std::move(dataset), std::move(options), std::move(sampler));
}

template <typename... Ts>
std::unique_ptr<
  DataLoader<
    StatefulImpl<
      datasets::StatefulBatchDataset<Ts...>
    >
  >
>
make_data_loader(
    datasets::StatefulBatchDataset<Ts...> dataset,
    DataLoaderOptions options = DataLoaderOptions()) {
  return nullptr;
  // return torch::make_unique<DataLoader<Dataset, Sampler>>(
  //     std::move(dataset), std::move(options), std::move(sampler));
}

/// Creates a new `DataLoader`, inferring the necessary template types from
/// the given arguments.
// template <
//     typename Sampler = samplers::RandomSampler,
//     typename Dataset,
//     typename =
//         torch::enable_if_t<std::is_constructible<Sampler, size_t>::value>>
//
// std::unique_ptr<DataLoader<StatelessImpl<Dataset, Sampler>>>
//
//
// make_data_loader(
//     Dataset dataset,
//     DataLoaderOptions options = DataLoaderOptions()) {
//   const optional<size_t> size = dataset.size();
//   AT_CHECK(
//       size.has_value(),
//       "Expected the dataset to be sized in "
//       "order to construct the Sampler");
//   return make_data_loader(
//       std::move(dataset), std::move(options), Sampler(*size));
// }

} // namespace data
} // namespace torch
