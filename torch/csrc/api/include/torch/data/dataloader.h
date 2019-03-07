#pragma once

#include <torch/data/dataloader/stateful.h>
#include <torch/data/dataloader/stateless.h>

#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace torch {
namespace data {

/// Creates a `DataLoader` instance for a stateless `dataset`, a `sampler` and
/// some `options`.
template <typename Dataset, typename Sampler>
torch::disable_if_t<
    Dataset::is_stateful,
    std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
make_data_loader(Dataset dataset, Sampler sampler, DataLoaderOptions options) {
  return torch::make_unique<StatelessDataLoader<Dataset, Sampler>>(
      std::move(dataset), std::move(sampler), std::move(options));
}

/// Creates a `DataLoader` instance for a stateless `dataset` and some
/// `options`. A sampler (by default a `RandomSampler`) will be constructed from
/// the size of the dataset.
template <typename Sampler = samplers::RandomSampler, typename Dataset>
torch::disable_if_t<
    Dataset::is_stateful || !std::is_constructible<Sampler, size_t>::value,
    std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
make_data_loader(
    Dataset dataset,
    DataLoaderOptions options = DataLoaderOptions()) {
  const optional<size_t> size = dataset.size();
  AT_CHECK(
      size.has_value(),
      "Expected the dataset to be sized in "
      "order to construct the Sampler");
  return make_data_loader(
      std::move(dataset), Sampler(*size), std::move(options));
}

/// Creates a `DataLoader` for a stateful `dataset` and some `options`.
template <typename Dataset, typename = torch::enable_if_t<Dataset::is_stateful>>
std::unique_ptr<StatefulDataLoader<Dataset>> make_data_loader(
    Dataset dataset,
    DataLoaderOptions options = DataLoaderOptions()) {
  return torch::make_unique<StatefulDataLoader<Dataset>>(
      std::move(dataset), std::move(options));
}
} // namespace data
} // namespace torch
