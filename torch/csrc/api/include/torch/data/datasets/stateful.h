#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace datasets {

/// A stateful dataset is a dataset that maintains some internal state, which
/// will be `reset()` at the beginning of each epoch. Subclasses can override
/// the `reset()` method to configure this behavior.
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = size_t>
class StatefulDataset
    : public BatchDataset<Self, optional<Batch>, BatchRequest> {
 public:
  /// Resets internal state of the dataset.
  virtual void reset() = 0;
};
} // namespace datasets
} // namespace data
} // namespace torch
