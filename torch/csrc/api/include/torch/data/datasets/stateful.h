#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace data {
namespace datasets {

/// A stateful dataset is a dataset that maintains some internal state, which
/// will be `reset()` at the beginning of each epoch. Subclasses can override
/// the `reset()` method to configure this behavior. Further, the return type of
/// a stateful dataset's `get_batch()` method is always an `optional`. When the
/// stateful dataset wants to indicate to the dataloader that its epoch has
/// ended, it should return an empty optional. The dataloader knows to modify
/// its implementation based on whether the dataset is stateless or stateful.
///
/// Note that when subclassing a from `StatefulDataset<Self, T>`, the return
/// type of `get_batch()`, which the subclass must override, will be
/// `optional<T>` (i.e. the type specified in the `StatefulDataset`
/// specialization is automatically boxed into an `optional` for the dataset's
/// `BatchType`).
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = size_t>
class StatefulDataset
    : public BatchDataset<Self, std::optional<Batch>, BatchRequest> {
 public:
  /// Resets internal state of the dataset.
  virtual void reset() = 0;

  /// Saves the statefulDataset's state to OutputArchive.
  virtual void save(serialize::OutputArchive& archive) const = 0;

  /// Deserializes the statefulDataset's state from the `archive`.
  virtual void load(serialize::InputArchive& archive) = 0;
};

/// Serializes a statefulDataset to `OutputArchive`.
template <typename... Args>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const StatefulDataset<Args...>& statefulDataset) {
  statefulDataset.save(archive);
  return archive;
}

/// Deserializes a statefulDataset from an `InputArchive`.
template <typename... Args>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    StatefulDataset<Args...>& statefulDataset) {
  statefulDataset.load(archive);
  return archive;
}

} // namespace datasets
} // namespace data
} // namespace torch
