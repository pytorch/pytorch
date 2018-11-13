#pragma once

#include <torch/types.h>

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
namespace samplers {
/// A `Sampler` is an object that yields an index with which to access a
/// dataset.
template <typename BatchRequest = std::vector<size_t>>
class Sampler {
 public:
  using BatchRequestType = BatchRequest;

  virtual ~Sampler() = default;

  /// Resets the `Sampler`'s internal state.
  /// Typically called before a new epoch.
  virtual void reset() = 0;

  /// Returns the next index if possible, or an empty optional if the
  /// sampler is exhausted for this epoch.
  virtual optional<BatchRequest> next(size_t batch_size) = 0;

  /// Serializes the `Sampler` to the `archive`.
  virtual void save(serialize::OutputArchive& archive) const = 0;

  /// Deserializes the `Sampler` from the `archive`.
  virtual void load(serialize::InputArchive& archive) = 0;
};
} // namespace samplers
} // namespace data
} // namespace torch
