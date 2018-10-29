#pragma once

#include <torch/tensor.h>

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

/// A `Sampler` is an object that yields indices with which to index into a
/// dataset.
class Sampler {
 public:
  virtual ~Sampler() = default;

  /// Resets the `Sampler`'s internal state.
  /// Typically called before a new epoch.
  virtual void reset() = 0;

  /// Returns the next batch of indices if possible, or an empty optional if the
  /// sampler is exhausted for this epoch.
  virtual optional<std::vector<size_t>> next(size_t batch_size) = 0;

  /// Serializes the `Sampler` to the `archive`.
  virtual void save(serialize::OutputArchive& archive) const = 0;

  /// Deserializes the `Sampler` from the `archive`.
  virtual void load(serialize::InputArchive& archive) = 0;
};

/// Serializes a `Sampler` into an `OutputArchive`.
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Sampler& sampler);

/// Deserializes a `Sampler` from an `InputArchive`.
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Sampler& sampler);
} // namespace samplers
} // namespace data
} // namespace torch
