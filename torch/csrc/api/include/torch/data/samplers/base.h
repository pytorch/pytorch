#pragma once

#include <torch/tensor.h>

#include <cstddef>
#include <vector>

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
};

} // namespace samplers
} // namespace data
} // namespace torch
