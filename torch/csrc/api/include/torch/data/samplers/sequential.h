#pragma once

#include <torch/data/samplers/base.h>
#include <torch/tensor.h>

#include <ATen/optional.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace torch {
namespace data {
namespace samplers {

/// A `Sampler` that returns indices sequentially.
class SequentialSampler : public Sampler {
 public:
  /// Creates a `SequentialSampler` that will return indices in the range
  /// `0...size - 1`.
  explicit SequentialSampler(size_t size) : size_(size) {}

  /// Resets the `SequentialSampler` to zero.
  void reset() override {
    index_ = 0;
  }

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override {
    const auto remaining_indices = size_ - index_;
    if (remaining_indices == 0) {
      return nullopt;
    }
    std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
    for (auto& i : index_batch) {
      i = index_++;
    }
    return index_batch;
  }

 private:
  size_t size_;
  size_t index_{0};
};

} // namespace samplers
} // namespace data
} // namespace torch
