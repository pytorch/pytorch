#pragma once

#include <torch/data/samplers/base.h>
#include <torch/tensor.h>

#include <ATen/optional.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {

/// A `Sampler` that returns random indices.
class RandomSampler : public Sampler {
 public:
  /// Constructs a `RandomSampler` with a size and dtype for the stored indices.
  ///
  /// The constructor will eagerly allocate all required indices, which is the
  /// sequence `0 ... size - 1`. `index_dtype` is the data type of the stored
  /// indices. You can change it to influence memory usage.
  explicit RandomSampler(int64_t size, Dtype index_dtype = torch::kInt64)
      : indices_(torch::randperm(size, index_dtype)) {}

  /// Resets the `RandomSampler` to a new set of indices.
  void reset() override {
    // This allocates a new chunk of memory every time (just FYI). It should be
    // amortized over the entire epoch hopefully.
    indices_ = torch::randperm(indices_.numel(), indices_.options());
    index_ = 0;
  }

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override {
    AT_ASSERT(index_ <= indices_.numel());
    const size_t remaining_indices = indices_.numel() - index_;
    if (remaining_indices == 0) {
      return nullopt;
    }
    std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
    auto slice = indices_.slice(/*dim=*/0, index_, index_ + index_batch.size());
    // You may want to store your indices with 32-bit or less, but here we need
    // to upcast to 64-bit. A batch itself won't hold too many indices, so that
    // should be ok. Note that if this indeed results in a type promotion, there
    // will be two allocations: one for the upcast slice, and one for the
    // returned `index_batch` vector.
    slice = slice.to(torch::kInt64);
    const auto* data = slice.data<int64_t>();
    std::copy(data, data + index_batch.size(), index_batch.begin());
    index_ += index_batch.size();
    return index_batch;
  }

 private:
  Tensor indices_;
  int64_t index_{0};
};
} // namespace samplers
} // namespace data
} // namespace torch
