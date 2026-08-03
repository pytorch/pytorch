#pragma once

#include <torch/csrc/Export.h>
#include <torch/data/samplers/base.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::jit::mobile {

/// A lighter `Sampler` that returns indices randomly and cannot be
/// serialized.
class TORCH_API RandomSampler : public torch::data::samplers::Sampler<> {
 public:
  /// Constructs a `RandomSampler` with a size and dtype for the stored indices.
  ///
  /// The constructor will eagerly allocate all required indices, which is the
  /// sequence `0 ... size - 1`. `index_dtype` is the data type of the stored
  /// indices. You can change it to influence memory usage.
  explicit RandomSampler(int64_t size, Dtype index_dtype = torch::kInt64);

  ~RandomSampler() override;

  /// Resets the `RandomSampler` to a new set of indices.
  void reset(std::optional<size_t> new_size = std::nullopt) override;

  /// Returns the next batch of indices.
  std::optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `RandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `RandomSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `RandomSampler`.
  size_t index() const noexcept;

 private:
  at::Tensor indices_;
  int64_t index_ = 0;
};

} // namespace torch::jit::mobile
