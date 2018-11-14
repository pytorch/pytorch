#pragma once

#include <torch/data/samplers/base.h>
#include <torch/types.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

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

/// A `Sampler` that returns indices sequentially.
class SequentialSampler : public Sampler<> {
 public:
  /// Creates a `SequentialSampler` that will return indices in the range
  /// `0...size - 1`.
  TORCH_API explicit SequentialSampler(size_t size);

  /// Resets the `SequentialSampler` to zero.
  TORCH_API void reset() override;

  /// Returns the next batch of indices.
  TORCH_API optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `SequentialSampler` to the `archive`.
  TORCH_API void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `SequentialSampler` from the `archive`.
  TORCH_API void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `SequentialSampler`.
  TORCH_API size_t index() const noexcept;

 private:
  size_t size_;
  size_t index_{0};
};

} // namespace samplers
} // namespace data
} // namespace torch
