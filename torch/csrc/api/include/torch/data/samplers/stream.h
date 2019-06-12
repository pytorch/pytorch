#pragma once

#include <torch/data/samplers/base.h>
#include <torch/data/samplers/custom_batch_request.h>
#include <torch/types.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>

namespace torch {
namespace serialize {
class InputArchive;
class OutputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace data {
namespace samplers {

/// A wrapper around a batch size value, which implements the
/// `CustomBatchRequest` interface.
struct TORCH_API BatchSize : public CustomBatchRequest {
  explicit BatchSize(size_t size);
  size_t size() const noexcept override;
  operator size_t() const noexcept;
  size_t size_;
};

/// A sampler for (potentially infinite) streams of data.
///
/// The major feature of the `StreamSampler` is that it does not return
/// particular indices, but instead only the number of elements to fetch from
/// the dataset. The dataset has to decide how to produce those elements.
class StreamSampler : public Sampler<BatchSize> {
 public:
  /// Constructs the `StreamSampler` with the number of individual examples that
  /// should be fetched until the sampler is exhausted.
  TORCH_API explicit StreamSampler(size_t epoch_size);

  /// Resets the internal state of the sampler.
  TORCH_API void reset() override;

  /// Returns a `BatchSize` object with the number of elements to fetch in the
  /// next batch. This number is the minimum of the supplied `batch_size` and
  /// the difference between the `epoch_size` and the current index. If the
  /// `epoch_size` has been reached, returns an empty optional.
  TORCH_API optional<BatchSize> next(size_t batch_size) override;

  /// Serializes the `StreamSampler` to the `archive`.
  TORCH_API void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `StreamSampler` from the `archive`.
  TORCH_API void load(serialize::InputArchive& archive) override;

 private:
  size_t examples_retrieved_so_far_ = 0;
  size_t epoch_size_;
};

} // namespace samplers
} // namespace data
} // namespace torch
