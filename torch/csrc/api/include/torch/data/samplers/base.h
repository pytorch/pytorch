#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>
#include <mutex>

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
  /// Optionally, accepts a new size when reseting the sampler.
  TORCH_API virtual void reset(optional<size_t> new_size) = 0;

  /// Returns the next index if possible, or an empty optional if the
  /// sampler is exhausted for this epoch.
  TORCH_API virtual optional<BatchRequest> next(size_t batch_size) = 0;

  /// Serializes the `Sampler` to the `archive`.
  TORCH_API virtual void save(serialize::OutputArchive& archive) const = 0;

  /// Deserializes the `Sampler` from the `archive`.
  TORCH_API virtual void load(serialize::InputArchive& archive) = 0;
};

/// Wraps a provided sampler to make it thread safe.
template <typename OriginalSampler>
class LockedSampler
    : public Sampler<typename OriginalSampler::BatchRequestType> {
 public:
  using BatchRequestType = typename OriginalSampler::BatchRequestType;

  explicit LockedSampler(OriginalSampler sampler) : sampler_(std::move(sampler)) {}

  void reset(optional<size_t> new_size) override {
    std::lock_guard<std::mutex> lock(this->mutex_);
    sampler_.reset(new_size);
  }

  optional<BatchRequestType> next(size_t batch_size) override {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return sampler_.next(batch_size);
  }

  void save(serialize::OutputArchive& archive) const override {
    std::lock_guard<std::mutex> lock(this->mutex_);
    sampler_.save(archive);
  }

  void load(serialize::InputArchive& archive) override {
    std::lock_guard<std::mutex> lock(this->mutex_);
    sampler_.load(archive);
  }

 private:
  // member variable for multi-threading lock.
  // declare it to be mutable for locking in const member function.
  mutable std::mutex mutex_;
  OriginalSampler sampler_;
};
} // namespace samplers
} // namespace data
} // namespace torch
