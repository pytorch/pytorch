#pragma once

#include <torch/tensor.h>

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

/// Wraps a provided sampler to make it thread safe.
template <typename OriginalSampler>
class ThreadSafeSampler : public Sampler {
 public:
  ThreadSafeSampler(OriginalSampler sampler) : sampler_(std::move(sampler)) {}

  void reset() override {
    std::lock_guard<std::mutex> lock(this->mutex_);
    sampler_.reset();
  }

  optional<std::vector<size_t>> next(size_t batch_size) override {
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
  std::mutex mutex_;
  OriginalSampler sampler_;
};

/// Simply return batch_size as a single index item with each next call.
class BatchSizeSampler : public samplers::Sampler {
 public:
  void reset() override {}

  c10::optional<std::vector<size_t>> next(size_t batch_size) override {
    return {{batch_size}};
  }

  void save(torch::serialize::OutputArchive& archive) const override {}

  void load(torch::serialize::InputArchive& archive) override {}
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
