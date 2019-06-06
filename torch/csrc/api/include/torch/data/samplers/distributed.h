#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/data/samplers/base.h>

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

/// A `Sampler` that selects a subset of indices to sample from and defines a
/// sampling behavior. In a distributed setting, this selects a subset of the
/// indices depending on the provided num_replicas and rank parameters. The
/// `Sampler` performs a rounding operation based on the `allow_duplicates`
/// parameter to decide the local sample count.
template <typename BatchRequest = std::vector<size_t>>
class DistributedSampler : public Sampler<BatchRequest> {
 public:
  DistributedSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true)
      : size_(size),
        num_replicas_(num_replicas),
        rank_(rank),
        epoch_(0),
        allow_duplicates_(allow_duplicates) {}

  /// Set the epoch for the current enumeration. This can be used to alter the
  /// sample selection and shuffling behavior.
  void set_epoch(size_t epoch) {
    epoch_ = epoch;
  }

  size_t epoch() const {
    return epoch_;
  }

 protected:
  size_t local_sample_count() {
    if (allow_duplicates_) {
      return (size_ + num_replicas_ - 1) / num_replicas_;
    } else {
      return size_ / num_replicas_;
    }
  }

  size_t size_;
  size_t num_replicas_;
  size_t rank_;
  size_t epoch_;
  bool allow_duplicates_;
};

/// Select samples randomly. The sampling order is shuffled at each `reset()`
/// call.
class TORCH_API DistributedRandomSampler : public DistributedSampler<> {
 public:
  DistributedRandomSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// Resets the `DistributedRandomSampler` to a new set of indices.
  void reset(optional<size_t> new_size = nullopt) override;

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `DistributedRandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `DistributedRandomSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `DistributedRandomSampler`.
  size_t index() const noexcept;

 private:
  void populate_indices();

  size_t begin_index_;
  size_t end_index_;
  size_t sample_index_;
  std::vector<size_t> all_indices_;
};

/// Select samples sequentially.
class TORCH_API DistributedSequentialSampler : public DistributedSampler<> {
 public:
  DistributedSequentialSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// Resets the `DistributedSequentialSampler` to a new set of indices.
  void reset(optional<size_t> new_size = nullopt) override;

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `DistributedSequentialSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `DistributedSequentialSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `DistributedSequentialSampler`.
  size_t index() const noexcept;

 private:
  void populate_indices();

  size_t begin_index_;
  size_t end_index_;
  size_t sample_index_;
  std::vector<size_t> all_indices_;
};

} // namespace samplers
} // namespace data
} // namespace torch
