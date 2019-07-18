#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/data/samplers.h>
#include <torch/types.h>

namespace torch {
namespace data {
namespace samplers {

/**
 * Implementation of a wrapper around Sampler classes
 *
 * This is needed for generating Python bindings, only.
 * On C++ implementation, a single process share the same ChunkDataset
 * class among all threads, so sampling is straingthforward.
 *
 * On Python implementation, on the other hand, DataLoader uses
 * multi-processing instead of multi-threading for parallelism.
 * Each Python Dataloader is a separate process with a copy of
 * ChunkDataset library (which includes samplers). To prevent different
 * processes to read the same data in parallel, sampler strides are needed
 * to coordinate workers. Each instance of SamplerWrapper must be configured
 * with different strides, so that sampling happens in a round-robin fashion:
 * stride0, stride1, ..., strideN, stride0, stride1, ..., strideN, ...
 *
 * For example, assume 2 Python Dataloader workers reading
 * the same ChunkDataset. Each worker needs to reset their
 * ChunkDataset instance so that one of them reads all even (stride 0)
 * batches while the other reads all odd batches (stride 1).
 *
 * NOTE: To preserve back compatibility with C++ implementation,
 * which doesn't need stride support due to multi-threading,
 * `reset()` method doesn't reset stride setting.
 */
class SamplerWrapper : public Sampler<> {
 public:
  explicit SamplerWrapper(
      std::shared_ptr<Sampler<>> sampler,
      size_t total_strides = 0,
      size_t current_stride = 0)
      : sampler_(std::move(sampler)),
        current_stride_(current_stride),
        total_strides_(total_strides) {}
  virtual ~SamplerWrapper() = default;

  /// Resets the underlying sampler to a new set of indices.
  void reset(torch::optional<size_t> new_size = nullopt) override {
    sampler_->reset(new_size);
  };

  /// Sets the underlying sampler to a new stride
  /// Stride depends on the number of used python Dataloader workers
  /// so `reset()` method will not change this parameter
  void set_current_stride(size_t stride) {
    current_stride_ = stride;
  };

  /// Sets the total number of strides for the underlying sampler
  /// Total strides depend on the number of used python Dataloader workers
  /// so `reset()` method will not change this parameter
  void set_total_strides(size_t total) {
    total_strides_ = total;
  };

  /// Returns current sampler stride
  size_t current_stride() {
    return current_stride_;
  };

  /// Returns the total number of sampler strides
  size_t total_strides() {
    return total_strides_;
  };

  /// Returns the next batch of indices.
  torch::optional<std::vector<size_t>> next(size_t batch_size) override {
    if (total_strides_ > 0) {
      torch::optional<std::vector<size_t>> indices;
      AT_ASSERT(batch_size == 1);
      while (true) {
        torch::optional<std::vector<size_t>> idx = sampler_->next(batch_size);
        if (!idx) {
          break;
        }
        if (idx.value()[0] % total_strides_ == current_stride_) {
          return idx;
        }
      }
      return indices;
    } else {
      return sampler_->next(batch_size);
    }
  }

  /// Serializes the internal `Sampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override {
    sampler_->save(archive);
  }

  /// Deserializes the internal `Sampler` from the `archive`.
  TORCH_API void load(serialize::InputArchive& archive) override {
    sampler_->load(archive);
  }

 private:
  std::shared_ptr<samplers::Sampler<>> sampler_;
  size_t current_stride_;
  size_t total_strides_;
};
} // namespace samplers

} // namespace data
} // namespace torch
