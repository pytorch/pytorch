#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/data/datasets/chunk.h>
namespace py = pybind11;

namespace torch {
namespace data {

void init_dataset_bindings(PyObject* module);
void init_dataset_bindings_impl(PyObject* module);
void init_dataset_bindings_test(PyObject* module);

namespace samplers {
class SamplerWrapper : public Sampler<> {
 public:
  explicit SamplerWrapper(
      std::shared_ptr<Sampler<>> sampler,
      size_t total_stride = 0,
      size_t current_stride = 0)
      : sampler_(std::move(sampler)),
        current_stride_(current_stride),
        total_stride_(total_stride) {}
  virtual ~SamplerWrapper() = default;

  /// Resets the underlying sampler to a new set of indices.
  void reset(torch::optional<size_t> new_size = nullopt) override {
    sampler_->reset(new_size);
  };

  /// Sets the underlying sampler to a new stride
  /// WARNING: `current_stride_` is not changed by the sampler `reset` routine
  void set_current_stride(size_t stride) {
    current_stride_ = stride;
  };

  /// Sets the total number of strides for the underlying sampler
  /// WARNING: `total_stride_` is not changed by the sampler `reset` routine
  void set_total_stride(size_t total) {
    total_stride_ = total;
  };

  /// Returns current sampler stride
  size_t current_stride() {
    return current_stride_;
  };

  /// Returns the total number of sampler strides
  size_t total_stride() {
    return total_stride_;
  };

  /// Returns the next batch of indices.
  torch::optional<std::vector<size_t>> next(size_t batch_size) override {
    if (total_stride_ > 0) {
      torch::optional<std::vector<size_t>> indices;
      AT_ASSERT(batch_size == 1);
      while (true) {
        torch::optional<std::vector<size_t>> idx = sampler_->next(batch_size);
        if (!idx) {
          break;
        }
        if (idx.value()[0] % total_stride_ == current_stride_) {
          return idx;
        }
      }
      return indices;
    } else {
      return sampler_->next(batch_size);
    }
  }

  /// Serializes the `RandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override {
    sampler_->save(archive);
  }

  /// Deserializes the `RandomSampler` from the `archive`.
  TORCH_API void load(serialize::InputArchive& archive) override {
    sampler_->load(archive);
  }

 private:
  std::shared_ptr<samplers::Sampler<>> sampler_;
  size_t current_stride_;
  size_t total_stride_;
};

/// This class creates a trampoline for virtual methods of `Sampler`
template <typename BatchRequest = std::vector<size_t>>
class PySampler : public Sampler<BatchRequest> {
 public:
  // Inherit the constructors
  using Sampler<BatchRequest>::Sampler;

  // Trampoline (need one for each virtual function)
  void reset(optional<size_t> new_size) override {
    PYBIND11_OVERLOAD_PURE(void, Sampler<BatchRequest>, reset, new_size);
  }
  optional<BatchRequest> next(size_t batch_size) override {
    PYBIND11_OVERLOAD_PURE(
        optional<BatchRequest>, Sampler<BatchRequest>, next, batch_size);
  }
  void save(serialize::OutputArchive& archive) const override {
    PYBIND11_OVERLOAD_PURE(void, Sampler<BatchRequest>, save, archive);
  }
  void load(serialize::InputArchive& archive) override {
    PYBIND11_OVERLOAD_PURE(void, Sampler<BatchRequest>, load, archive);
  }
};

/// This class creates a trampoline for virtual methods of `DistributedSampler`
template <typename BatchRequest = std::vector<size_t>>
class PyDistributedSampler : public DistributedSampler<BatchRequest> {
 public:
  // Inherit the constructors
  using DistributedSampler<BatchRequest>::DistributedSampler;

  // Trampoline (need one for each virtual function)
  void reset(optional<size_t> new_size) override {
    PYBIND11_OVERLOAD_PURE(
        void, DistributedSampler<BatchRequest>, reset, new_size);
  }
  optional<BatchRequest> next(size_t batch_size) override {
    PYBIND11_OVERLOAD_PURE(
        optional<BatchRequest>,
        DistributedSampler<BatchRequest>,
        next,
        batch_size);
  }
  void save(serialize::OutputArchive& archive) const override {
    PYBIND11_OVERLOAD_PURE(
        void, DistributedSampler<BatchRequest>, save, archive);
  }
  void load(serialize::InputArchive& archive) override {
    PYBIND11_OVERLOAD_PURE(
        void, DistributedSampler<BatchRequest>, load, archive);
  }
};
} // namespace samplers

namespace datasets {
/// This class creates a trampoline for virtual methods of `ChunkDataReader`
template <typename DataType>
class PyChunkDataReader : public ChunkDataReader<DataType> {
 public:
  // Inherit the constructors
  using ChunkDataReader<DataType>::ChunkDataReader;

  // Trampoline (need one for each virtual function)
  typename ChunkDataReader<DataType>::ChunkType read_chunk(
      size_t chunk_index) override {
    PYBIND11_OVERLOAD_PURE(
        typename ChunkDataReader<DataType>::ChunkType,
        ChunkDataReader<DataType>,
        read_chunk,
        chunk_index);
  }
  size_t chunk_count() override {
    PYBIND11_OVERLOAD_PURE(size_t, ChunkDataReader<DataType>, chunk_count);
  }
  void reset() override {
    PYBIND11_OVERLOAD_PURE(void, ChunkDataReader<DataType>, reset);
  }
};

} // namespace datasets
} // namespace data
} // namespace torch
