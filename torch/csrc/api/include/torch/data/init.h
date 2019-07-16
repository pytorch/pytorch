#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/data/datasets/chunk.h>
namespace py = pybind11;

namespace torch {
namespace data {

/// Main function that calls all python ChunkDataset binding functions
void init_dataset_bindings(PyObject* module);

namespace samplers {
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
