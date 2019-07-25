#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/data/datasets/chunk.h>
namespace py = pybind11;

namespace torch {
namespace data {

/// ChunkDataset API bindings
void init_dataset_bindings(PyObject* module);

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

/* Generates Python bindings for a specific ChunkDataReader implementation
 *
 * A reference for the binded class is returned so that its constructors
 * are manually binded by the user. This is necessary as the constructor
 * signature are implementation specific
 */
template <typename SpecificChunkDataReader>
py::class_<SpecificChunkDataReader>& bind_chunkdatareader(
    py::module& m,
    const std::string& typestr) {
  // SpecificChunkDataReader
  return py::class_<SpecificChunkDataReader>(
             m,
             typestr.c_str(),
             "Performs data chunking and reading of entire chunks")
      .def(
          "read_chunk",
          &SpecificChunkDataReader::read_chunk,
          "Returns data from a chunk file",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &SpecificChunkDataReader::chunk_count,
          "Returns the number of chunks")
      .def("reset", &SpecificChunkDataReader::reset, "Not used");
}

/* Generates Python bindings for a specific ChunkDataset implementation */
template <typename SpecificChunkDataReader>
void bind_chunkdataset(py::module& m, const std::string& typestr) {
  // SpecificChunkDataset
  using SpecificChunkDataset = ChunkDataset<
      SpecificChunkDataReader,
      samplers::SamplerWrapper,
      samplers::SamplerWrapper>;
  py::class_<SpecificChunkDataset>(
      m,
      typestr.c_str(),
      "A stateful dataset that supports hierarchical sampling and prefetching of entire chunks."
      " Unlike regular `Dataset`, `ChunkDataset` requires two samplers to operate and keep internal state:"
      " 1) A chunk sampler selects which chunk to load next; and"
      " 2) An example sampler determines the order of `Example`s that are returned in each `get_batch` call.")
      .def(
          py::init([](SpecificChunkDataReader chunk_reader,
                      ChunkDatasetOptions options,
                      bool shuffle_chunks,
                      bool shuffle_samples,
                      size_t num_replicas,
                      size_t rank,
                      bool allow_duplicates) {
            size_t chunk_count = chunk_reader.chunk_count();
            // create chunk sampler.
            std::shared_ptr<torch::data::samplers::DistributedSampler<>>
                chunk_sampler = nullptr;
            if (shuffle_chunks) {
              chunk_sampler = std::make_shared<
                  torch::data::samplers::DistributedRandomSampler>(
                  chunk_count, num_replicas, rank, allow_duplicates);
            } else {
              chunk_sampler = std::make_shared<
                  torch::data::samplers::DistributedSequentialSampler>(
                  chunk_count, num_replicas, rank, allow_duplicates);
            }

            // create example sampler.
            std::shared_ptr<samplers::Sampler<>> example_sampler = nullptr;
            if (shuffle_samples) {
              example_sampler = std::make_shared<samplers::RandomSampler>(1);
            } else {
              example_sampler =
                  std::make_shared<samplers::SequentialSampler>(1);
            }
            return torch::make_unique<datasets::ChunkDataset<
                SpecificChunkDataReader,
                samplers::SamplerWrapper,
                samplers::SamplerWrapper>>(
                std::move(chunk_reader),
                samplers::SamplerWrapper(chunk_sampler),
                samplers::SamplerWrapper(example_sampler),
                options);
          }),
          std::string("Creates and returns a new `" + typestr + "` instance")
              .c_str(),
          py::arg("chunk_reader"),
          py::arg("options"),
          py::arg("shuffle_chunks") = true,
          py::arg("shuffle_samples") = true,
          py::arg("num_replicas") = 1,
          py::arg("rank") = 0,
          py::arg("allow_duplicates") = true)
      .def(
          "get_batch",
          (typename SpecificChunkDataset::BatchType(SpecificChunkDataset::*)(size_t)) &SpecificChunkDataset::get_batch,
          "Returns a batch created from preloaded chunks",
          py::arg("batch_size"),
          py::return_value_policy::take_ownership)
      .def(
          "get_batch",
          (typename SpecificChunkDataset::BatchType(SpecificChunkDataset::*)()) &
              SpecificChunkDataset::get_batch,
          "Returns a batch created from preloaded chunks",
          py::return_value_policy::take_ownership)
      .def(
          "reset",
          &SpecificChunkDataset::reset,
          "Resets any internal state and starts the internal prefetching mechanism for the chunk dataset")
      .def("size", &SpecificChunkDataset::size, "Not used")
      .def(
          "set_chunk_sampler_stride",
          [](SpecificChunkDataset& self, const int& stride) {
            auto chunk_sampler = self.chunk_sampler();
            chunk_sampler.set_current_stride(stride);
          },
          "Set current stride for the internal chunk sampler",
          py::arg("stride"));
}

} // namespace datasets
} // namespace data
} // namespace torch
