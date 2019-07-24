#include <torch/csrc/api/include/torch/data/init.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/data/datasets/chunk.h>
#include <test/cpp/api/dataloader.h>

#include <c10/macros/Export.h>
#include <caffe2/serialize/inline_container.h>

#include <ATen/core/function_schema.h>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch {
namespace data {

using namespace torch::data;
using namespace torch::data::datasets;
using namespace torch::serialize;

void init_dataset_bindings(PyObject* module) {
  // Getting reference to python submodule
  auto m = py::handle(module).cast<py::module>();
  auto data = m.def_submodule("data");
  auto chunk = data.def_submodule("chunk");

  /// ChunkDatasetOptions
  py::class_<ChunkDatasetOptions>(chunk, "ChunkDatasetOptions")
      .def(
          py::init<size_t, size_t, size_t>(),
          "Create and return a new ChunkDatasetOptions instance",
          py::arg("preloader_count"),
          py::arg("batch_size"),
          py::arg("cache_size") = 2048)
      .def_readwrite(
          "preloader_count",
          &ChunkDatasetOptions::preloader_count_,
          "Number of preloader threads")
      .def_readwrite(
          "batch_size", &ChunkDatasetOptions::batch_size_, "Batch size")
      .def_readwrite(
          "cache_size",
          &ChunkDatasetOptions::cache_size_,
          "Cache size to be preloaded before batching");

  /// ChunkDataReader for all standard types
  /// (uint8_t, int8_t, int16_t, int32_t, int64_t, float, double)
  py::class_<ChunkDataReader<uint8_t>, PyChunkDataReader<uint8_t>>(
      chunk,
      "ChunkDataReaderUint8T",
      "Chunk reader performs data chunking and reading of entire chunks with uint8_t data")
      .def(
          "read_chunk",
          &ChunkDataReader<uint8_t>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<uint8_t>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<uint8_t>::reset,
          "Resets any internal state associate with this reader");
  py::class_<ChunkDataReader<int8_t>, PyChunkDataReader<int8_t>>(
      chunk,
      "ChunkDataReaderInt8T",
      "Chunk reader performs data chunking and reading of entire chunks with int8_t data")
      .def(
          "read_chunk",
          &ChunkDataReader<int8_t>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<int8_t>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<int8_t>::reset,
          "Resets any internal state associate with this reader");
  py::class_<ChunkDataReader<int16_t>, PyChunkDataReader<int16_t>>(
      chunk,
      "ChunkDataReaderInt16T",
      "Chunk reader performs data chunking and reading of entire chunks with int16_t data")
      .def(
          "read_chunk",
          &ChunkDataReader<int16_t>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<int16_t>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<int16_t>::reset,
          "Resets any internal state associate with this reader");
  py::class_<ChunkDataReader<int32_t>, PyChunkDataReader<int32_t>>(
      chunk,
      "ChunkDataReaderInt32T",
      "Chunk reader performs data chunking and reading of entire chunks with int32_t data")
      .def(
          "read_chunk",
          &ChunkDataReader<int32_t>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<int32_t>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<int32_t>::reset,
          "Resets any internal state associate with this reader");
  py::class_<ChunkDataReader<int64_t>, PyChunkDataReader<int64_t>>(
      chunk,
      "ChunkDataReaderInt64T",
      "Chunk reader performs data chunking and reading of entire chunks with int64_t data")
      .def(
          "read_chunk",
          &ChunkDataReader<int64_t>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<int64_t>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<int64_t>::reset,
          "Resets any internal state associate with this reader");
  py::class_<ChunkDataReader<float>, PyChunkDataReader<float>>(
      chunk,
      "ChunkDataReaderFloat",
      "Chunk reader performs data chunking and reading of entire chunks with float data")
      .def(
          "read_chunk",
          &ChunkDataReader<float>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<float>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<float>::reset,
          "Resets any internal state associate with this reader");
  py::class_<ChunkDataReader<double>, PyChunkDataReader<double>>(
      chunk,
      "ChunkDataReaderDouble",
      "Chunk reader performs data chunking and reading of entire chunks with double data")
      .def(
          "read_chunk",
          &ChunkDataReader<double>::read_chunk,
          "Read an entire chunk",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &ChunkDataReader<double>::chunk_count,
          "Returns the number of chunks available in this reader")
      .def(
          "reset",
          &ChunkDataReader<double>::reset,
          "Resets any internal state associate with this reader");
}

} // namespace data
} // namespace torch
