#include <torch/csrc/api/include/torch/data/init.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/data/datasets/chunk.h>

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
using namespace torch::data::samplers;
using namespace torch::serialize;

void initDataBindings(PyObject* module) {
  // Getting reference to python submodule
  auto m = py::handle(module).cast<py::module>();
  auto data = m.def_submodule("data");

  /// ChunkDatasetOptions
  py::class_<ChunkDatasetOptions>(data, "ChunkDatasetOptions")
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

  /// Sampler
  py::class_<Sampler<>, std::shared_ptr<Sampler<>>, PySampler<>> sampler(
      data, "Sampler");
  sampler.def(py::init<>(), "Create and return a new `Sampler` instance");
  sampler.def(
      "reset",
      &Sampler<>::reset,
      "Resets the `Sampler`'s internal state. Typically called before a new epoch. Optionally, accepts a new size when reseting the sampler",
      py::arg("new_size"));
  sampler.def(
      "next",
      &Sampler<>::next,
      "Returns the next index if possible, or an empty optional if the sampler is exhausted for this epoch",
      py::arg("batch_size"));
  sampler.def(
      "save",
      &Sampler<>::save,
      "Serializes the `RandomSampler` to the `archive`",
      py::arg("archive"));
  sampler.def(
      "load",
      &Sampler<>::load,
      "Deserializes the `Sampler` from the `archive`",
      py::arg("archive"));

  /// SamplerWrapper
  py::class_<SamplerWrapper, std::shared_ptr<SamplerWrapper>> sampler_wrapper(
      data, "SamplerWrapper", sampler);
  sampler_wrapper.def(
      py::init<std::shared_ptr<Sampler<>>, size_t>(),
      "Create and return a new `SamplerWrapper` instance",
      py::arg("sampler"),
      py::arg("stride") = 0);
  sampler_wrapper.def(
      "reset",
      &SamplerWrapper::reset,
      "Resets the `Sampler`'s internal state. Typically called before a new epoch. Optionally, accepts a new size when reseting the sampler",
      py::arg("new_size"));
  sampler_wrapper.def(
      "next",
      &SamplerWrapper::next,
      "Returns the next index if possible, or an empty optional if the sampler is exhausted for this epoch",
      py::arg("batch_size"));
  sampler_wrapper.def(
      "save",
      &SamplerWrapper::save,
      "Serializes the `RandomSampler` to the `archive`",
      py::arg("archive"));
  sampler_wrapper.def(
      "load",
      &SamplerWrapper::load,
      "Deserializes the `Sampler` from the `archive`",
      py::arg("archive"));
  sampler_wrapper.def(
      "set_current_stride",
      &SamplerWrapper::set_current_stride,
      "Sets the underlying sampler to a new stride",
      py::arg("stride"));
  sampler_wrapper.def(
      "set_total_stride",
      &SamplerWrapper::set_total_stride,
      "Sets the total number of strides for the underlying sampler",
      py::arg("total"));
  sampler_wrapper.def(
      "stride",
      &SamplerWrapper::current_stride,
      "Returns current sampler stride");
  sampler_wrapper.def(
      "total_stride",
      &SamplerWrapper::total_stride,
      "Returns the total number of sampler strides");

  /// DistributedSampler
  py::class_<DistributedSampler<>, PyDistributedSampler<>> distributed_sampler(
      data, "DistributedSampler", sampler);
  distributed_sampler.def(
      py::init<size_t, size_t, size_t, bool>(),
      "Create and return a new `DistributedSampler` instance",
      py::arg("size"),
      py::arg("num_replicas") = 1,
      py::arg("rank") = 0,
      py::arg("allow_duplicates") = true);
  distributed_sampler.def(
      "set_epoch",
      &DistributedSampler<>::set_epoch,
      "Set the epoch for the current enumeration. This can be used to alter the sample selection and shuffling behavior",
      py::arg("epoch"));
  distributed_sampler.def(
      "epoch", &DistributedSampler<>::epoch, "Returns the current epoch");

  /// DistributedRandomSampler
  py::class_<DistributedRandomSampler>(
      data, "DistributedRandomSampler", distributed_sampler)
      .def(
          py::init<size_t, size_t, size_t, bool>(),
          "Create and return a new `DistributedRandomSampler` instance",
          py::arg("size"),
          py::arg("num_replicas") = 1,
          py::arg("rank") = 0,
          py::arg("allow_duplicates") = true)
      .def(
          "reset",
          &DistributedRandomSampler::reset,
          "Resets the `Sampler`'s internal state. Typically called before a new epoch. Optionally, accepts a new size when reseting the sampler",
          py::arg("new_size"))
      .def(
          "next",
          &DistributedRandomSampler::next,
          "Returns the next index if possible, or an empty optional if the sampler is exhausted for this epoch",
          py::arg("batch_size"))
      .def(
          "save",
          &DistributedRandomSampler::save,
          "Serializes the `RandomSampler` to the `archive`",
          py::arg("archive"))
      .def(
          "load",
          &DistributedRandomSampler::load,
          "Deserializes the `Sampler` from the `archive`",
          py::arg("archive"))
      .def(
          "index",
          &DistributedRandomSampler::index,
          "Returns the current index of the `DistributedRandomSampler`");

  /// DistributedSequentialSampler
  py::class_<DistributedSequentialSampler>(
      data, "DistributedSequentialSampler", distributed_sampler)
      .def(
          py::init<size_t, size_t, size_t, bool>(),
          "Create and return a new `DistributedSequentialSampler` instance",
          py::arg("size"),
          py::arg("num_replicas") = 1,
          py::arg("rank") = 0,
          py::arg("allow_duplicates") = true)
      .def(
          "reset",
          &DistributedSequentialSampler::reset,
          "Resets the `Sampler`'s internal state. Typically called before a new epoch. Optionally, accepts a new size when reseting the sampler",
          py::arg("new_size"))
      .def(
          "next",
          &DistributedSequentialSampler::next,
          "Returns the next index if possible, or an empty optional if the sampler is exhausted for this epoch",
          py::arg("batch_size"))
      .def(
          "save",
          &DistributedSequentialSampler::save,
          "Serializes the `RandomSampler` to the `archive`",
          py::arg("archive"))
      .def(
          "load",
          &DistributedSequentialSampler::load,
          "Deserializes the `Sampler` from the `archive`",
          py::arg("archive"))
      .def(
          "index",
          &DistributedSequentialSampler::index,
          "Returns the current index of the `DistributedSequentialSampler`");

  /// RandomSampler
  py::class_<RandomSampler>(data, "RandomSampler", distributed_sampler)
      .def(
          py::init<int64_t, at::ScalarType>(),
          "Create and return a new `RandomSampler` instance",
          py::arg("size"),
          py::arg("index_dtype"))
      .def(
          "reset",
          &RandomSampler::reset,
          "Resets the `Sampler`'s internal state. Typically called before a new epoch. Optionally, accepts a new size when reseting the sampler",
          py::arg("new_size"))
      .def(
          "next",
          &RandomSampler::next,
          "Returns the next index if possible, or an empty optional if the sampler is exhausted for this epoch",
          py::arg("batch_size"))
      .def(
          "save",
          &RandomSampler::save,
          "Serializes the `RandomSampler` to the `archive`",
          py::arg("archive"))
      .def(
          "load",
          &RandomSampler::load,
          "Deserializes the `Sampler` from the `archive`",
          py::arg("archive"))
      .def(
          "index",
          &RandomSampler::index,
          "Returns the current index of the `RandomSampler`");

  /// SequentialSampler
  py::class_<SequentialSampler>(data, "SequentialSampler", distributed_sampler)
      .def(
          py::init<size_t>(),
          "Create and return a new `SequentialSampler` instance",
          py::arg("size"))
      .def(
          "reset",
          &SequentialSampler::reset,
          "Resets the `Sampler`'s internal state. Typically called before a new epoch. Optionally, accepts a new size when reseting the sampler",
          py::arg("new_size"))
      .def(
          "next",
          &SequentialSampler::next,
          "Returns the next index if possible, or an empty optional if the sampler is exhausted for this epoch",
          py::arg("batch_size"))
      .def(
          "save",
          &SequentialSampler::save,
          "Serializes the `RandomSampler` to the `archive`",
          py::arg("archive"))
      .def(
          "load",
          &SequentialSampler::load,
          "Deserializes the `Sampler` from the `archive`",
          py::arg("archive"))
      .def(
          "index",
          &SequentialSampler::index,
          "Returns the current index of the `SequentialSampler`");

  /// ChunkDataReader
  py::class_<ChunkDataReader<uint8_t>, PyChunkDataReader<uint8_t>>(
      data,
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
      data,
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
      data,
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
      data,
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
      data,
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
      data,
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
      data,
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
