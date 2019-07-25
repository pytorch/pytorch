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
}

} // namespace data
} // namespace torch
