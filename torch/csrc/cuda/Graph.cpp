#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);

  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<>())
      // I'm not sure this is the correct order of all the arguments. Pybind11
      // docs aren't clear. But it works.
      .def(
          "capture_begin",
          torch::wrap_pybind_function(&at::cuda::CUDAGraph::capture_begin),
          py::call_guard<py::gil_scoped_release>(),
          py::arg("pool") = c10::cuda::MempoolId_t{0, 0})
      .def(
          "capture_end",
          torch::wrap_pybind_function(&at::cuda::CUDAGraph::capture_end),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "replay",
          torch::wrap_pybind_function(&at::cuda::CUDAGraph::replay),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reset",
          torch::wrap_pybind_function(&at::cuda::CUDAGraph::reset),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "pool",
          torch::wrap_pybind_function(&at::cuda::CUDAGraph::pool),
          py::call_guard<py::gil_scoped_release>());
}
