#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer (csrc/Module.cpp)
// I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject *module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  shared_ptr_class_<::at::cuda::CUDAGraph>(module, "_CudaGraphBase")
      .def(py::init<>())
      .def("capture_begin",
           &::at::cuda::CUDAGraph::capture_begin,
           py::call_guard<py::gil_scoped_release>(),
           R"(``capture_begin`` begins Cuda graph capture on the current stream.)")
      .def("capture_end",
           &::at::cuda::CUDAGraph::capture_end,
           py::call_guard<py::gil_scoped_release>(),
           R"(``capture_end`` ends Cuda graph capture on the current stream.
           After ``capture_end``, ``replay`` may be called on this instance.)")
      .def("replay",
           &::at::cuda::CUDAGraph::replay,
           py::call_guard<py::gil_scoped_release>(),
           R"(``replay`` replays the Cuda graph captured by this instance.)")
      // reset is called in __del__ on the Python side
      // (see class Graph in torch/cuda/streams.py for reasons and caveats)
      .def("reset",
           &::at::cuda::CUDAGraph::reset,
           py::call_guard<py::gil_scoped_release>(),
           R"(``reset`` deletes the graph currently held by this instance.)");
}
