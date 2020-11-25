#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer (csrc/Module.cpp)
// I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject *module) {
  auto torch_C_m = py::handle(module).cast<py::module>();

  shared_ptr_class_<::at::cuda::CudaGraph>(module, "_CudaGraphBase")
      .def(py::init<>()),
      .def("capture_begin",
           &::at::cuda::CudaGraph::capture_end,
           py::call_guard<py::gil_scoped_release>(),
           R"(``capture_begin`` begins Cuda graph capture on the current stream.)")
      .def("capture_end",
           &::at::cuda::CudaGraph::capture_end,
           py::call_guard<py::gil_scoped_release>(),
           R"(``capture_end`` ends Cuda graph capture on the current stream.
           After ``capture_end``, ``replay`` may be called on this instance.)")
      .def("replay",
           &::at::cuda::CudaGraph::replay,
           py::call_guard<py::gil_scoped_release>(),
           R"(``replay`` replays the Cuda graph captured by this instance.)");
      .def("drop_graph",
           &::at::cuda::CudaGraph::drop_graph,
           py::call_guard<py::gil_scoped_release>(),
           R"(``drop_graph`` deletes the graph currently held by this instance.)");
}
