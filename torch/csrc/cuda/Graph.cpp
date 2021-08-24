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

  torch_C_m
      .def("graph_pool_handle",
           &::at::cuda::graph_pool_handle,
           R"(
.. warning
    This API is a prototype and may change in future releases.

Returns an opaque token representing the id of a graph memory pool.
See :ref:`Graph memory management<graph-memory-management>`.
           )");

  shared_ptr_class_<::at::cuda::CUDAGraph>
      (torch_C_m,
       "CUDAGraph",
       R"(
.. warning
    This API is a prototype and may change in future releases.

Wrapper around a CUDA graph.
       )")
      .def(py::init<>())
      // I'm not sure this is the correct order of all the arguments. Pybind11 docs
      // aren't clear. But it works.
      .def("capture_begin",
           &::at::cuda::CUDAGraph::capture_begin,
           py::call_guard<py::gil_scoped_release>(),
           R"(
Begins capturing CUDA work on the current stream.

Typically, you shouldn't call ``capture_begin`` yourself.
Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables,
which call ``capture_begin`` internally.

Arguments:
    pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
           )",
           py::arg("pool") = c10::cuda::MempoolId_t{0, 0})
      .def("capture_end",
           &::at::cuda::CUDAGraph::capture_end,
           py::call_guard<py::gil_scoped_release>(),
           R"(
Ends CUDA graph capture on the current stream.
After ``capture_end``, ``replay`` may be called on this instance.

Typically, you shouldn't call ``capture_end`` yourself.
Use :class:`~torch.cuda.graph` or :func:`~torch.cuda.make_graphed_callables,
which call ``capture_end`` internally.
           )")
      .def("replay",
           &::at::cuda::CUDAGraph::replay,
           py::call_guard<py::gil_scoped_release>(),
           R"(Replays the CUDA work captured by this graph.)")
      // reset is called in __del__ on the Python side
      // (see class Graph in torch/cuda/streams.py for reasons and caveats)
      .def("reset",
           &::at::cuda::CUDAGraph::reset,
           py::call_guard<py::gil_scoped_release>(),
           R"(Deletes the graph currently held by this instance.)")
      .def("pool",
           &::at::cuda::CUDAGraph::pool,
           py::call_guard<py::gil_scoped_release>(),
           R"(
Returns an opaque token representing the id of this graph's memory pool.
This id can optionally be passed to another graph's ``capture_begin``,
which hints the other graph may share the same memory pool.
           )");
}
