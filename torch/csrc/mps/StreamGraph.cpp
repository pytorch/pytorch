//  torch/csrc/mps/StreamGraph.cpp
//
//  pybind11 bindings for at::mps::MPSStreamGraph, exposed to Python as
//  torch._C._MPSStreamGraph and wrapped in torch.mps.MPSGraph for the user
//  API. Pattern mirrors torch/csrc/cuda/Graph.cpp.

#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/utils/pybind.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSStreamGraph.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THMPGraph_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();

  shared_ptr_class_<at::mps::MPSStreamGraph>(torch_C_m, "_MPSStreamGraph")
      .def(py::init<std::size_t>(),
           py::arg("max_commands") = 4096)
      .def(
          "capture_begin",
          [](at::mps::MPSStreamGraph& self) {
            // Use the currently-active MPS stream from the runtime. Matches
            // CUDA's "current stream" convention; user-selected streams via
            // `with torch.mps.stream(s):` work transparently.
            auto* stream = at::mps::getCurrentMPSStream();
            TORCH_CHECK(
                stream != nullptr,
                "MPSGraph.capture_begin: no current MPS stream "
                "(is the MPS backend initialized?)");
            self.capture_begin(stream);
          })
      .def("capture_end",
           [](at::mps::MPSStreamGraph& self) { self.capture_end(); })
      .def("replay", [](at::mps::MPSStreamGraph& self) { self.replay(); })
      .def("num_commands",
           [](const at::mps::MPSStreamGraph& self) {
             return self.num_commands();
           })
      .def("is_capturing",
           [](const at::mps::MPSStreamGraph& self) {
             return self.is_capturing();
           })
      .def("is_ready",
           [](const at::mps::MPSStreamGraph& self) {
             return self.is_ready();
           });
}
