#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/cuda/comm.h"

#include <chrono>

namespace torch { namespace cuda { namespace python {

void initCommMethods(PyObject *module) {
  auto m = py::cast<py::module>(module);
  m.def("_broadcast_coalesced", [](std::vector<at::Tensor>& tensors, std::vector<int64_t> devices, std::size_t buffer_size) {
     return broadcast_coalesced(tensors, devices, buffer_size);
   }, py::arg("tensors"), py::arg("devices"), py::arg("buffer_size"),
      py::call_guard<py::gil_scoped_release>())
   .def("_broadcast", [](at::Tensor& tensor, std::vector<int64_t> devices) {
     return broadcast(tensor, devices);
   }, py::call_guard<py::gil_scoped_release>());
}

}}}
