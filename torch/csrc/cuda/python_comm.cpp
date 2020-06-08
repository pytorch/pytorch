#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/cuda/comm.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/cuda/THCP.h>
#include <pybind11/pybind11.h>
#include <ATen/core/functional.h>

#include <ATen/ATen.h>

#include <THC/THC.h>

#include <cstddef>
#include <vector>

namespace torch { namespace cuda { namespace python {
void initCommMethods(PyObject *module) {
  auto m = py::cast<py::module>(module);
  m.def(
       "_broadcast_coalesced",
       [](std::vector<at::Tensor>& tensors,
          std::vector<int64_t> devices,
          size_t buffer_size) {
         return broadcast_coalesced(tensors, devices, buffer_size);
       },
       py::arg("tensors"),
       py::arg("devices"),
       py::arg("buffer_size"),
       py::call_guard<py::gil_scoped_release>())
      .def(
          "_broadcast",
          [](at::Tensor& tensor, std::vector<int64_t> devices) {
            return broadcast(tensor, devices);
          },
          py::call_guard<py::gil_scoped_release>(),
          py::arg("tensor"),
          py::arg("devices"))
      .def(
          "_broadcast_out",
          [](at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) {
            return broadcast_out(tensor, out_tensors);
          },
          py::call_guard<py::gil_scoped_release>(),
          py::arg("tensor"),
          py::arg("out"))
      .def(
          "_scatter",
          [](at::Tensor& tensor,
             std::vector<int64_t>& devices,
             c10::optional<std::vector<int64_t>> chunk_sizes,
             int64_t dim,
             c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>> streams;
            if (py_streams) {
              py::handle handle = *py_streams;
              streams = THPUtils_PySequence_to_CUDAStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return scatter(tensor, devices, chunk_sizes, dim, streams);
          },
          py::arg("tensor"),
          py::arg("devices"),
          py::arg("chunk_sizes"),
          py::arg("dim"),
          py::arg("streams"))
      .def(
          "_scatter_out",
          [](at::Tensor& tensor,
             std::vector<at::Tensor>& out_tensors,
             int64_t dim,
             c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>> streams;
            if (py_streams) {
              py::handle handle = *py_streams;
              streams = THPUtils_PySequence_to_CUDAStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return scatter_out(tensor, out_tensors, dim, streams);
          },
          py::arg("tensor"),
          py::arg("out"),
          py::arg("dim"),
          py::arg("streams"))
      .def(
          "_gather",
          [](std::vector<at::Tensor>& tensors,
             int64_t dim,
             c10::optional<int32_t> destination_index) {
            return gather(tensors, dim, destination_index);
          },
          py::arg("tensors"),
          py::arg("dim"),
          py::arg("destination_index"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_gather_out",
          [](std::vector<at::Tensor>& tensors,
             at::Tensor& out_tensor,
             int64_t dim) {
            return gather_out(tensors, out_tensor, dim);
          },
          py::arg("tensors"),
          py::arg("out"),
          py::arg("dim"),
          py::call_guard<py::gil_scoped_release>());
}
}}}
