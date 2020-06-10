#include <torch/csrc/utils.h>
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
          py::object py_devices,
          size_t buffer_size) {
          auto devices = THPUtils_unpackPySequence_to_DeviceList(py_devices.ptr());
          // Note: We're holding the GIL up to here.
          pybind11::gil_scoped_release no_gil;
         return broadcast_coalesced(tensors, devices, buffer_size);
       },
       py::arg("tensors"),
       py::arg("devices"),
       py::arg("buffer_size"))
      .def(
          "_broadcast",
          [](at::Tensor& tensor, py::object py_devices) {
            auto devices = THPUtils_unpackPySequence_to_DeviceList(py_devices.ptr());
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return broadcast(tensor, devices);
          },
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
             py::object py_devices,
             const c10::optional<std::vector<int64_t>>& chunk_sizes,
             int64_t dim,
             c10::optional<py::object> py_streams) {
            auto devices = THPUtils_unpackPySequence_to_DeviceList(py_devices.ptr());
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
             const c10::optional<py::object>& py_streams) {
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
          "_reduce",
          [](std::vector<at::Tensor>& tensors,
             torch::utils::comm::ReduceOp op,
             const c10::optional<py::object>& py_device) {
            c10::optional<at::Device> device;
            if (py_device) {
              device = THPUtils_unpackDevice(py_device->ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return reduce(tensors, op, device);
          },
          py::arg("tensors"),
          py::arg("op"),
          py::arg("destination"))
      .def(
          "_reduce_out",
          [](std::vector<at::Tensor>& tensors,
             at::Tensor& out_tensor,
             torch::utils::comm::ReduceOp op) {
            return reduce_out(tensors, out_tensor, op);
          },
          py::arg("tensors"),
          py::arg("out"),
          py::arg("op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_gather",
          [](std::vector<at::Tensor>& tensors,
             int64_t dim,
             const c10::optional<py::object>& py_device) {
            c10::optional<at::Device> device;
            if (py_device) {
              device = THPUtils_unpackDevice(py_device->ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return gather(tensors, dim, device);
          },
          py::arg("tensors"),
          py::arg("dim"),
          py::arg("destination"))
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
