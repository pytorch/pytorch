#if USE_DISTRIBUTED
#include <chrono>

#include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "distributed/c10d/ProcessGroupOCCL.hpp"

namespace py = pybind11;

void initProcessGroupBindings(py::module& m) {
  py::class_<c10d::ProcessGroupOCCL, c10d::Backend, c10::intrusive_ptr<c10d::ProcessGroupOCCL>>(m, "ProcessGroupOCCL")
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& /*store*/,
                      int rank,
                      int size,
                      std::chrono::milliseconds /*timeout*/) {
            return c10::make_intrusive<::c10d::ProcessGroupOCCL>(rank, size);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(30 * 60 * 1000));
}
#endif // USE_DISTRIBUTED
