#include <torch/csrc/jit/testing/init.h>
#include <pybind11/functional.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
namespace testing {

void initJitTestingBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<FileCheck>(m, "FileCheck")
      .def(py::init<>())
      .def("check", &FileCheck::check)
      .def("check_not", &FileCheck::check_not)
      .def("check_same", &FileCheck::check_same)
      .def("check_next", &FileCheck::check_next)
      .def("check_count", &FileCheck::check_count)
      .def("check_dag", &FileCheck::check_dag)
      .def("run", &FileCheck::run);
}

} // namespace testing
} // namespace jit
} // namespace torch
