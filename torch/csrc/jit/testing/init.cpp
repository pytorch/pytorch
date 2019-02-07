#include <torch/csrc/jit/testing/init.h>
#include <pybind11/functional.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
namespace testing {

void initJitTestingBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<testing::FileCheck>(m, "FileCheck")
      .def_static("run", &testing::FileCheck::checkFile);
}

} // namespace testing
} // namespace jit
} // namespace torch
