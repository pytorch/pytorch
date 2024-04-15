#include <ATen/cpu/Utils.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  cpu.def("_is_cpu_support_vnni", at::cpu::is_cpu_support_vnni);
}

} // namespace cpu
} // namespace torch
