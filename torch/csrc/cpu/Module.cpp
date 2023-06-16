#include <ATen/cpu/CPUUtils.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("cpu", "cpu related pybind.");
  cpu.def("is_cpu_support_vnni", at::cpu::is_cpu_support_vnni);
}

} // namespace cpu
} // namespace torch
