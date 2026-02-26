#include <ATen/cpu/Utils.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  cpu.def("_init_amx", at::cpu::init_amx);
  cpu.def("_get_cpu_capability", []() {
    py::dict result;
    for (auto& [key, val] : at::cpu::get_cpu_capabilities()) {
      result[py::str(key)] = torch::jit::toPyObject(std::move(val));
    }
    return result;
  });
}

} // namespace torch::cpu
