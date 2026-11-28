#include <ATen/cpu/Utils.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  cpu.def("_release_unused_memory", []() {
    const auto* allocator = c10::GetCPUAllocator();
    // Both built-in allocators ultimately use c10::alloc_cpu/free_cpu. Other
    // allocators may not use the allocator selected by the PyTorch build.
    if (allocator != c10::GetDefaultCPUAllocator() &&
        allocator != c10::GetDefaultMobileCPUAllocator()) {
      return false;
    }
    py::gil_scoped_release no_gil;
    return c10::release_unused_cpu_memory();
  });
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
