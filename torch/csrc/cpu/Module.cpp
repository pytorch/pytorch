#include <ATen/cpu/Utils.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  cpu.def("_is_cpu_support_avx2", at::cpu::is_cpu_support_avx2);
  cpu.def("_is_cpu_support_avx512", at::cpu::is_cpu_support_avx512);
  cpu.def("_is_cpu_support_avx512_vnni", at::cpu::is_cpu_support_avx512_vnni);
  cpu.def("_is_cpu_support_amx_tile", at::cpu::is_cpu_support_amx_tile);
  cpu.def("_is_cpu_support_amx_fp16", at::cpu::is_cpu_support_amx_fp16);
  cpu.def("_init_amx", at::cpu::init_amx);
  cpu.def("_L1d_cache_size", at::cpu::L1d_cache_size);
  cpu.def("_L2_cache_size", at::cpu::L2_cache_size);
}

} // namespace torch::cpu
