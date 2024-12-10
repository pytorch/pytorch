#include <ATen/cpu/Utils.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  cpu.def("_is_avx2_supported", at::cpu::is_avx2_supported);
  cpu.def("_is_avx512_supported", at::cpu::is_avx512_supported);
  cpu.def("_is_avx512_vnni_supported", at::cpu::is_avx512_vnni_supported);
  cpu.def("_is_avx512_bf16_supported", at::cpu::is_avx512_bf16_supported);
  cpu.def("_is_amx_tile_supported", at::cpu::is_amx_tile_supported);
  cpu.def("_is_amx_fp16_supported", at::cpu::is_amx_fp16_supported);
  cpu.def("_init_amx", at::cpu::init_amx);
  cpu.def("_is_arm_sve_supported", at::cpu::is_arm_sve_supported);
  cpu.def("_get_max_arm_sve_length", at::cpu::get_max_arm_sve_length);
  cpu.def("_L1d_cache_size", at::cpu::L1d_cache_size);
  cpu.def("_L2_cache_size", at::cpu::L2_cache_size);
}

} // namespace torch::cpu
