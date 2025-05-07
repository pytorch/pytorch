#include <torch/csrc/itt.h>
#include <torch/csrc/itt_wrapper.h>

namespace torch::profiler {
void initIttBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto itt = m.def_submodule("_itt", "VTune ITT bindings");
  itt.def("is_available", itt_is_available);
  itt.def("rangePush", itt_range_push);
  itt.def("rangePop", itt_range_pop);
  itt.def("mark", itt_mark);
}
} // namespace torch::profiler
