#include <torch/csrc/utils/pybind.h>
#include <roctx.h>

namespace torch { namespace cuda { namespace shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
  nvtx.def("rangePushA", roctxRangePushA);
  nvtx.def("rangePop", roctxRangePop);
  nvtx.def("markA", roctxMarkA);
}

} // namespace shared
} // namespace cuda
} // namespace torch
