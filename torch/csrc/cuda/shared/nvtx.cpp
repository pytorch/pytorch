#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#include <nvToolsExt.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace cuda {
namespace shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("rangeStartA", nvtxRangeStartA);
  nvtx.def("rangeEnd", nvtxRangeEnd);
  nvtx.def("markA", nvtxMarkA);
}

} // namespace shared
} // namespace cuda
} // namespace torch
