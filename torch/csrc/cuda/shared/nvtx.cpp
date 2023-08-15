#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#ifndef FBCODE_CAFFE2
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("rangeStartA", nvtxRangeStartA);
  nvtx.def("rangeEnd", nvtxRangeEnd);
  nvtx.def("markA", nvtxMarkA);
}

} // namespace torch::cuda::shared
