#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#ifdef TORCH_CUDA_USE_NVTX3
#include <nvtx3/nvtx3.hpp>
#else
#include <nvToolsExt.h>
#endif
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#ifdef TORCH_CUDA_USE_NVTX3
  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
#else
  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
#endif
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("rangeStartA", nvtxRangeStartA);
  nvtx.def("rangeEnd", nvtxRangeEnd);
  nvtx.def("markA", nvtxMarkA);
}

} // namespace torch::cuda::shared
