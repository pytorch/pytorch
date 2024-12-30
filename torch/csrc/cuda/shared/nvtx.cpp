#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#ifdef TORCH_CUDA_USE_NVTX3
#include <nvtx3/nvtx3.hpp>
#else
#include <nvToolsExt.h>
#endif
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

struct RangeHandle {
  nvtxRangeId_t id;
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  nvtxRangeEnd(handle->id);
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_nvtxRangeEnd(void* handle, std::intptr_t stream) {
  cudaLaunchHostFunc((cudaStream_t)stream, device_callback_range_end, handle);
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  handle->id = nvtxRangeStartA(handle->msg);
}

static void* device_nvtxRangeStart(const char* msg, std::intptr_t stream) {
  RangeHandle* handle = (RangeHandle*)calloc(sizeof(RangeHandle), 1);
  handle->msg = strdup(msg);
  handle->id = 0;
  cudaLaunchHostFunc(
      (cudaStream_t)stream, device_callback_range_start, (void*)handle);
  return handle;
}

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
  nvtx.def("deviceRangeStart", device_nvtxRangeStart);
  nvtx.def("deviceRangeEnd", device_nvtxRangeEnd);
}

} // namespace torch::cuda::shared
