#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#if !defined(USE_ROCM)
#include <cuda_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace torch {
namespace cuda {
namespace shared {

#ifdef USE_ROCM
namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace
#endif

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaOutputMode_t is used in cudaProfilerInitialize only. The latter is gone
  // in CUDA 12.
  py::enum_<cudaOutputMode_t>(
      cudart,
      "cuda"
      "OutputMode")
      .value("KeyValuePair", cudaKeyValuePair)
      .value("CSV", cudaCSV);
#endif

  py::enum_<cudaError_t>(
      cudart,
      "cuda"
      "Error")
      .value("success", cudaSuccess);

  cudart.def(
      "cuda"
      "GetErrorString",
      cudaGetErrorString);
  cudart.def(
      "cuda"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStart
#endif
  );
  cudart.def(
      "cuda"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStop
#endif
  );
  cudart.def(
      "cuda"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(
            cudaHostRegister((void*)ptr, size, flags));
      });
  cudart.def(
      "cuda"
      "HostUnregister",
      [](uintptr_t ptr) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(cudaHostUnregister((void*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamCreate",
      [](uintptr_t ptr) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(cudaStreamCreate((cudaStream_t*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamDestroy",
      [](uintptr_t ptr) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(cudaStreamDestroy((cudaStream_t)ptr));
      });
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  // cudaProfilerInitialize is no longer needed after CUDA 12:
  // https://forums.developer.nvidia.com/t/cudaprofilerinitialize-is-deprecated-alternative/200776/3
  cudart.def(
      "cuda"
      "ProfilerInitialize",
      cudaProfilerInitialize);
#endif
  cudart.def(
      "cuda"
      "MemGetInfo",
      [](int device) -> std::pair<size_t, size_t> {
        c10::cuda::CUDAGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace shared
} // namespace cuda
} // namespace torch
