#include <torch/csrc/utils/pybind.h>
#include <cuda.h>
#include <cuda_runtime.h>
#if !defined(USE_ROCM)
#include <cuda_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace torch { namespace cuda { namespace shared {

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#if !defined(USE_ROCM)
  py::enum_<cudaOutputMode_t>(cudart, "cuda" "OutputMode")
      .value("KeyValuePair", cudaKeyValuePair)
      .value("CSV", cudaCSV);
#endif

  py::enum_<cudaError_t>(cudart, "cuda" "Error")
      .value("success", cudaSuccess);

  cudart.def("cuda" "GetErrorString", cudaGetErrorString);
  cudart.def("cuda" "ProfilerStart", cudaProfilerStart);
  cudart.def("cuda" "ProfilerStop", cudaProfilerStop);
  cudart.def("cuda" "HostRegister", [](uintptr_t ptr, size_t size, unsigned int flags) -> cudaError_t {
    return cudaHostRegister((void*)ptr, size, flags);
  });
  cudart.def("cuda" "HostUnregister", [](uintptr_t ptr) -> cudaError_t {
    return cudaHostUnregister((void*)ptr);
  });
  cudart.def("cuda" "StreamCreate", [](uintptr_t ptr) -> cudaError_t {
    return cudaStreamCreate((cudaStream_t*)ptr);
  });
  cudart.def("cuda" "StreamDestroy", [](uintptr_t ptr) -> cudaError_t {
    return cudaStreamDestroy((cudaStream_t)ptr);
  });
#if !defined(USE_ROCM)
  cudart.def("cuda" "ProfilerInitialize", cudaProfilerInitialize);
#endif
  cudart.def("cuda" "MemGetInfo", [](int device) -> std::pair<size_t, size_t> {
    c10::cuda::CUDAGuard guard(device);
    size_t device_free = 0;
    size_t device_total = 0;
    cudaMemGetInfo(&device_free, &device_total);
    return {device_free, device_total};
  });
}

} // namespace shared
} // namespace cuda
} // namespace torch
