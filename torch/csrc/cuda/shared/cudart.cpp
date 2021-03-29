#include <torch/csrc/utils/pybind.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

namespace torch { namespace cuda { namespace shared {

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#ifndef __HIP_PLATFORM_HCC__
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
#ifndef __HIP_PLATFORM_HCC__
  cudart.def("cuda" "ProfilerInitialize", cudaProfilerInitialize);
#endif
}

} // namespace shared
} // namespace cuda
} // namespace torch
