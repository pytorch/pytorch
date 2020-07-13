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

#ifndef __HIP_PLATFORM_HCC__
  py::enum_<cudaOutputMode_t>(cudart, "cudaOutputMode")
      .value("KeyValuePair", cudaKeyValuePair)
      .value("CSV", cudaCSV);
#endif

  py::enum_<cudaError_t>(cudart, "cudaError")
      .value("success", cudaSuccess);

  cudart.def("cudaGetErrorString", cudaGetErrorString);
  cudart.def("cudaProfilerStart", cudaProfilerStart);
  cudart.def("cudaProfilerStop", cudaProfilerStop);
  cudart.def("cudaHostRegister", cudaHostRegister);
#ifndef __HIP_PLATFORM_HCC__
  cudart.def("cudaProfilerInitialize", cudaProfilerInitialize);
#endif

  // The HIP rewrite rules will change the names of the functions defined above
  // when building with HIP. To provide a consistent interface, we also define
  // versions of those functions that won't be touched by the rewrite (but will
  // still correctly resolve to CUDA and HIP). We leave the ones above for
  // backwards compatibility.
  cudart.def("getErrorString", cudaGetErrorString);
  cudart.def("profilerStart", cudaProfilerStart);
  cudart.def("profilerStop", cudaProfilerStop);
  cudart.def("hostRegister", cudaHostRegister);
}

} // namespace shared
} // namespace cuda
} // namespace torch
