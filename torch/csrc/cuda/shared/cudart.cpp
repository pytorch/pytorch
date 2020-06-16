#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
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
  py::enum_<cudaOutputMode_t>(cudart, "hipOutputMode")
      .value("KeyValuePair", hipKeyValuePair)
      .value("CSV", hipCSV);
#endif

  py::enum_<hipError_t>(cudart, "cudaError")
      .value("success", hipSuccess);

  cudart.def("hipGetErrorString", hipGetErrorString);
  cudart.def("hipProfilerStart", hipProfilerStart);
  cudart.def("hipProfilerStop", hipProfilerStop);
  cudart.def("hipHostRegister", hipHostRegister);
#ifndef __HIP_PLATFORM_HCC__
  cudart.def("hipProfilerInitialize", hipProfilerInitialize);
#endif
}

} // namespace shared
} // namespace cuda
} // namespace torch
