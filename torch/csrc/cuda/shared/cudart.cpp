#include <torch/csrc/utils/pybind.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

namespace torch { namespace cuda { namespace shared {

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  py::enum_<cudaOutputMode_t>(cudart, "cudaOutputMode")
      .value("KeyValuePair", cudaKeyValuePair)
      .value("CSV", cudaCSV);

  py::enum_<cudaError_t>(cudart, "cudaError")
      .value("success", cudaSuccess);

  cudart.def("cudaGetErrorString", cudaGetErrorString);
  cudart.def("cudaProfilerStart", cudaProfilerStart);
  cudart.def("cudaProfilerStop", cudaProfilerStop);
}

} // namespace shared
} // namespace cuda
} // namespace torch
