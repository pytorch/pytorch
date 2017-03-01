// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.
#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/common_cudnn.h"
#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {
namespace python {

REGISTER_CUDA_OPERATOR(Python, GPUFallbackOp<PythonOp>);
REGISTER_CUDA_OPERATOR(PythonGradient, GPUFallbackOp<PythonGradientOp>);

REGISTER_BLOB_FETCHER((TypeMeta::Id<TensorCUDA>()), TensorFetcher<CUDAContext>);
REGISTER_BLOB_FEEDER(CUDA, TensorFeeder<CUDAContext>);

namespace py = pybind11;

void addCUDAGlobalMethods(py::module& m) {
  m.def("num_cuda_devices", &NumCudaDevices);
  m.def("set_default_gpu_id", &SetDefaultGPUID);
  m.def("get_default_gpu_id", &GetDefaultGPUID);
  m.def("get_cudnn_version", &cudnnCompiledVersion);
  m.def("get_cuda_peer_access_pattern", []() {
    std::vector<std::vector<bool>> pattern;
    CAFFE_ENFORCE(caffe2::GetCudaPeerAccessPattern(&pattern));
    return pattern;
  });
};

PYBIND11_PLUGIN(caffe2_pybind11_state_gpu) {
  py::module m(
      "caffe2_pybind11_state_gpu",
      "pybind11 stateful interface to Caffe2 workspaces - GPU edition");

  addGlobalMethods(m);
  addCUDAGlobalMethods(m);
  addObjectMethods(m);
  return m.ptr();
}
} // namespace python
} // namespace caffe2
