/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  m.def("get_cuda_version", &CudaVersion);
  m.def("get_cudnn_version", &cudnnCompiledVersion);
  m.def("get_cuda_peer_access_pattern", []() {
    std::vector<std::vector<bool>> pattern;
    CAFFE_ENFORCE(caffe2::GetCudaPeerAccessPattern(&pattern));
    return pattern;
  });
  m.def("get_device_properties", [](int deviceid) {
      auto& prop = GetDeviceProperty(deviceid);
      std::map<std::string, py::object> obj;
      obj["name"] = py::cast(prop.name);
      obj["major"] = py::cast(prop.major);
      obj["minor"] = py::cast(prop.minor);
      return obj;
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
