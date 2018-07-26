// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.

#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"

#ifdef CAFFE2_USE_TRT
#include "caffe2/contrib/tensorrt/tensorrt_tranformer.h"
#endif // CAFFE2_USE_TRT

namespace caffe2 {
namespace python {

REGISTER_CUDA_OPERATOR(Python, GPUFallbackOp<PythonOp<CPUContext, false>>);
REGISTER_CUDA_OPERATOR(
    PythonGradient,
    GPUFallbackOp<PythonGradientOp<CPUContext, false>>);

REGISTER_CUDA_OPERATOR(PythonDLPack, PythonOp<CUDAContext, true>);
REGISTER_CUDA_OPERATOR(
    PythonDLPackGradient,
    PythonGradientOp<CUDAContext, true>);

REGISTER_BLOB_FETCHER((TypeMeta::Id<TensorCUDA>()), TensorFetcher<CUDAContext>);
REGISTER_BLOB_FEEDER(CUDA, TensorFeeder<CUDAContext>);

namespace py = pybind11;

void addCUDAGlobalMethods(py::module& m) {
  m.def("num_cuda_devices", &NumCudaDevices);
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
    obj["totalGlobalMem"] = py::cast(prop.totalGlobalMem);
    return obj;
  });
  m.def(
      "onnx_to_trt_op",
      [](const py::bytes& onnx_model_str,
         const std::unordered_map<std::string, std::vector<int>>&
             output_size_hints,
         int max_batch_size,
         int max_workspace_size,
         int verbosity,
         bool debug_builder) -> py::bytes {
#ifdef CAFFE2_USE_TRT
        TensorRTTransformer t(
            max_batch_size, max_workspace_size, verbosity, debug_builder);
        auto op_def =
            t.BuildTrtOp(onnx_model_str.cast<std::string>(), output_size_hints);
        std::string out;
        op_def.SerializeToString(&out);
        return py::bytes(out);
#else
        CAFFE_THROW("Please build Caffe2 with USE_TENSORRT=1");
#endif // CAFFE2_USE_TRT
      });
  m.def(
      "transform_trt",
      [](const py::bytes& pred_net_str,
         const std::unordered_map<std::string, std::vector<int>>& shapes,
         int max_batch_size,
         int max_workspace_size,
         int verbosity,
         bool debug_builder,
         bool build_serializable_op) -> py::bytes {
#ifdef CAFFE2_USE_TRT
        caffe2::NetDef pred_net;
        if (!ParseProtoFromLargeString(
                pred_net_str.cast<std::string>(), &pred_net)) {
          LOG(ERROR) << "broken pred_net protobuf";
        }
        std::unordered_map<std::string, TensorShape> tensor_shapes;
        for (const auto& it : shapes) {
          tensor_shapes.emplace(
              it.first, CreateTensorShape(it.second, TensorProto::FLOAT));
        }
        TensorRTTransformer ts(
            max_batch_size,
            max_workspace_size,
            verbosity,
            debug_builder,
            build_serializable_op);
        ts.Transform(GetCurrentWorkspace(), &pred_net, tensor_shapes);
        std::string pred_net_str2;
        pred_net.SerializeToString(&pred_net_str2);
        return py::bytes(pred_net_str2);
#else
        CAFFE_THROW("Please build Caffe2 with USE_TENSORRT=1");
#endif // CAFFE2_USE_TRT
      });
};

void addCUDAObjectMethods(py::module& m) {
  py::class_<DLPackWrapper<CUDAContext>>(m, "DLPackTensorCUDA")
      .def_property_readonly(
          "data",
          [](DLPackWrapper<CUDAContext>* t) -> py::object {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                CUDA,
                "Expected CUDA device option for CUDA tensor");

            return t->data();
          },
          "Return DLPack tensor with tensor's data.")
      .def(
          "feed",
          [](DLPackWrapper<CUDAContext>* t, py::object obj) {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                CUDA,
                "Expected CUDA device option for CUDA tensor");
            t->feed(obj);
          },
          "Copy data from given DLPack tensor into this tensor.")
      .def_property_readonly(
          "_shape",
          [](const DLPackWrapper<CUDAContext>& t) { return t.tensor->dims(); })
      .def(
          "_reshape",
          [](DLPackWrapper<CUDAContext>* t, std::vector<TIndex> dims) {
            t->tensor->Resize(dims);
          });
}

PYBIND11_MODULE(caffe2_pybind11_state_gpu, m) {
  m.doc() = "pybind11 stateful interface to Caffe2 workspaces - GPU edition";

  addGlobalMethods(m);
  addCUDAGlobalMethods(m);
  addObjectMethods(m);
  addCUDAObjectMethods(m);
}
} // namespace python
} // namespace caffe2
