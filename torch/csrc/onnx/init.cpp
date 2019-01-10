#include "torch/csrc/onnx/init.h"
#include "torch/csrc/onnx/onnx.pb.h"
#include "torch/csrc/onnx/onnx.h"

namespace torch { namespace onnx {
void initONNXBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto onnx = m.def_submodule("_onnx");
  py::enum_<onnx_TensorProto_DataType>(onnx, "TensorProtoDataType")
      .value("UNDEFINED", onnx_TensorProto_DataType_UNDEFINED)
      .value("FLOAT", onnx_TensorProto_DataType_FLOAT)
      .value("UINT8", onnx_TensorProto_DataType_UINT8)
      .value("INT8", onnx_TensorProto_DataType_INT8)
      .value("UINT16", onnx_TensorProto_DataType_UINT16)
      .value("INT16", onnx_TensorProto_DataType_INT16)
      .value("INT32", onnx_TensorProto_DataType_INT32)
      .value("INT64", onnx_TensorProto_DataType_INT64)
      .value("STRING", onnx_TensorProto_DataType_STRING)
      .value("BOOL", onnx_TensorProto_DataType_BOOL)
      .value("FLOAT16", onnx_TensorProto_DataType_FLOAT16)
      .value("DOUBLE", onnx_TensorProto_DataType_DOUBLE)
      .value("UINT32", onnx_TensorProto_DataType_UINT32)
      .value("UINT64", onnx_TensorProto_DataType_UINT64)
      .value("COMPLEX64", onnx_TensorProto_DataType_COMPLEX64)
      .value("COMPLEX128", onnx_TensorProto_DataType_COMPLEX128);

  py::class_<ModelProto>(onnx, "ModelProto")
      .def("prettyPrint", &ModelProto::prettyPrint);
}
}} // namespace torch::onnx
