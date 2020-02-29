#include <torch/csrc/onnx/init.h>
#include <torch/csrc/onnx/onnx.h>
#include <onnx/onnx_pb.h>

namespace torch { namespace onnx {
void initONNXBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto onnx = m.def_submodule("_onnx");
  py::enum_<::ONNX_NAMESPACE::TensorProto_DataType>(onnx, "TensorProtoDataType")
      .value("UNDEFINED", ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED)
      .value("FLOAT", ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      .value("UINT8", ::ONNX_NAMESPACE::TensorProto_DataType_UINT8)
      .value("INT8", ::ONNX_NAMESPACE::TensorProto_DataType_INT8)
      .value("UINT16", ::ONNX_NAMESPACE::TensorProto_DataType_UINT16)
      .value("INT16", ::ONNX_NAMESPACE::TensorProto_DataType_INT16)
      .value("INT32", ::ONNX_NAMESPACE::TensorProto_DataType_INT32)
      .value("INT64", ::ONNX_NAMESPACE::TensorProto_DataType_INT64)
      .value("STRING", ::ONNX_NAMESPACE::TensorProto_DataType_STRING)
      .value("BOOL", ::ONNX_NAMESPACE::TensorProto_DataType_BOOL)
      .value("FLOAT16", ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)
      .value("DOUBLE", ::ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)
      .value("UINT32", ::ONNX_NAMESPACE::TensorProto_DataType_UINT32)
      .value("UINT64", ::ONNX_NAMESPACE::TensorProto_DataType_UINT64)
      .value("COMPLEX64", ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64)
      .value("COMPLEX128", ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128);

  py::enum_<OperatorExportTypes>(onnx, "OperatorExportTypes")
    .value("ONNX", OperatorExportTypes::ONNX)
    .value("ONNX_ATEN", OperatorExportTypes::ONNX_ATEN)
    .value("ONNX_ATEN_FALLBACK", OperatorExportTypes::ONNX_ATEN_FALLBACK)
    .value("RAW", OperatorExportTypes::RAW);

  onnx.attr("IR_VERSION") = IR_VERSION;
  onnx.attr("PRODUCER_VERSION") = py::str(PRODUCER_VERSION);

#ifdef PYTORCH_ONNX_CAFFE2_BUNDLE
  onnx.attr("PYTORCH_ONNX_CAFFE2_BUNDLE") = true;
#else
  onnx.attr("PYTORCH_ONNX_CAFFE2_BUNDLE") = false;
#endif
}
}} // namespace torch::onnx
