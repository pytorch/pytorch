#include "caffe2/operators/onnxifi_op.h"

namespace caffe2 {

namespace {

void CPUTensorToTensorProto(
    const TensorCPU& cpu_tensor,
    ::ONNX_NAMESPACE::TensorProto* t) {
  const auto len = cpu_tensor.size();
  if (cpu_tensor.template IsType<float>()) {
    t->set_data_type(::ONNX_NAMESPACE::TensorProto::FLOAT);
    const float* data = cpu_tensor.template data<float>();
    for (auto i = 0; i < len; ++i) {
      t->add_float_data(*data++);
    }
  } else if (cpu_tensor.template IsType<int64_t>()) {
    t->set_data_type(::ONNX_NAMESPACE::TensorProto::INT64);
    const int64_t* data = cpu_tensor.template data<int64_t>();
    for (auto i = 0; i < len; ++i) {
      t->add_int64_data(*data++);
    }
  } else if (cpu_tensor.template IsType<int32_t>()) {
    t->set_data_type(::ONNX_NAMESPACE::TensorProto::INT32);
    const int32_t* data = cpu_tensor.template data<int32_t>();
    for (auto i = 0; i < len; ++i) {
      t->add_int32_data(*data++);
    }
  } else {
    CAFFE_THROW(
        "Don't know how to convert workspace tensor type ",
        cpu_tensor.meta().name(),
        " to ONNX TensorProto");
  }
}

void BlobToTensorProto(
    const std::string& name,
    Workspace* ws,
    ::ONNX_NAMESPACE::TensorProto* t) {
  // Set name
  t->set_name(name);
  const Blob* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob ", name, " doesn't exist");

  // Set dims
  const auto shape = GetTensorShapeOfBlob(blob);
  for (const auto i : shape.dims()) {
    t->add_dims(i);
  }

  // Set values
  CAFFE_ENFORCE(
      blob->template IsType<TensorCPU>(),
      "Initialization blob ",
      name,
      " needs to be either TensorCPU or TensorCUDA");
  CPUTensorToTensorProto(blob->template Get<TensorCPU>(), t);
}
}

template <>
void OnnxifiOp<float, CPUContext>::BuildInitializationList(
    Workspace* ws,
    ::ONNX_NAMESPACE::GraphProto* g,
    std::unordered_set<std::string>* initialization_list) {
  const std::vector<string>& ws_blobs = ws->Blobs();

  for (const auto& s : ws_blobs) {
    auto it = initialization_list->find(s);
    if (it != initialization_list->end()) {
      auto* init_tensor = g->add_initializer();
      BlobToTensorProto(s, ws, init_tensor);
      initialization_list->erase(it);
    }
  }
  CAFFE_ENFORCE(
      initialization_list->empty(), "Unfulfilled initialization list");
  for (const auto& t : g->initializer()) {
    VLOG(2) << "Initializer: " << t.name();
  }
}

template <>
bool OnnxifiOp<float, CPUContext>::RunOnDevice() {
  for (unsigned i = 0U; i < InputSize(); ++i) {
    const auto& input_tensor = Input(i);
    const auto& tensor_dims = input_tensor.dims();
    auto& tensor_descriptor = input_desc_.at(i);
    tensor_descriptor.dataType = ONNXIFI_DATATYPE_FLOAT32;
    tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    tensor_descriptor.dimensions = tensor_dims.size();
    input_shapes_.emplace_back();
    auto& input_shape = input_shapes_.back();
    for (unsigned j = 0U; j < tensor_descriptor.dimensions; ++j) {
      input_shape.push_back(tensor_dims[j]);
    }
    tensor_descriptor.shape = input_shape.data();
    tensor_descriptor.buffer = (onnxPointer)(input_tensor.data<float>());
  }

  for (unsigned i = 0U; i < OutputSize(); ++i) {
    auto* output_tensor = Output(i);
    std::vector<TIndex> tensor_dims;
    SetOutputShape(i, &tensor_dims);
    output_tensor->Resize(tensor_dims);
    auto& tensor_descriptor = output_desc_.at(i);
    tensor_descriptor.dataType = ONNXIFI_DATATYPE_FLOAT32;
    tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    tensor_descriptor.dimensions = tensor_dims.size();
    tensor_descriptor.shape = new uint64_t[tensor_descriptor.dimensions];
    output_shapes_.emplace_back();
    auto& output_shape = output_shapes_.back();
    for (unsigned j = 0U; j < tensor_descriptor.dimensions; ++j) {
      output_shape.push_back(tensor_dims[j]);
    }
    tensor_descriptor.shape = output_shape.data();
    tensor_descriptor.buffer =
        (onnxPointer)(output_tensor->mutable_data<float>());
  }

  CAFFE_ENFORCE_EQ(
      lib_->onnxSetGraphIO(
          graph_,
          input_desc_.size(),
          input_desc_.data(),
          output_desc_.size(),
          output_desc_.data()),
      ONNXIFI_STATUS_SUCCESS);

  // TODO (support async)
  CAFFE_ENFORCE_EQ(
      lib_->onnxRunGraph(graph_, nullptr, nullptr), ONNXIFI_STATUS_SUCCESS);

  return true;
}

REGISTER_CPU_OPERATOR(Onnxifi, OnnxifiOp<float, CPUContext>);
OPERATOR_SCHEMA(Onnxifi)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
    The Onnixifi operator is a black-box operator to lower the computation to Onnxifi backend
    )DOC")
    .Arg("onnxifi_backend", "(string default=\"\") Name of the backend")
    .Arg("onnxifi_backend_suffix", "(string default=\"\") Function suffix of the backend")
    .Arg(
        "onnxifi_backend_path",
        "(string default=\"\") Path to the onnxifi bakcend dynamic library")
    .Arg("onnxifi_backend_idx", "(int default 0) Backend index to be used")
    .Arg(
        "onnx_model",
        "(string default=\"\") Serialized ONNX model to be converted to backend representation")
    .Arg(
        "initializers",
        "Initialization pair indicating the mapping of the name between NetDef and ONNX model");
} // namespace caffe2
