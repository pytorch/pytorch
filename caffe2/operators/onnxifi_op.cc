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

void BlobToTensorDescriptor(
    const std::string& name,
    Workspace* ws,
    onnxTensorDescriptor* desc,
    std::vector<std::vector<uint64_t>>* shapes) {
  const Blob* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob ", name, " doesn't exist");

  // Memory type
  // We only allow weights to be CPU tensor for now
  CAFFE_ENFORCE(
      blob->template IsType<TensorCPU>(),
      "Initialization blob ",
      name,
      " needs to be TensorCPU");
  desc->memoryType = ONNXIFI_MEMORY_TYPE_CPU;

  // Data type
  const auto& cpu_tensor = blob->template Get<TensorCPU>();
  if (cpu_tensor.template IsType<float>()) {
    desc->dataType = ONNXIFI_DATATYPE_FLOAT32;
    desc->buffer = (onnxPointer)(cpu_tensor.data<float>());
  } else if (cpu_tensor.template IsType<int64_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT64;
    desc->buffer = (onnxPointer)(cpu_tensor.data<int64_t>());
  } else if (cpu_tensor.template IsType<int32_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT32;
    desc->buffer = (onnxPointer)(cpu_tensor.data<int32_t>());
  }

  // Set dims
  const auto shape = GetTensorShapeOfBlob(blob);
  desc->dimensions = shape.dims_size();
  shapes->emplace_back();
  auto& shape_tmp = shapes->back();
  for (const auto d : shape.dims()) {
    shape_tmp.push_back(d);
  }
  desc->shape = shape_tmp.data();
}
} // namespace

template <>
std::vector<onnxTensorDescriptor>
OnnxifiOp<float, CPUContext>::BuildInitializationList(
    Workspace* ws,
    std::unordered_set<std::string>* initialization_list,
    std::vector<std::string>* weight_names,
    std::vector<std::vector<uint64_t>>* weight_shapes) {
  const std::vector<string>& ws_blobs = ws->Blobs();
  std::vector<onnxTensorDescriptor> descs;
  for (const auto& s : ws_blobs) {
    auto it = initialization_list->find(s);
    if (it != initialization_list->end()) {
      weight_names->emplace_back(s);
      descs.emplace_back();
      auto& tensor_desc = descs.back();
      tensor_desc.name = weight_names->back().c_str();
      BlobToTensorDescriptor(s, ws, &tensor_desc, weight_shapes);
      initialization_list->erase(it);
    }
  }
  CAFFE_ENFORCE(
      initialization_list->empty(), "Unfulfilled initialization list");
  return descs;
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
    The Onnxifi operator is a black-box operator to lower the computation to Onnxifi backend
    )DOC")
    .Arg("onnxifi_backend", "(string default=\"\") Name of the backend")
    .Arg(
        "onnx_model",
        "(string default=\"\") Serialized ONNX model to be converted to backend representation")
    .Arg(
        "initializers",
        "Initialization pair indicating the mapping of the name between NetDef and ONNX model");
} // namespace caffe2
