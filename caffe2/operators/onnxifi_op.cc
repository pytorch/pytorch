#include "caffe2/operators/onnxifi_op.h"

namespace caffe2 {

namespace {

void SetInputTensorDescriptorTypeAndBuffer(
    const Tensor& cpu_tensor,
    onnxTensorDescriptorV1* desc) {
  if (cpu_tensor.template IsType<float>()) {
    desc->dataType = ONNXIFI_DATATYPE_FLOAT32;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<float>());
  } else if (cpu_tensor.template IsType<int32_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT32;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int32_t>());
  } else if (cpu_tensor.template IsType<int8_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT8;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int8_t>());
  } else if (cpu_tensor.template IsType<uint8_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_UINT8;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<uint8_t>());
  } else if (cpu_tensor.template IsType<int64_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT64;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int64_t>());
  } else if (cpu_tensor.template IsType<int16_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT16;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int16_t>());
  } else if (cpu_tensor.template IsType<uint16_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_UINT16;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<uint16_t>());
  } else {
    CAFFE_THROW(
        "Unsupported tensor type in ONNXIFI: ", cpu_tensor.dtype().name());
  }
}

void SetOutputTensorDescriptorTypeAndBuffer(
    uint64_t onnxifi_type,
    Tensor* cpu_tensor,
    onnxTensorDescriptorV1* desc) {
  desc->dataType = onnxifi_type;
  switch (onnxifi_type) {
    case (ONNXIFI_DATATYPE_FLOAT32):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<float>());
      break;
    case (ONNXIFI_DATATYPE_INT32):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<int32_t>());
      break;
    case (ONNXIFI_DATATYPE_INT8):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<int8_t>());
      break;
    case (ONNXIFI_DATATYPE_UINT8):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<uint8_t>());
      break;
    case (ONNXIFI_DATATYPE_INT64):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<int64_t>());
      break;
    case (ONNXIFI_DATATYPE_INT16):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<int16_t>());
      break;
    case (ONNXIFI_DATATYPE_UINT16):
      desc->buffer =
          reinterpret_cast<onnxPointer>(cpu_tensor->mutable_data<uint16_t>());
      break;
    default:
      CAFFE_THROW("Unsupported ONXNIFI data type: ", onnxifi_type);
  }
}

void BlobToTensorDescriptor(
    const std::string& name,
    Workspace* ws,
    onnxTensorDescriptorV1* desc,
    std::vector<std::vector<uint64_t>>* shapes) {
  const Blob* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob ", name, " doesn't exist");

  // Memory type
  // We only allow weights to be CPU tensor for now
  CAFFE_ENFORCE(
      BlobIsTensorType(*blob, CPU),
      "Initialization blob ",
      name,
      " needs to be TensorCPU");
  desc->tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  desc->memoryType = ONNXIFI_MEMORY_TYPE_CPU;

  // Data type
  const auto& cpu_tensor = blob->template Get<TensorCPU>();
  SetInputTensorDescriptorTypeAndBuffer(cpu_tensor, desc);

  // Set dims
  const auto shape = cpu_tensor.sizes();
  desc->dimensions = shape.size();
  shapes->emplace_back(shape.cbegin(), shape.cend());
  desc->shape = shapes->back().data();
}
} // namespace

template <>
std::vector<onnxTensorDescriptorV1>
OnnxifiOp<float, CPUContext>::BuildInitializationList(
    Workspace* ws,
    std::unordered_set<std::string>* initialization_list,
    std::vector<std::string>* weight_names,
    std::vector<std::vector<uint64_t>>* weight_shapes) {
  const std::vector<string>& ws_blobs = ws->Blobs();
  // Since onnxTensorDescriptorV1.name will point into the memory in
  // weight_names, we need to prevent weight_names from reallocating by
  // reserving enough memory ahead of time
  weight_names->reserve(ws_blobs.size());
  std::vector<onnxTensorDescriptorV1> descs;
  for (const auto& s : ws_blobs) {
    auto it = initialization_list->find(s);
    if (it != initialization_list->end()) {
      weight_names->emplace_back(s);
      onnxTensorDescriptorV1 tensor_desc;
      tensor_desc.name = weight_names->back().c_str();
      BlobToTensorDescriptor(s, ws, &tensor_desc, weight_shapes);
      descs.push_back(tensor_desc);
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
    const auto tensor_dims = input_tensor.sizes();
    auto& tensor_descriptor = input_desc_.at(i);
    tensor_descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
    tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    tensor_descriptor.dimensions = tensor_dims.size();
    input_shapes_.emplace_back(tensor_dims.cbegin(), tensor_dims.cend());
    tensor_descriptor.shape = input_shapes_.back().data();
    SetInputTensorDescriptorTypeAndBuffer(input_tensor, &tensor_descriptor);
  }

  for (unsigned i = 0U; i < OutputSize(); ++i) {
    auto* output_tensor = Output(i);
    std::vector<size_t> tensor_dims;
    uint64_t type = SetOutputShapeAndType(i, &tensor_dims);
    output_tensor->Resize(tensor_dims);
    auto& tensor_descriptor = output_desc_.at(i);
    tensor_descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
    tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    tensor_descriptor.dimensions = tensor_dims.size();
    CAFFE_ENFORCE(
        tensor_descriptor.dimensions != 0,
        tensor_descriptor.name,
        " has 0 dim");
    output_shapes_.emplace_back(tensor_dims.cbegin(), tensor_dims.cend());
    tensor_descriptor.shape = output_shapes_.back().data();
    SetOutputTensorDescriptorTypeAndBuffer(
        type, output_tensor, &tensor_descriptor);
  }

  CAFFE_ENFORCE_EQ(
      lib_->onnxSetGraphIO(
          graph_,
          input_desc_.size(),
          input_desc_.data(),
          output_desc_.size(),
          output_desc_.data()),
      ONNXIFI_STATUS_SUCCESS);

  onnxMemoryFenceV1 input_fence;
  input_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  input_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
  CAFFE_ENFORCE_EQ(
      lib_->onnxInitEvent(backend_, &input_fence.event),
      ONNXIFI_STATUS_SUCCESS);
  onnxMemoryFenceV1 output_fence;
  output_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  output_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;

  // Call the asycn run on backend, singal event on input fence and wait for the
  // event on output fence
  CAFFE_ENFORCE_EQ(
      lib_->onnxSignalEvent(input_fence.event), ONNXIFI_STATUS_SUCCESS);
  CAFFE_ENFORCE_EQ(
      lib_->onnxRunGraph(graph_, &input_fence, &output_fence),
      ONNXIFI_STATUS_SUCCESS);
  CAFFE_ENFORCE_EQ(
      lib_->onnxWaitEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);

  // Destroy the event objects
  CAFFE_ENFORCE_EQ(
      lib_->onnxReleaseEvent(input_fence.event), ONNXIFI_STATUS_SUCCESS);
  CAFFE_ENFORCE_EQ(
      lib_->onnxReleaseEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);

  return true;
}

REGISTER_CPU_OPERATOR(Onnxifi, OnnxifiOp<float, CPUContext>);
OPERATOR_SCHEMA(Onnxifi)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
    The Onnxifi operator is a black-box operator to lower the computation to Onnxifi backend
    )DOC")
    .Arg(
        "onnx_model",
        "(string default=\"\") Serialized ONNX model to be converted to backend representation")
    .Arg(
        "initializers",
        "Initialization pair indicating the mapping of the name between NetDef and ONNX model");
} // namespace caffe2
