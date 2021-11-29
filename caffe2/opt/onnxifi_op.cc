#include "caffe2/opt/onnxifi_op.h"
#include "caffe2/operators/slice_op.h"
#include "caffe2/opt/bound_shape_inferencer.h"

#include <c10/util/irange.h>

namespace caffe2 {

namespace {

void setInputTensorDescriptorTypeAndBuffer(
    const Tensor& cpu_tensor,
    onnxTensorDescriptorV1* desc) {
  if (cpu_tensor.template IsType<int32_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT32;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int32_t>());
  } else if (cpu_tensor.template IsType<c10::Half>()) {
    desc->dataType = ONNXIFI_DATATYPE_FLOAT16;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<c10::Half>());
  } else if (cpu_tensor.template IsType<float>()) {
    desc->dataType = ONNXIFI_DATATYPE_FLOAT32;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<float>());
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

void setInputTensorDescriptorTypeAndBuffer(
    const int8::Int8TensorCPU& cpu_int8tensor,
    onnxTensorDescriptorV1* desc) {
  const Tensor& cpu_tensor = cpu_int8tensor.t;
  if (cpu_tensor.template IsType<uint8_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_UINT8;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<uint8_t>());
  } else if (cpu_tensor.template IsType<int8_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT8;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int8_t>());
  } else if (cpu_tensor.template IsType<int32_t>()) {
    desc->dataType = ONNXIFI_DATATYPE_INT32;
    desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int32_t>());
  } else {
    CAFFE_THROW(
        "Unsupported Int8Tensor type in ONNXIFI: ", cpu_tensor.dtype().name());
  }
  desc->quantizationParams = 1;
  desc->quantizationAxis = 1;
  desc->scales = &cpu_int8tensor.scale;
  desc->biases = &cpu_int8tensor.zero_point;
}

template <typename T>
void adjustQuantizedOffsetImpl(Tensor* t, uint8_t offset) {
  auto* data = t->mutable_data<T>();
  for (auto i: c10::irange(t->numel())) {
    data[i] -= offset;
  }
}

void adjustQuantizedOffset(Tensor* t, uint8_t offset) {
  if (t->template IsType<uint8_t>()) {
    adjustQuantizedOffsetImpl<uint8_t>(t, offset);
  }
}

TypeMeta OnnxifiTypeToDataType(uint64_t onnxifi_type) {
  static std::map<uint64_t, TypeMeta> data_type_map{
      {ONNXIFI_DATATYPE_FLOAT32, TypeMeta::Make<float>()},
      {ONNXIFI_DATATYPE_FLOAT16, TypeMeta::Make<c10::Half>()},
      {ONNXIFI_DATATYPE_INT32, TypeMeta::Make<int>()},
      {ONNXIFI_DATATYPE_INT8, TypeMeta::Make<int8_t>()},
      {ONNXIFI_DATATYPE_UINT8, TypeMeta::Make<uint8_t>()},
      {ONNXIFI_DATATYPE_INT64, TypeMeta::Make<int64_t>()},
      {ONNXIFI_DATATYPE_INT16, TypeMeta::Make<int16_t>()},
      {ONNXIFI_DATATYPE_UINT16, TypeMeta::Make<uint16_t>()},
  };
  const auto it = data_type_map.find(onnxifi_type);
  CAFFE_ENFORCE(
      it != data_type_map.end(),
      "Unsupported ONNXIFI data type: ",
      onnxifi_type);
  return it->second;
}

void setOutputTensorDescriptorTypeAndBuffer(
    uint64_t onnxifi_type,
    Tensor* cpu_tensor,
    onnxTensorDescriptorV1* desc) {
  desc->dataType = onnxifi_type;
  desc->buffer = reinterpret_cast<onnxPointer>(
      cpu_tensor->raw_mutable_data(OnnxifiTypeToDataType(onnxifi_type)));
}

#ifndef C10_MOBILE
void copyDescriptor(
    const ExternalTensorDescriptor* from,
    onnxTensorDescriptorV1* to) {
  to->dataType = from->dataType;
  to->buffer = from->buffer;
  to->isOffline = from->isOffline;
  to->quantizationParams = from->quantizationParams;
  to->quantizationAxis = from->quantizationAxis;
  to->scales = from->scales;
  to->biases = from->biases;
  to->dimensions = from->dimensions;
  to->shape = from->shape;
}
#endif

void BlobToTensorDescriptor(
    const std::string& name,
    Workspace* ws,
    onnxTensorDescriptorV1* desc,
    std::vector<std::vector<uint64_t>>* shapes,
    std::vector<std::vector<float>>* all_scales,
    std::vector<std::vector<int32_t>>* all_offsets) {
  const Blob* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob ", name, " doesn't exist");
  const bool is_int8tensor =
      blob->meta().id() == TypeMeta::Id<int8::Int8TensorCPU>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool is_external_tensor;
#ifndef C10_MOBILE
  auto function_ptr =
      ExternalTensorFunctionsBaseRegistry()->Create(blob->meta().id());
  is_external_tensor = function_ptr != nullptr;
#else
  is_external_tensor = false;
#endif
  // Memory type
  // We only allow weights to be CPU tensor or int8tensor for now
  CAFFE_ENFORCE(
      (BlobIsTensorType(*blob, CPU) || BlobIsInt8TensorCPUType(*blob) ||
       is_external_tensor),
      "Initialization blob ",
      name,
      " needs to be TensorCPU or Int8TensorCPU or Int8FCDNNLowPPackedWeightBlob Based class: ",
      blob->TypeName());
  desc->tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  desc->memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  desc->isOffline = false;

  if (is_int8tensor) {
    // Data type
    const auto& cpu_int8tensor = blob->template Get<int8::Int8TensorCPU>();
    const auto& cpu_tensor = cpu_int8tensor.t;
    setInputTensorDescriptorTypeAndBuffer(cpu_int8tensor, desc);
    // Set dims
    const auto shape = cpu_tensor.sizes();
    desc->dimensions = shape.size();
    shapes->emplace_back(shape.cbegin(), shape.cend());
    desc->shape = shapes->back().data();
  } else if (is_external_tensor) {
#ifndef C10_MOBILE
    ExternalTensorDescriptor ext_desc;
    function_ptr->SetupExternalTensorDescriptor(
        blob, shapes, all_scales, all_offsets, &ext_desc);
    copyDescriptor(&ext_desc, desc);
#endif
  } else {
    // Data type
    const auto& cpu_tensor = blob->template Get<TensorCPU>();
    setInputTensorDescriptorTypeAndBuffer(cpu_tensor, desc);
    // Set dims
    const auto shape = cpu_tensor.sizes();
    desc->dimensions = shape.size();
    shapes->emplace_back(shape.cbegin(), shape.cend());
    desc->shape = shapes->back().data();
    desc->quantizationParams = 0;
  }
}

uint64_t getOnnxifiDataType(caffe2::TensorProto::DataType t) {
#define CAFFE2_TO_ONNXIFI_TYPE(x) \
  case (caffe2::TensorProto::x):  \
    return ONNXIFI_DATATYPE_##x
  switch (t) {
    CAFFE2_TO_ONNXIFI_TYPE(INT8);
    CAFFE2_TO_ONNXIFI_TYPE(UINT8);
    CAFFE2_TO_ONNXIFI_TYPE(UINT16);
    CAFFE2_TO_ONNXIFI_TYPE(INT16);
    CAFFE2_TO_ONNXIFI_TYPE(INT32);
    CAFFE2_TO_ONNXIFI_TYPE(INT64);
    CAFFE2_TO_ONNXIFI_TYPE(FLOAT16);
    case (caffe2::TensorProto::FLOAT):
      return ONNXIFI_DATATYPE_FLOAT32;
    default:
      LOG(WARNING) << "Unsupported Caffe2 tensor type: " << t;
      return ONNXIFI_DATATYPE_UNDEFINED;
  }
#undef CAFFE2_TO_ONNXIFI_TYPE
}

} // namespace

namespace details {
TensorInfo::TensorInfo(const TensorProto& t)
    : onnxifi_type(getOnnxifiDataType(t.data_type())),
      quantized(false),
      quantizationAxis(0),
      quantizationParams(0) {
  for (const auto d : t.dims()) {
    dims.push_back(d);
  }
}

TensorInfo::TensorInfo(const QTensorProto& t)
    : onnxifi_type(getOnnxifiDataType(t.data_type())),
      quantized(true),
      quantizationAxis(t.has_axis() ? t.axis() : 0),
      quantizationParams(t.scales_size() ? t.scales_size() : 1) {
  for (const auto d : t.dims()) {
    dims.push_back(d);
  }
  if (t.scales_size()) {
    for (const auto d : t.scales()) {
      scales.push_back(static_cast<float>(d));
    }
    for (const auto d : t.biases()) {
      biases.push_back(static_cast<int32_t>(d));
    }
  } else {
    scales.push_back(static_cast<float>(t.scale()));
    biases.push_back(static_cast<int32_t>(t.bias()));
  }
}
} // namespace details

template <>
std::vector<onnxTensorDescriptorV1>
OnnxifiOp<CPUContext>::buildInitializationList(
    Workspace* ws,
    const std::vector<std::string>& initializers,
    std::vector<std::string>* weight_names,
    std::vector<std::vector<uint64_t>>* weight_shapes,
    std::vector<std::vector<float>>* all_scales,
    std::vector<std::vector<int32_t>>* all_offsets) const {
  std::unordered_set<std::string> initialization_list(
      initializers.begin(), initializers.end());
  const std::vector<string>& ws_blobs = ws->Blobs();
  // Since onnxTensorDescriptorV1.name will point into the memory in
  // weight_names, we need to prevent weight_names from reallocating by
  // reserving enough memory ahead of time
  weight_names->reserve(ws_blobs.size());
  std::vector<onnxTensorDescriptorV1> descs;
  for (const auto& s : ws_blobs) {
    auto it = initialization_list.find(s);
    if (it != initialization_list.end()) {
      weight_names->emplace_back(s);
      onnxTensorDescriptorV1 tensor_desc;
      tensor_desc.name = weight_names->back().c_str();
      BlobToTensorDescriptor(
          s, ws, &tensor_desc, weight_shapes, all_scales, all_offsets);
      descs.push_back(tensor_desc);
      initialization_list.erase(it);
    }
  }
  CAFFE_ENFORCE(initialization_list.empty(), "Unfulfilled initialization list");
  return descs;
}

template <>
details::OutputReshapeInfo OnnxifiOp<CPUContext>::initOutputReshapeInfo()
    const {
  details::OutputReshapeInfo output_reshape_info;
  output_reshape_info.begins.reserve(output_names_.size());
  output_reshape_info.ends.reserve(output_names_.size());
  output_reshape_info.fast_path.reserve(output_names_.size());
  for (auto i: c10::irange(output_names_.size())) {
    const auto it = output_shape_hints_.find(i);
    CAFFE_ENFORCE(
        it != output_shape_hints_.end(),
        "Cannot find output shape hints for ",
        output_names_[i]);
    int64_t num_dims = it->second.dims.size();
    // Initialize the tensors used to slice the output
    output_reshape_info.begins.emplace_back();
    ReinitializeTensor(
        &output_reshape_info.begins.back(),
        {num_dims},
        at::dtype<int32_t>().device(CPU));
    output_reshape_info.ends.emplace_back();
    ReinitializeTensor(
        &output_reshape_info.ends.back(),
        {num_dims},
        at::dtype<int32_t>().device(CPU));
  }
  return output_reshape_info;
}

template <>
template <typename DimContainer>
void OnnxifiOp<CPUContext>::fillOutputReshapeInfo(
    const DimContainer& real_shape,
    c10::ArrayRef<uint64_t> max_shape,
    details::OutputReshapeInfo& output_reshape_info,
    int currentIndex) {
  CAFFE_ENFORCE_EQ(real_shape.size(), max_shape.size());
  const auto dim_size = real_shape.size();
  auto& begin = output_reshape_info.begins[currentIndex];
  begin.Resize(dim_size);
  int32_t* begin_ptr = begin.template mutable_data<int32_t>();
  auto& end = output_reshape_info.ends[currentIndex];
  end.Resize(dim_size);
  int32_t* end_ptr = end.template mutable_data<int32_t>();
  int32_t mismatch = 0;
  for (auto j: c10::irange(dim_size)) {
    CAFFE_ENFORCE_GE(
        max_shape[j],
        real_shape[j],
        "It is weird that max shape of ",
        output_names_[currentIndex],
        " is smaller than real shape at dim ",
        j,
        " (",
        max_shape[j],
        " vs ",
        real_shape[j],
        ")");
    begin_ptr[j] = 0;
    if (max_shape[j] > static_cast<uint64_t>(real_shape[j])) {
      end_ptr[j] = real_shape[j];
      mismatch += j;
    } else {
      end_ptr[j] = max_shape[j];
    }
  }

  if (dim_size > 0) {
    output_reshape_info.fast_path[currentIndex] = !mismatch;
  } else {
    output_reshape_info.fast_path[currentIndex] = false;
  }
}

template <>
void OnnxifiOp<CPUContext>::extractOutputBatchSizes(int current_batch_size) {
  auto& output_reshape_info =
      output_reshape_info_.emplace(current_batch_size, initOutputReshapeInfo())
          .first->second;

  if (use_passed_output_shapes_) {
    const auto shape_info_it = output_shapes_per_bs_.find(current_batch_size);
    CAFFE_ENFORCE(
        shape_info_it != output_shapes_per_bs_.end(),
        "Unable to find outputs shapes for bs=",
        current_batch_size);
    CAFFE_ENFORCE_EQ(shape_info_it->second.size(), OutputSize());

    for (int i = 0; i < OutputSize(); ++i) {
      fillOutputReshapeInfo(
          shape_info_it->second[i],
          output_shapes_max_bs_[i],
          output_reshape_info,
          i);
    }
  } else {
    BoundShapeSpec spec(current_batch_size, max_seq_size_);
    auto bound_shape_inferencer =
        BoundShapeInferencerRegistry()->Create("C10", spec);
    for (int i = 0; i < InputSize(); ++i) {
      at::IntArrayRef dim0;
      bool quantized = false;
      if (this->template InputIsType<int8::Int8TensorCPU>(i)) {
        const auto& input_tensor_int8 =
            this->template Input<int8::Int8TensorCPU>(i);
        const auto& t0 = input_tensor_int8.t;
        dim0 = t0.sizes();
        quantized = true;
      } else {
        const auto& t0 = Input(i);
        dim0 = t0.sizes();
      }
      TensorShape shape;
      for (const auto d : dim0) {
        shape.add_dims(d);
      }
      std::vector<TensorBoundShape::DimType> dim_type(
          shape.dims_size(), TensorBoundShape_DimType_CONSTANT);
      if (dim_type.size()) {
        dim_type[0] = TensorBoundShape_DimType_BATCH;
      }
      input_shape_info_[input_names_[i]] =
          ShapeInfo(dim_type, std::move(shape), quantized);
    }
    bound_shape_inferencer->InferBoundShapeAndType(
        netdef_, input_shape_info_, nullptr, false);
    const auto& shape_info = bound_shape_inferencer->shape_info();
    for (int i = 0; i < OutputSize(); ++i) {
      const auto find_res = shape_info.find(output_names_[i]);
      CAFFE_ENFORCE(find_res != shape_info.end());
      fillOutputReshapeInfo(
          find_res->second.shape.dims(),
          output_shapes_max_bs_[i],
          output_reshape_info,
          i);
    }
  }
}

template <>
int OnnxifiOp<CPUContext>::extractOutputBatchSizes() {
  if (use_onnx_ || !adjust_output_batch_) {
    return max_batch_size_;
  }

  // Get the real batch size from nominal input. If it's equal to
  // max_batch_size, mark that we don't need to adjust batch size and return.
  // Otherwise, do a pass of shape inference to get the real shapes of the
  // outputs.
  const Tensor* t = nullptr;
  if (this->template InputIsType<int8::Int8TensorCPU>(nominal_batch_idx_)) {
    const auto& input_tensor_int8 =
        this->template Input<int8::Int8TensorCPU>(nominal_batch_idx_);
    t = &input_tensor_int8.t;
  } else {
    t = &Input(nominal_batch_idx_);
  }

  CAFFE_ENFORCE(
      t, "Null input shape tensor ptr. Possibly unsupported tensor type");
  CAFFE_ENFORCE(
      !t->sizes().empty(),
      input_names_[nominal_batch_idx_],
      " cannot be empty");
  const auto dims = t->sizes();
  const int current_batch_size = dims[0];
  if (current_batch_size == max_batch_size_) {
    return max_batch_size_;
  }

  // We still need to adjust output size but we can skip the shape inference as
  // it was done before.
  if (output_reshape_info_.count(current_batch_size)) {
    return current_batch_size;
  }

  extractOutputBatchSizes(current_batch_size);

  return current_batch_size;
}

template <>
void OnnxifiOp<CPUContext>::adjustOutputBatchSizes(int current_batch_size) {
  auto it = output_reshape_info_.find(current_batch_size);
  CAFFE_ENFORCE(
      it != output_reshape_info_.end(),
      "Cannot find current_batch_size ",
      current_batch_size,
      " in output_reshape_info_");
  const auto& output_reshape_info = it->second;
  CPUContext context;
  Tensor tmp(CPU);
  for (int i = 0; i < OutputSize(); ++i) {
    Tensor* output_tensor = quantized_outputs_[i]
        ? (&this->template Output<int8::Int8TensorCPU>(i)->t)
        : Output(i);
    const auto& end = output_reshape_info.ends[i];
    if (output_reshape_info.fast_path[i]) {
      output_tensor->ShrinkTo(end.data<int32_t>()[0]);
    } else {
      // We need to use generic Slice
      SliceImpl<int32_t, CPUContext>(
          &tmp, *output_tensor, output_reshape_info.begins[i], end, &context);
      output_tensor->CopyFrom(tmp);
    }
  }
}

template <>
void OnnxifiOp<CPUContext>::setOutputShapeAndType(
    int output_idx,
    c10::SmallVector<int64_t, 4>& tensor_dims_int64) {
  tensor_dims_int64.clear();
  std::vector<size_t> tensor_dims;
  uint64_t type = ONNXIFI_DATATYPE_FLOAT32;
  const auto it = output_shape_hints_.find(output_idx);
  CAFFE_ENFORCE(
      it != output_shape_hints_.end(),
      "Cannot find shape hint for output: ",
      output_names_[output_idx]);
  const auto& info = it->second;
  std::copy(
      info.dims.begin(), info.dims.end(), std::back_inserter(tensor_dims));
  type = it->second.onnxifi_type;
  auto& tensor_descriptor = output_desc_[output_idx];
  tensor_descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  tensor_descriptor.dimensions = tensor_dims.size();
  CAFFE_ENFORCE(
      tensor_descriptor.dimensions != 0, tensor_descriptor.name, " has 0 dim");
  auto& output_shape = output_shapes_max_bs_[output_idx];
  output_shape.clear();
  output_shape.insert(
      output_shape.begin(), tensor_dims.cbegin(), tensor_dims.cend());
  tensor_descriptor.shape = output_shape.data();
  std::copy(
      tensor_dims.cbegin(),
      tensor_dims.cend(),
      std::back_inserter(tensor_dims_int64));

  // Setup the output C2 tensor
  if (!info.quantized) {
    // Normal Tensor
    auto* output_tensor = Output(
        output_idx,
        tensor_dims_int64,
        at::dtype(OnnxifiTypeToDataType(type)).device(CPU));
    setOutputTensorDescriptorTypeAndBuffer(
        type, output_tensor, &tensor_descriptor);
  } else if (info.quantizationParams == 1) {
    // single quantizer, output Int8Tensor
    auto* output_tensor =
        this->template Output<int8::Int8TensorCPU>(output_idx);
    output_tensor->t.Resize(tensor_dims_int64);
    setOutputTensorDescriptorTypeAndBuffer(
        type, &output_tensor->t, &tensor_descriptor);
    tensor_descriptor.quantizationParams = 1;
    tensor_descriptor.quantizationAxis = 1;
    tensor_descriptor.scales = &output_tensor->scale;
    tensor_descriptor.biases = &output_tensor->zero_point;
  } else {
    CAFFE_THROW(
        "OnnxifiOp does not support output tensor with multi-quantization params: ",
        output_names_[output_idx]);
  }
}

string mapOnnxStateToString(onnxEventState state) {
  switch (state) {
    case ONNXIFI_EVENT_STATE_NONSIGNALLED:
      return "ONNXIFI_EVENT_STATE_NONSIGNALLED";
    default:
      return "ONNXIFI_EVENT_STATE_STRING_NOT_MAPPED";
  }
}

string mapOnnxStatusToString(onnxStatus status) {
  switch (status) {
    case ONNXIFI_STATUS_SUCCESS:
      return "ONNXIFI_STATUS_SUCCESS";
    case ONNXIFI_STATUS_FALLBACK:
      return "ONNXIFI_STATUS_FALLBACK";
    case ONNXIFI_STATUS_INVALID_ID:
      return "ONNXIFI_STATUS_INVALID_ID";
    case ONNXIFI_STATUS_INVALID_SIZE:
      return "ONNXIFI_STATUS_INVALID_SIZE";
    case ONNXIFI_STATUS_INVALID_POINTER:
      return "ONNXIFI_STATUS_INVALID_POINTER";
    case ONNXIFI_STATUS_INVALID_PROTOBUF:
      return "ONNXIFI_STATUS_INVALID_PROTOBUF";
    case ONNXIFI_STATUS_INVALID_MODEL:
      return "ONNXIFI_STATUS_INVALID_MODEL";
    case ONNXIFI_STATUS_INVALID_BACKEND:
      return "ONNXIFI_STATUS_INVALID_BACKEND";
    case ONNXIFI_STATUS_INVALID_GRAPH:
      return "ONNXIFI_STATUS_INVALID_GRAPH";
    case ONNXIFI_STATUS_INVALID_EVENT:
      return "ONNXIFI_STATUS_INVALID_EVENT";
    case ONNXIFI_STATUS_INVALID_STATE:
      return "ONNXIFI_STATUS_INVALID_STATE";
    case ONNXIFI_STATUS_INVALID_NAME:
      return "ONNXIFI_STATUS_INVALID_NAME";
    case ONNXIFI_STATUS_INVALID_SHAPE:
      return "ONNXIFI_STATUS_INVALID_SHAPE";
    case ONNXIFI_STATUS_INVALID_DATATYPE:
      return "ONNXIFI_STATUS_INVALID_DATATYPE";
    case ONNXIFI_STATUS_INVALID_MEMORY_TYPE:
      return "ONNXIFI_STATUS_INVALID_MEMORY_TYPE";
    case ONNXIFI_STATUS_INVALID_MEMORY_LOCATION:
      return "ONNXIFI_STATUS_INVALID_MEMORY_LOCATION";
    case ONNXIFI_STATUS_INVALID_FENCE_TYPE:
      return "ONNXIFI_STATUS_INVALID_FENCE_TYPE";
    case ONNXIFI_STATUS_INVALID_PROPERTY:
      return "ONNXIFI_STATUS_INVALID_PROPERTY";
    case ONNXIFI_STATUS_UNSUPPORTED_TAG:
      return "ONNXIFI_STATUS_UNSUPPORTED_TAG";
    case ONNXIFI_STATUS_UNSUPPORTED_VERSION:
      return "ONNXIFI_STATUS_UNSUPPORTED_VERSION";
    case ONNXIFI_STATUS_UNSUPPORTED_OPERATOR:
      return "ONNXIFI_STATUS_UNSUPPORTED_OPERATOR";
    case ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE:
      return "ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE";
    case ONNXIFI_STATUS_UNSUPPORTED_SHAPE:
      return "ONNXIFI_STATUS_UNSUPPORTED_SHAPE";
    case ONNXIFI_STATUS_UNSUPPORTED_DATATYPE:
      return "ONNXIFI_STATUS_UNSUPPORTED_DATATYPE";
    case ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE:
      return "ONNXIFI_STATUS_UNSUPPORTED_MEMORY_TYPE";
    case ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE:
      return "ONNXIFI_STATUS_UNSUPPORTED_FENCE_TYPE";
    case ONNXIFI_STATUS_UNSUPPORTED_PROPERTY:
      return "ONNXIFI_STATUS_UNSUPPORTED_PROPERTY";
    case ONNXIFI_STATUS_UNIDENTIFIED_NAME:
      return "ONNXIFI_STATUS_UNIDENTIFIED_NAME";
    case ONNXIFI_STATUS_MISMATCHING_SHAPE:
      return "ONNXIFI_STATUS_MISMATCHING_SHAPE";
    case ONNXIFI_STATUS_MISMATCHING_DATATYPE:
      return "ONNXIFI_STATUS_MISMATCHING_DATATYPE";
    case ONNXIFI_STATUS_NO_SYSTEM_MEMORY:
      return "ONNXIFI_STATUS_NO_SYSTEM_MEMORY";
    case ONNXIFI_STATUS_NO_DEVICE_MEMORY:
      return "ONNXIFI_STATUS_NO_DEVICE_MEMORY";
    case ONNXIFI_STATUS_NO_SYSTEM_RESOURCES:
      return "ONNXIFI_STATUS_NO_SYSTEM_RESOURCES";
    case ONNXIFI_STATUS_NO_DEVICE_RESOURCES:
      return "ONNXIFI_STATUS_NO_DEVICE_RESOURCES";
    case ONNXIFI_STATUS_BACKEND_UNAVAILABLE:
      return "ONNXIFI_STATUS_BACKEND_UNAVAILABLE";
    case ONNXIFI_STATUS_INTERNAL_ERROR:
      return "ONNXIFI_STATUS_INTERNAL_ERROR";
    case ONNXIFI_STATUS_FATAL_ERROR:
      return "ONNXIFI_STATUS_FATAL_ERROR";
    default:
      return "ONNXIFI_STATUS_STRING_NOT_MAPPED";
  }
}

template <>
bool OnnxifiOp<CPUContext>::RunOnDevice() {
  CAFFE_ENFORCE_EQ(input_desc_.size(), InputSize());
  for (auto i: c10::irange(InputSize())) {
    auto& tensor_descriptor = input_desc_[i];
    tensor_descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
    tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
    at::IntArrayRef tensor_dims;
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    if (this->template InputIsType<int8::Int8TensorCPU>(i)) {
      const auto& input_tensor_int8 =
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          this->template Input<int8::Int8TensorCPU>(i);
      const auto& cpu_tensor = input_tensor_int8.t;
      tensor_dims = cpu_tensor.sizes();
      setInputTensorDescriptorTypeAndBuffer(
          input_tensor_int8, &tensor_descriptor);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      const auto& input_tensor = Input(i);
      tensor_dims = input_tensor.sizes();
      setInputTensorDescriptorTypeAndBuffer(input_tensor, &tensor_descriptor);
    }
    auto& input_shape = input_shapes_[i];
    input_shape.clear();
    input_shape.insert(
        input_shape.begin(), tensor_dims.cbegin(), tensor_dims.cend());
    tensor_descriptor.dimensions = tensor_dims.size();
    tensor_descriptor.shape = input_shape.data();
  }

  CAFFE_ENFORCE_EQ(output_desc_.size(), OutputSize());
  c10::SmallVector<int64_t, 4> tensor_dims_int64;
  for (auto i: c10::irange(OutputSize())) {
    setOutputShapeAndType(i, tensor_dims_int64);
  }
  bool ext_supported = false;
  onnxMemoryFenceV1 input_fence;
  onnxMemoryFenceV1 output_fence;
  std::vector<int> output_batch_sizes;
  int current_batch_size = max_batch_size_;
#ifdef ONNXIFI_ENABLE_EXT
  /**
   * If onnxifi extension mode is enabled,
   * and onnxSetIOAndRunGraph is supported in backend,
   * then we run through this workflow;
   * Else we fallback to non-onnxifi-extension workflow.
   **/
  if (onnxSetIOAndRunGraphPointer_ != nullptr) {
    ext_supported = true;
    output_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
    output_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
    traces_.reset();
    if (enable_tracing_) {
      traces_ = std::shared_ptr<onnxTraceEventList>(
          new onnxTraceEventList(), [this](onnxTraceEventList* p) {
            if (p && onnxReleaseTraceEventsPointer_) {
              CAFFE_ENFORCE_EQ(
                  (*onnxReleaseTraceEventsPointer_)(p), ONNXIFI_STATUS_SUCCESS);
            }
            delete p;
          });
      traces_->numEvents = 0;
    }

    const onnxStatus status = (*onnxSetIOAndRunGraphPointer_)(
        graph_,
        input_desc_.size(),
        input_desc_.data(),
        output_desc_.size(),
        output_desc_.data(),
        &output_fence,
        traces_.get());
    CAFFE_ENFORCE_EQ(
        status,
        ONNXIFI_STATUS_SUCCESS,
        "Reason: onnxSetIOAndRunGraph returned status code ",
        mapOnnxStatusToString(status));

    // Check if we should rely on Onnxifi to provide current batch size
    if (use_onnxifi_batch_size_ && onnxGetCurrentBatchSizePointer_ != nullptr) {
      int64_t onnxifiBatchSize;
      if ((*onnxGetCurrentBatchSizePointer_)(&onnxifiBatchSize) == ONNXIFI_STATUS_SUCCESS) {
        current_batch_size = onnxifiBatchSize;

        if (current_batch_size != max_batch_size_ &&
            output_reshape_info_.count(current_batch_size) == 0) {
          extractOutputBatchSizes(current_batch_size);
        }
      } else {
        current_batch_size = extractOutputBatchSizes();
      }
    } else {
      current_batch_size = extractOutputBatchSizes();
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    onnxEventState eventState;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    onnxStatus eventStatus;
    std::string message;
    size_t messageLength = 512;
    message.resize(messageLength);

    CAFFE_ENFORCE_EQ(
        (*onnxWaitEventForPointer_)(
            output_fence.event,
            timeout_,
            &eventState,
            &eventStatus,
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<char*>(message.data()),
            &messageLength),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_EQ(
        eventState,
        ONNXIFI_EVENT_STATE_SIGNALLED,
        "Onnxifi run timeouted out after ",
        timeout_,
        " ms.",
        "Reason: Onnxifi run returned event state code ",
        mapOnnxStateToString(eventState));
    if (eventStatus != ONNXIFI_STATUS_SUCCESS) {
      if (messageLength == 0) {
        CAFFE_THROW("onnxifi internal error");
      } else {
        CAFFE_THROW(message);
      }
    }
    CAFFE_ENFORCE_EQ(
        lib_->onnxReleaseEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);
  }
#endif
  if (!ext_supported) {
    CAFFE_ENFORCE_EQ(
        lib_->onnxSetGraphIO(
            graph_,
            input_desc_.size(),
            input_desc_.data(),
            output_desc_.size(),
            output_desc_.data()),
        ONNXIFI_STATUS_SUCCESS);

    input_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
    input_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitEvent(backend_, &input_fence.event),
        ONNXIFI_STATUS_SUCCESS);
    output_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
    output_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;

    // Call the async run on backend, signal event on input fence and wait for
    // the event on output fence
    CAFFE_ENFORCE_EQ(
        lib_->onnxRunGraph(graph_, &input_fence, &output_fence),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_EQ(
        lib_->onnxSignalEvent(input_fence.event), ONNXIFI_STATUS_SUCCESS);
    current_batch_size = extractOutputBatchSizes();
    CAFFE_ENFORCE_EQ(
        lib_->onnxWaitEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);

    // Destroy the event objects
    CAFFE_ENFORCE_EQ(
        lib_->onnxReleaseEvent(input_fence.event), ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_EQ(
        lib_->onnxReleaseEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);
  }

  if (adjust_quantized_offset_) {
    for (auto i: c10::irange(OutputSize())) {
      if (quantized_outputs_[i]) {
        auto* int8_tensor = this->template Output<int8::Int8TensorCPU>(i);
        int8_tensor->zero_point += adjust_quantized_offset_;
        adjustQuantizedOffset(&int8_tensor->t, adjust_quantized_offset_);
      }
    }
  }

  if (adjust_output_batch_ && current_batch_size != max_batch_size_) {
    adjustOutputBatchSizes(current_batch_size);
  }
  enable_tracing_ = false;
  return true;
}

REGISTER_CPU_OPERATOR(Onnxifi, OnnxifiOp<CPUContext>);
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
        "Initialization pair indicating the mapping of the name between NetDef and ONNX model")
    .Arg(
        "output_resize_hints",
        "A list of key/value pairs indicating which input index to look up for real batch size for the given max output batch size");
} // namespace caffe2
