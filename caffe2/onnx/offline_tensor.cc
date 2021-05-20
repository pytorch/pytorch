#include "caffe2/onnx/offline_tensor.h"

namespace caffe2 {

#ifndef C10_MOBILE
namespace {
// These constants need to be aligned with onnxifi.h
constexpr uint64_t kONNXIFI_DATATYPE_FLOAT16 = 10;
constexpr uint64_t kONNXIFI_DATATYPE_FLOAT32 = 1;
constexpr uint64_t kONNXIFI_DATATYPE_UINT8 = 2;
constexpr uint64_t kONNXIFI_DATATYPE_INT32 = 6;
constexpr uint64_t kONNXIFI_DATATYPE_INT8 = 3;
constexpr uint64_t kONNXIFI_DATATYPE_INT64 = 7;
constexpr uint64_t kONNXIFI_DATATYPE_INT16 = 5;
constexpr uint64_t kONNXIFI_DATATYPE_UINT16 = 4;
} // namespace

CAFFE_KNOWN_TYPE(OfflineTensor);

bool OfflineTensorShapeFunctions::IsSameMetaType(TypeIdentifier id) {
  return id == TypeMeta::Id<OfflineTensor>();
}

TypeIdentifier OfflineTensorShapeFunctions::GetTypeMetaId() {
  return TypeMeta::Id<OfflineTensor>();
}

TypeMeta OfflineTensorShapeFunctions::GetExternalTensorType(const void* c) {
  const OfflineTensor* offline_tensor =
      reinterpret_cast<const OfflineTensor*>(c);

  return offline_tensor->shape_tensor.dtype();
}

vector<int64_t> OfflineTensorShapeFunctions::GetExternalTensorInfo(
    const void* c,
    size_t* capacity,
    DeviceOption* device) {
  const OfflineTensor* offline_tensor =
      reinterpret_cast<const OfflineTensor*>(c);
  return GetTensorInfo(&(offline_tensor->shape_tensor), capacity, device);
}

void OfflineTensorShapeFunctions::SetupExternalTensorDescriptor(
    const Blob* blob,
    std::vector<std::vector<uint64_t>>* shapes,
    std::vector<std::vector<float>>* /* unused */,
    std::vector<std::vector<int32_t>>* /* unused */,
    ExternalTensorDescriptor* desc) {
  const auto& offline_tensor = blob->template Get<OfflineTensor>();
  const Tensor& shape_tensor = offline_tensor.shape_tensor;

  if (shape_tensor.template IsType<float>()) {
    desc->dataType = kONNXIFI_DATATYPE_FLOAT32;
  } else if (shape_tensor.template IsType<int32_t>()) {
    desc->dataType = kONNXIFI_DATATYPE_INT32;
  } else if (shape_tensor.template IsType<int8_t>()) {
    desc->dataType = kONNXIFI_DATATYPE_INT8;
  } else if (shape_tensor.template IsType<uint8_t>()) {
    desc->dataType = kONNXIFI_DATATYPE_UINT8;
  } else if (shape_tensor.template IsType<int64_t>()) {
    desc->dataType = kONNXIFI_DATATYPE_INT64;
  } else if (shape_tensor.template IsType<int16_t>()) {
    desc->dataType = kONNXIFI_DATATYPE_INT16;
  } else if (shape_tensor.template IsType<c10::Half>()) {
    desc->dataType = kONNXIFI_DATATYPE_FLOAT16;
  } else if (shape_tensor.template IsType<uint16_t>()) {
    desc->dataType = kONNXIFI_DATATYPE_UINT16;
  } else {
    CAFFE_THROW("Unsupported tensor type: ", shape_tensor.dtype().name());
  }
  desc->buffer = 0;

  desc->quantizationParams = 0;
  desc->quantizationAxis = 0;

  // Set up dim and shape
  const auto shape = shape_tensor.sizes();
  desc->dimensions = shape.size();
  shapes->emplace_back(shape.cbegin(), shape.cend());
  desc->shape = shapes->back().data();

  // It is an offline tensor
  desc->isOffline = 1;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EXTERNAL_TENSOR_FUNCTIONS(
    (TypeMeta::Id<OfflineTensor>()),
    OfflineTensorShapeFunctions);
#endif
} // namespace caffe2
