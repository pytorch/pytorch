#pragma once

#include <c10/core/Storage.h>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

#ifndef C10_MOBILE
struct CAFFE2_API OfflineTensor {
  // A shell tensor to record shape and dtype
  Tensor shape_tensor{CPU};

  void setShapeAndType(
      const std::vector<int>& sizes,
      at::Device device,
      caffe2::TypeMeta data_type) {
    shape_tensor.unsafeGetTensorImpl()->set_storage_and_dtype(
        at::Storage::create_legacy(device), data_type);
    shape_tensor.Resize(sizes);
    CHECK(!shape_tensor.storage_initialized());
    CHECK(shape_tensor.dtype_initialized());
  }
};

class OfflineTensorShapeFunctions : public ExternalTensorFunctionsBase {
 public:
  explicit OfflineTensorShapeFunctions() : ExternalTensorFunctionsBase() {}
  ~OfflineTensorShapeFunctions() override {}
  bool isQuantized() const override {
    return false;
  }
  bool IsSameMetaType(TypeIdentifier id) override;
  void SetupExternalTensorDescriptor(
      const Blob* blob,
      std::vector<std::vector<uint64_t>>* shapes,
      std::vector<std::vector<float>>* all_scales,
      std::vector<std::vector<int32_t>>* all_offsets,
      ExternalTensorDescriptor* desc) override;
  void LoadInfoOfBlob(
      const Blob* /* unused */,
      std::vector<float>* /* unused */,
      std::vector<float>* /* unused */,
      uint32_t* /* unused */) override {}
  TypeIdentifier GetTypeMetaId() override;
  TypeMeta GetExternalTensorType(const void* c) override;
  vector<int64_t> GetExternalTensorInfo(
      const void* c,
      size_t* capacity,
      DeviceOption* device) override;
};
#endif
} // namespace caffe2
