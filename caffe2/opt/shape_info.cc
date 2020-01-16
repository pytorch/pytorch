#include "caffe2/opt/shape_info.h"
#include "caffe2/core/tensor_int8.h"

#include "caffe2/core/operator.h"

namespace caffe2 {

ShapeInfo getShapeInfoFromBlob(const Blob* blob) {
  ShapeInfo shape_info;
  shape_info.shape = GetTensorShapeOfBlob(blob);
  if (!shape_info.shape.unknown_shape()) {
    shape_info.setDimType(std::vector<TensorBoundShape::DimType>(
        shape_info.shape.dims_size(), TensorBoundShape_DimType_CONSTANT));
  }
  if (blob->meta().id() == TypeMeta::Id<int8::Int8TensorCPU>()) {
    shape_info.is_quantized = true;
    LoadInt8TensorInfoOfBlob(
        &shape_info.q_info.scale,
        &shape_info.q_info.offset,
        &shape_info.q_info.axis,
        blob);
  } else {
#ifndef C10_MOBILE
    auto function_ptr =
        ExternalTensorFunctionsBaseRegistry()->Create(blob->meta().id());
    if (function_ptr != nullptr) {
      shape_info.is_quantized = function_ptr->isQuantized();
      function_ptr->LoadInfoOfBlob(
          blob,
          &shape_info.q_info.scale,
          &shape_info.q_info.offset,
          &shape_info.q_info.axis);
    }
#endif
  }
  return shape_info;
}

bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs) {
  return lhs.getDimType() == rhs.getDimType() &&
      lhs.shape.SerializeAsString() == rhs.shape.SerializeAsString();
}

ShapeInfo constructShapeInfoWithDefaultDimType(
    TensorShape shape,
    TensorBoundShape_DimType defaultFirstDimType) {
  std::vector<TensorBoundShape_DimType> dimType(
      shape.dims_size(), TensorBoundShape_DimType_CONSTANT);
  if (dimType.size()) {
    dimType[0] = defaultFirstDimType;
  }
  return ShapeInfo(dimType, shape);
}

} // namespace caffe2
