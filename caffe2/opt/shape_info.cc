#include "caffe2/opt/shape_info.h"

#include "caffe2/core/operator.h"

namespace caffe2 {

ShapeInfo getShapeInfoFromBlob(const Blob* blob) {
  ShapeInfo shape_info;
  shape_info.shape = GetTensorShapeOfBlob(blob);
  shape_info.dim_type = shape_info.shape.unknown_shape()
      ? ShapeInfo::DimType::UNKNOWN
      : ShapeInfo::DimType::CONSTANT;
  return shape_info;
}

bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs) {
  return lhs.dim_type == rhs.dim_type &&
      lhs.shape.SerializeAsString() == rhs.shape.SerializeAsString();
}

} // namespace caffe2
