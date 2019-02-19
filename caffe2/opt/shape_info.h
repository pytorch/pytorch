#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

struct CAFFE2_API ShapeInfo {
  enum DimType : int8_t { UNKNOWN = 0, CONSTANT = 1, BATCH = 2, SEQ = 3 };
  ShapeInfo() {}
  ShapeInfo(DimType t, TensorShape&& s) : dim_type(t), shape(std::move(s)) {}
  ShapeInfo(DimType t, const TensorShape& s) : dim_type(t), shape(s) {}

  // type of the shape according its first dim
  DimType dim_type{DimType::UNKNOWN};
  TensorShape shape;
};

using ShapeInfoMap = std::unordered_map<std::string, ShapeInfo>;

// Generates ShapeInfo from Blob.
ShapeInfo getShapeInfoFromBlob(const Blob* blob);

bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs);

} // namespace caffe2
