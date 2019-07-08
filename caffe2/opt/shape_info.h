#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

struct CAFFE2_API QShapeInfo {
  QShapeInfo(float o = 0, float s = 1, uint32_t a = 1) {
    offset.clear();
    scale.clear();
    offset.push_back(o);
    scale.push_back(s);
    axis = a;
  }

  uint32_t axis;
  vector<float> offset;
  vector<float> scale;
};

CAFFE2_API void LoadInt8FCDNNLowPPackedWeightBlobInfoOfBlob(
    std::vector<float>* scale,
    std::vector<float>* offset,
    uint32_t* axis,
    const Blob* b);

struct CAFFE2_API ShapeInfo {
  enum DimType : int8_t { UNKNOWN = 0, CONSTANT = 1, BATCH = 2, SEQ = 3 };
  ShapeInfo(bool q = false) : is_quantized(q) {}
  ShapeInfo(DimType t, TensorShape&& s, bool q = false)
      : dim_type(t), shape(std::move(s)), is_quantized(q) {}
  ShapeInfo(DimType t, const TensorShape& s, bool q = false)
      : dim_type(t), shape(s), is_quantized(q) {}

  ShapeInfo(bool q, const QShapeInfo& info) : is_quantized(q), q_info(info) {}
  ShapeInfo(DimType t, TensorShape&& s, bool q, const QShapeInfo& info)
      : dim_type(t), shape(std::move(s)), is_quantized(q), q_info(info) {}
  ShapeInfo(DimType t, const TensorShape& s, bool q, const QShapeInfo& info)
      : dim_type(t), shape(s), is_quantized(q), q_info(info) {}

  // type of the shape according its first dim
  DimType dim_type{DimType::UNKNOWN};
  TensorShape shape;

  // quantization related information
  bool is_quantized;
  QShapeInfo q_info;
};

using ShapeInfoMap = std::unordered_map<std::string, ShapeInfo>;

// Generates ShapeInfo from Blob.
ShapeInfo getShapeInfoFromBlob(const Blob* blob);

bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs);

} // namespace caffe2
