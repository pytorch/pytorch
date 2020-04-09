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

struct CAFFE2_API ShapeInfo {
  ShapeInfo(bool q = false) : is_quantized(q) {}
  ShapeInfo(
      std::vector<TensorBoundShape_DimType>&& t,
      TensorShape&& s,
      bool q = false)
      : shape(std::move(s)),
        is_quantized(q),
        dim_type(std::move(t)),
        dim_type_is_set(true) {}
  ShapeInfo(
      const std::vector<TensorBoundShape_DimType>& t,
      TensorShape&& s,
      bool q = false)
      : shape(std::move(s)),
        is_quantized(q),
        dim_type(t),
        dim_type_is_set(true) {}
  ShapeInfo(
      const std::vector<TensorBoundShape_DimType>& t,
      const TensorShape& s,
      bool q = false)
      : shape(s), is_quantized(q), dim_type(t), dim_type_is_set(true) {}

  ShapeInfo(bool q, const QShapeInfo& info) : is_quantized(q), q_info(info) {}
  ShapeInfo(
      const std::vector<TensorBoundShape_DimType>& t,
      TensorShape&& s,
      bool q,
      const QShapeInfo& info)
      : shape(std::move(s)),
        is_quantized(q),
        q_info(info),
        dim_type(t),
        dim_type_is_set(true) {}
  ShapeInfo(
      const std::vector<TensorBoundShape_DimType>& t,
      const TensorShape& s,
      bool q,
      const QShapeInfo& info)
      : shape(s),
        is_quantized(q),
        q_info(info),
        dim_type(t),
        dim_type_is_set(true) {}

  void setDimType(const std::vector<TensorBoundShape_DimType>& dim_types) {
    if (shape.dims_size()) {
      CAFFE_ENFORCE_EQ(shape.dims_size(), dim_types.size());
    }
    dim_type = dim_types;
    dim_type_is_set = true;
  }

  void setDimType(int idx, TensorBoundShape_DimType type) {
    CAFFE_ENFORCE(
        dim_type.size() > idx, dim_type.size(), "vs", dim_type.size());
    dim_type[idx] = type;
    dim_type_is_set = true;
  }

  bool dimTypeIsSet() {
    return dim_type_is_set;
  }

  const std::vector<TensorBoundShape_DimType>& getDimType() const {
    return dim_type;
  }

  TensorBoundShape_DimType getDimType(int idx) const {
    if (dim_type.size() > idx) {
      return dim_type[idx];
    } else {
      return TensorBoundShape_DimType_UNKNOWN;
    }
  }

  TensorShape shape;

  // quantization related information
  bool is_quantized;
  QShapeInfo q_info;

 private:
  // type of the shape for every dimension
  // dim_type.size == shape.dims.size
  std::vector<TensorBoundShape_DimType> dim_type;
  bool dim_type_is_set = false;
};

using ShapeInfoMap = std::unordered_map<std::string, ShapeInfo>;

// Generates ShapeInfo from Blob.
ShapeInfo getShapeInfoFromBlob(const Blob* blob);

bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs);

// Construct a ShapeInfo instance from TensorShape and constructed dimType.
// Default first dimension of dimType is BATCH, reason:
// We treat first dimension of hinted shapes as BATCH.
// If there are shape hints on blobs in the workspace,
// since they are already inserted as CONSTANT, it will take effect here.
// For SEQ typed tensors, there are only a few of them and they will be
// handled by BoundShapeInferencer.
CAFFE2_API ShapeInfo constructShapeInfoWithDefaultDimType(
    TensorShape shape,
    TensorBoundShape_DimType defaultFirstDimType =
        TensorBoundShape_DimType_BATCH);

CAFFE2_API void parseShapeInfoMapFromString(const std::string&, ShapeInfoMap&);

} // namespace caffe2
