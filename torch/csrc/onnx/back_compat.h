#pragma once

#include <onnx/onnx_pb.h>

namespace torch {
namespace onnx {

// The following constants are defined here to avoid breaking Meta's internal
// usage of ONNX which pre-dates ONNX 1.14 and thus does not support FLOAT8:
// cf. https://github.com/pytorch/pytorch/pull/106379#issuecomment-1675189340
// -abock, 2023-08-25
//
// ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN
constexpr auto TensorProto_DataType_FLOAT8E4M3FN
  = static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(17);
// ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2
constexpr auto TensorProto_DataType_FLOAT8E5M2
  = static_cast<::ONNX_NAMESPACE::TensorProto_DataType>(19);

} // namespace onnx
} // namespace torch
