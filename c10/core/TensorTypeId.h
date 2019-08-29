#pragma once

#include <iostream>
#include <string>
#include "c10/macros/Macros.h"

namespace c10 {

// NB: Ordering will be subject to change
enum class TensorTypeId : uint8_t {
  UndefinedTensorId,
  CPUTensorId, // PyTorch/Caffe2 supported
  CUDATensorId, // PyTorch/Caffe2 supported
  SparseCPUTensorId, // PyTorch only
  SparseCUDATensorId, // PyTorch only
  MKLDNNTensorId, // Caffe2 only
  OpenGLTensorId, // Caffe2 only
  OpenCLTensorId, // Caffe2 only
  IDEEPTensorId, // Caffe2 only
  HIPTensorId, // PyTorch/Caffe2 supported
  SparseHIPTensorId, // PyTorch only
  MSNPUTensorId, // PyTorch only
  XLATensorId, // PyTorch only
  MkldnnCPUTensorId,
  QuantizedCPUTensorId, // PyTorch only
  ComplexCPUTensorId, // PyTorch only
  ComplexCUDATensorId // PyTorch only
};

C10_API const char* toString(TensorTypeId);
C10_API std::ostream& operator<<(std::ostream&, TensorTypeId);

// For backwards compatibility with XLA repository
// (I don't want to fix this in XLA right now because there might be
// more renaming coming in the future.)
static inline TensorTypeId XLATensorId() {
  return TensorTypeId::XLATensorId;
}

} // namespace c10

// NB: You really shouldn't use this instance; this enum is guaranteed
// to be pretty small so a regular array should be acceptable.
namespace std {
template <>
struct hash<c10::TensorTypeId> {
  size_t operator()(c10::TensorTypeId x) const {
    return static_cast<size_t>(x);
  }
};
}
