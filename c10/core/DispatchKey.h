#pragma once

#include <iostream>
#include <string>
#include "c10/macros/Macros.h"

namespace c10 {

// A "bit" in a DispatchKeySet, which may have a unique dispatch handler
// for it.  Higher bit indexes get handled by dispatching first (because
// we "count leading zeros")
enum class DispatchKey : uint8_t {
  // This is not a "real" tensor id, but it exists to give us a "nullopt"
  // element we can return for cases when a DispatchKeySet contains no elements.
  // You can think a more semantically accurate definition of DispatchKey is:
  //
  //    using DispatchKey = optional<RealDispatchKey>
  //
  // and UndefinedTensorId == nullopt.  We didn't actually represent
  // it this way because optional<RealDispatchKey> would take two
  // words, when DispatchKey fits in eight bits.
  UndefinedTensorId = 0,

  // This pool of IDs is not really ordered, but it is merged into
  // the hierarchy for convenience and performance
  CPUTensorId, // PyTorch/Caffe2 supported
  CUDATensorId, // PyTorch/Caffe2 supported
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
  ComplexCUDATensorId, // PyTorch only

  // See Note [Private use TensorId]
  PrivateUse1_TensorId,
  PrivateUse2_TensorId,
  PrivateUse3_TensorId,

  // Sparse has multi-dispatch with dense; handle it first
  SparseCPUTensorId, // PyTorch only
  SparseCUDATensorId, // PyTorch only

  // Custom pseudorandom number generator
  CustomRNGKeyId,

  // WARNING! If you add more "wrapper" style tensor ids (tensor
  // ids which don't get kernels directly defined in native_functions.yaml;
  // examples are tracing or profiling) here, you need to also adjust
  // legacyExtractDispatchKey in c10/core/DispatchKeySet.h to mask them out.

  VariableTensorId,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a single
  // process test.  Use it by creating a TensorImpl with this DispatchKey, and
  // then registering operators to operate on this type id.
  TESTING_ONLY_GenericWrapperTensorId,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a ingle
  // process test.  Use it by toggling the mode on and off via
  // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
  // to operate on this type id.
  TESTING_ONLY_GenericModeTensorId,

  // See Note [Private use TensorId]
  PrivateUse1_PreAutogradTensorId,
  PrivateUse2_PreAutogradTensorId,
  PrivateUse3_PreAutogradTensorId,

  NumDispatchKeys, // Sentinel
};

// Note [Private use TensorId]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Private use tensor IDs are preallocated tensor type IDs for use in user
// applications.  Similar to private use fields in HTTP, they can be used
// by end users for experimental or private applications, without needing
// to "standardize" the tensor ID (which would be done by submitting a PR
// to PyTorch to add your type ID).
//
// Private use tensor IDs are appropriate to use if you want to experiment
// with adding a new tensor type (without having to patch PyTorch first) or
// have a private, non-distributed application that needs to make use of a
// new tensor type.  Private use tensor IDs are NOT appropriate to use for
// libraries intended to be distributed to further users: please contact
// the PyTorch developers to get a type ID registered in this case.
//
// We provide two classes of private user tensor id: regular TensorIds
// and PreAutogradTensorIds.  TensorIds serve the role of ordinary "backend"
// TensorIds; if you were adding support for a new type of accelerator, you
// would use a TensorId, and reuse autograd definitions already defined in
// PyTorch for operators you define.  PreAutogradTensorIds serve as "wrapper"
// TensorIds: they are most appropriate for tensors that compose multiple
// internal tensors, and for cases when the built-in autograd formulas for
// operators are not appropriate.

static_assert(
  static_cast<uint8_t>(DispatchKey::NumDispatchKeys) < 64,
  "DispatchKey is used as index into 64-bit bitmask; you must have less than 64 entries");

C10_API const char* toString(DispatchKey);
C10_API std::ostream& operator<<(std::ostream&, DispatchKey);

// For backwards compatibility with XLA repository
// (I don't want to fix this in XLA right now because there might be
// more renaming coming in the future.)
static inline DispatchKey XLATensorId() {
  return DispatchKey::XLATensorId;
}

} // namespace c10

// NB: You really shouldn't use this instance; this enum is guaranteed
// to be pretty small so a regular array should be acceptable.
namespace std {
template <>
struct hash<c10::DispatchKey> {
  size_t operator()(c10::DispatchKey x) const {
    return static_cast<size_t>(x);
  }
};
}
