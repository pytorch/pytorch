#pragma once

#include <iostream>
#include <string>
#include <c10/macros/Macros.h>

namespace c10 {

// Semantically, a dispatch key identifies a possible "level" in our
// dispatch, for which a handler may be registered.  Traditional
// backends like CPU and CUDA get dispatch keys; however, so do
// "wrapping" layers like Variable (for autograd handling).
//
// In implementation terms, the dispatch key identifies a specific "bit" in a
// DispatchKeySet.  Higher bit indexes get handled by dispatching first (because
// we "count leading zeros" when we extract the highest priority dispatch
// key.)
enum class DispatchKey : uint8_t {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // This is not a "real" tensor id, but it exists to give us a "nullopt"
  // element we can return for cases when a DispatchKeySet contains no elements.
  // You can think a more semantically accurate definition of DispatchKey is:
  //
  //    using DispatchKey = optional<RealDispatchKey>
  //
  // and Undefined == nullopt.  We didn't actually represent
  // it this way because optional<RealDispatchKey> would take two
  // words, when DispatchKey fits in eight bits.

  Undefined = 0,

  // Define an alias for Undefined to represent CatchAll (long term
  // this will get eliminated, but for now it's convenient)
  CatchAll = Undefined,



  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // A "backend" is colloquially used to refer to handlers for dispatch
  // which actually implement the numerics of an operation in question.
  //
  // Due to the nature of the enum, these backends are specified in
  // an ordered way, but for most backends this order is not semantically
  // meaningful (e.g., it's valid to reorder these backends without changing
  // semantics).  The only situation when backend ordering is meaningful
  // is when the backend participates in multiple dispatch with another
  // backend; e.g., CPU and SparseCPU (sparse must have
  // higher priority).

  // Here are backends which you think of as traditionally specifying
  // how to implement operations on some device.
  CPU,    // registered at build/aten/src/ATen/CPUType.cpp
  CUDA,   // registered at build/aten/src/ATen/CUDAType.cpp
  HIP,    // NB: I think this is not actually used, due to Note [Masquerading as CUDA]
  MSNPU,  // unused externally, but tested at test/cpp_extensions/msnpu_extension.cpp
  XLA,    // lives out of tree at https://github.com/pytorch/xla

  // These are Caffe2 device types which we grandfathered into
  // DispatchKey.
  // TODO: Caffe2-only DispatchKeys actually should be removed from this enum
  // and just simply be undispatchable.
  MKLDNN, // (MKLDNN is treated as another "device" in Caffe2)
  OpenGL,
  OpenCL,
  IDEEP,

  // Here are backends which specify more specialized operators
  // based on the dtype of the tensor.
  QuantizedCPU, // registered at build/aten/src/ATen/QuantizedCPUType.cpp
  ComplexCPU,   // lives out of tree at https://gitlab.com/pytorch-complex/pytorch-cpu-strided-complex
  ComplexCUDA,  // and https://gitlab.com/pytorch-complex/pytorch-cuda-strided-complex
                        // tested at test/cpp_extensions/complex_registration_extension.cpp
                        // TODO: Remove Complex dispatch keys when Complex is moved in tree

  // This backend is to support custom RNGs; it lets you go
  // to a different kernel if you pass in a generator that is not a
  // traditional CPUGeneratorImpl/CUDAGeneratorImpl.  To make use of this
  // key:
  //  1) set it as a second parameter of at::Generator constructor call in
  //     the user-defined PRNG class.
  //  2) use it as a dispatch key while registering custom kernels
  //     (templatized kernels specialized for user-defined PRNG class)
  // intended for out of tree use; tested by aten/src/ATen/test/rng_test.cpp
  CustomRNGKeyId,

  // Here are backends which specify more specialized operators
  // based on the layout of the tensor.  Note that the sparse backends
  // are one case where ordering matters: sparse multi-dispatches with
  // the corresponding dense tensors, and must be handled before them.
  MkldnnCPU,  // registered at build/aten/src/ATen/MkldnnCPUType.cpp
                      // NB: not to be confused with MKLDNN, which is Caffe2 only
  SparseCPU,  // registered at build/aten/src/ATen/SparseCPUType.cpp
  SparseCUDA, // registered at build/aten/src/ATen/SparseCUDAType.cpp
  SparseHIP,  // TODO: I think this is not actually used, due to Note [Masquerading as CUDA]

  // Here are reserved backends for user-defined backends, see Note [Private use DispatchKey]
  // To see some example about how to use this, check out MSNPU
  PrivateUse1,
  PrivateUse2,
  PrivateUse3,

  // In some situations, it is not immediately obvious what the correct
  // backend for function is, because the function in question doesn't
  // have any "tensor" arguments.  In this case, a BackendSelect function
  // can be registered to implement the custom determination of the
  // correct backend.
  BackendSelect,



  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AUTOGRAD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // All backends are oblivious to autograd; autograd is handled as a
  // layer which happens on top of all backends.  It inspects the autograd
  // metadata of all inputs, determines what autograd metadata should be
  // constructed by the output, and otherwise defers to the backend to
  // actually do the numeric computation.  Autograd contains
  // the bulk of this logic.
  Autograd,

  // Pre-autograd dispatch keys allow backends to override the autograd behavior
  // (aka Autograd) for operators which have a Variable kernel
  // already registered.  For example, XLA wants to define autograd for
  // einsum directly.  Registering a custom autograd implementation at the
  // XLA key won't work because we process Autograd
  // before XLA.  This key has higher priority and gets processed
  // first.  You generally should NOT redispatch after handling autograd
  // here (since that would result in execution of the Autograd
  // operator, which you're trying to skip).  In PreAutograd implementations,
  // you are responsible for handling autograd yourself, or deferring to other
  // operators which support autograd.
  XLAPreAutograd,

  // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
  // and inputs are saved for backward in the post-autocast type.
  Autocast,

  // Here are some reserved pre-autograd keys for user-defined backends, see Note [Private use DispatchKey]
  PrivateUse1_PreAutograd,
  PrivateUse2_PreAutograd,
  PrivateUse3_PreAutograd,



  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // There are a number of alternative modes which may want to handle before
  // autograd; for example, error checking, tracing, profiling or vmap.  They
  // go here.

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a single
  // process test.  Use it by creating a TensorImpl with this DispatchKey, and
  // then registering operators to operate on this type id.  See
  // aten/src/ATen/test/backend_fallback_test.cpp for a usage example.
  TESTING_ONLY_GenericWrapper,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a ingle
  // process test.  Use it by toggling the mode on and off via
  // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
  // to operate on this type id.  See aten/src/ATen/test/backend_fallback_test.cpp
  // for a usage example
  TESTING_ONLY_GenericMode,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  NumDispatchKeys, // Sentinel

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // The aliases exist for backwards compatibility reasons, they shouldn't
  // be used
  CPUTensorId = CPU,
  CUDATensorId = CUDA,
};



// Note [Private use DispatchKey]
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
// We provide two classes of private user tensor id: regular DispatchKeys
// and PreAutograd DispatchKeys.  DispatchKeys serve the role of ordinary "backend"
// DispatchKeys; if you were adding support for a new type of accelerator, you
// would use a DispatchKey, and reuse autograd definitions already defined in
// PyTorch for operators you define.  PreAutograd DispatchKeys serve as "wrapper"
// DispatchKeys: they are most appropriate for tensors that compose multiple
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
static inline DispatchKey XLA() {
  return DispatchKey::XLA;
}

// These are some convenience identifiers for dispatch keys which are
// shorter to type than their long counterparts.  Note that some of these
// dispatch keys directly correspond to DeviceType; and most APIs that
// accept DispatchKey also accept DeviceType; e.g.,
// torch::dispatch(torch::kCPU, ...) is also valid.
constexpr DispatchKey kAutograd = DispatchKey::Autograd;

} // namespace c10

namespace torch {
  // Expose the constant, but not the TYPE (DispatchKey is an implementation
  // detail!)
  using c10::kAutograd;
}

// NB: You really shouldn't use this instance; this enum is guaranteed
// to be pretty small so a regular array should be acceptable.
namespace std {
template <>
struct hash<c10::DispatchKey> {
  typedef size_t result_type;
  typedef c10::DispatchKey argument_type;

  size_t operator()(c10::DispatchKey x) const {
    return static_cast<size_t>(x);
  }
};
}
