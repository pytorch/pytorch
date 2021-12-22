#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <ostream>
#include <string>
#include <vector>
#include <stdexcept>

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
//
//

// NOTE: Keep the list in sync with `DispatchKey` in tools/codegen/model.py
enum class BackendBit : uint8_t {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // A "backend" is colloquially used to refer to handlers for dispatch
  // which actually implement the numerics of an operation in question.
  //
  // Due to the nature of the enum, these backends are specified in
  // an ordered way, but for most backends this order is not semantically
  // meaningful (e.g., it's valid to reorder these backends without changing
  // semantics).  The only situation when backend ordering is meaningful
  // is when the backend participates in multiple dispatch with another
  // backend; e.g., CPU and CUDA (cuda must have higher priority).

  // These keys don't correspond to individual kernels.
  // Instead, they represent the backends that are allowed to override specific pieces of functionality:
  // - dense kernels (e.g. DispatchKey::CPU)
  // - sparse kernels (e.g. DispatchKey::SparseCPU)
  // - quantized kernels (e.g. DispatchKey::QuantizedCPU)
  // - autograd kernels (e.g. DispatchKey::AutogradCPU)
  // We reserve space in the runtime operator table for this full cross product of [backends] x [functionality]
  // [For backends in this list] x [keys below that are explicitly marked as having per-backend functionality]

  InvalidBit = 0,
  CPUBit,
  CUDABit,
  HIPBit,
  XLABit,
  MLCBit,
  XPUBit,
  HPUBit,
  VEBit,
  LazyBit,
  NestedTensorBit,
  PrivateUse1Bit,
  PrivateUse2Bit,
  PrivateUse3Bit,
  // Define an alias key to represent end of backend dispatch keys.
  // If you add new backend keys after PrivateUse3, please also update it here.
  // (But you shouldn't: private use keys should have higher precedence than
  // all built-in keys)
  EndOfBackendKeys = PrivateUse3Bit,
};

enum class DispatchKey : uint16_t {

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

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Functionality Keys ~~~~~~~~~~~~~~~~~~~~~~ //
  // [Note: Functionality Dispatch Keys]
  // TODO: fill this in.
  // See [Note: Per-Backend Functionality Dispatch Keys]

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Dense,

  // Below are non-extensible backends.
  // These are backends that currently don't have their own overrides for Autograd/Sparse/Quantized kernels,
  // and we therefore don't waste space in the runtime operator table allocating space for them.
  // If any of these backends ever need to customize, e.g., Autograd, then we'll need to add a DispatchKey::*Bit for them.

  FPGA, // Xilinx support lives out of tree at
  // https://gitlab.com/pytorch-complex/vitis_kernels

  // ONNX Runtime, lives out of tree at https://github.com/pytorch/ort and
  // https://github.com/microsoft/onnxruntime, and is also used to test general
  // backend/extension machinery in the core. cf:
  // - test/cpp_extensions/ort_extension.cpp
  // - test/test_torch.py
  // - aten/src/ATen/test/extension_backend_test.cpp
  ORT,

  Vulkan,
  Metal,

  // A meta tensor is a tensor without any data associated with it.  (They
  // have also colloquially been referred to as tensors on the "null" device).
  // A meta tensor can be used to dry run operators without actually doing any
  // computation, e.g., add on two meta tensors would give you another meta
  // tensor with the output shape and dtype, but wouldn't actually add anything.
  Meta,

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Quantized,

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
  MkldnnCPU, // registered at build/aten/src/ATen/RegisterMkldnnCPU.cpp
  // NB: not to be confused with MKLDNN, which is Caffe2 only

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Sparse,

  SparseCsrCPU,
  SparseCsrCUDA,

  // In some situations, it is not immediately obvious what the correct
  // backend for function is, because the function in question doesn't
  // have any "tensor" arguments.  In this case, a BackendSelect function
  // can be registered to implement the custom determination of the
  // correct backend.
  BackendSelect,

  Python,

  // The named dispatch key is set for any tensors with named dimensions.
  // Although we have a dispatch key for named tensors, for historical reasons,
  // this dispatch key doesn't do any of the substantive functionality for named
  // tensor (though, hypothetically, it could!)  At the moment, it's just
  // responsible for letting us give good error messages when operations
  // don't support named tensors.
  //
  // NB: If you ever consider moving named tensor functionality into
  // this dispatch key, note that it might be necessary add another dispatch
  // key that triggers before composite operators, in case a composite operator
  // has named dimension propagation that doesn't match that of its
  // constituent parts.
  Named,

  // The Conjugate dispatch key is set for any tensors that need to perform
  // conjugation
  // This is implemented at a dispatch level right before any backends run
  Conjugate,

  // The Negative dispatch key is set for any tensors that need to perform
  // negation
  // This is implemented at a dispatch level right before any backends run
  Negative,

  ZeroTensor, // registered at build/aten/src/ATen/RegisterZeroTensor.cpp

  // See Note [Out-of-tree vmap+grad prototype]. The purpose of this key
  // is to insert code after the "autograd subsystem" runs, so this key should
  // be directly after ADInplaceOrView and all of the autograd keys.
  FuncTorchDynamicLayerBackMode,

  // Note [ADInplaceOrView key]
  // ADInplaceOrView key is used by inplace or view ops to register a kernel
  // that does additional setup for future autograd computation.
  //
  // 1. For inplace ops this kernel does version bump
  // 2. For view ops this kernel does `as_view` setup where we properly setup
  //    DifferentiableViewMeta on the view tensors.
  //
  // For other ops it's fallthrough kernel since there's no extra
  // work to do.
  //
  // Note [Dream: skip VariableType kernel when requires_grad=false]
  //
  // In an ideal world where we can skip VariableType kernel for inputs
  // with requires_grad=false, instead of a fallthrough kernel, we'll
  // register a kernel shown below to all functional ops as well:
  // torch::Tensor my_functional_op(...) {
  //   {
  //     // Note for every op in VariableType, you need to go through
  //     // `AutoDispatchBelowADInplaceOrView` guard exactly once to add the
  //     // key to TLS excluded set. If you don't go through it at all,
  //     // inplace/view ops called through `at::` inside your backend
  //     // kernel will dispatch to ADInplaceOrView kernels and do a lot
  //     // of extra work.
  //     at::AutoDispatchBelowADInplaceOrView guard;
  //     at::redispatch::my_functional_op(...);
  //   }
  // }
  // But this work is currently blocked since it adds an extra dispatch
  // for all ops and it's non-trivial overhead at model level(a few percents).
  // Thus our current approach takes advantage of the fact every kernel go
  // through VariableType kernel first and pulls the
  // `at::AutoDispatchBelowADInplaceOrView` guard of functional ops
  // up to the `VariableType` kernel. Thus we only add the extra dispatch
  // to view/inplace ops to minimize its perf impact to real models.
  ADInplaceOrView,
  // Note [Alias Dispatch Key : Autograd]
  // All backends are oblivious to autograd; autograd is handled as a
  // layer which happens on top of all backends. It inspects the autograd
  // metadata of all inputs, determines what autograd metadata should be
  // constructed by the output, and otherwise defers to the backend to
  // actually do the numeric computation.  Autograd contains
  // the bulk of this logic.

  // Autograd is now an alias dispatch key which by default maps to all
  // backend-specific autograd keys.
  // Backend-specific allow backends to override the default kernel registered
  // to Autograd key as needed.
  // For example, XLA wants to define autograd for einsum directly.
  // Registering a custom autograd implementation at the XLA key won't work
  // because we process Autograd before XLA.  This key has higher priority and
  // gets processed first.  You generally should NOT redispatch after handling
  // autograd here (since that would result in execution of the Autograd
  // operator, which you're trying to skip).  In AutogradXLA implementations,
  // you are responsible for handling autograd yourself, or deferring to other
  // operators which support autograd.

  // Currently we only have backend-specific autograd keys for CPU/CUDA/XLA and
  // reserved user-defined backends. All other in-tree backends share the
  // AutogradOther key. We can add specific autograd key for those backends
  // upon request.
  AutogradOther,

  // See [Note: Per-Backend Functionality Dispatch Keys]
  AutogradFunctionality,

  Tracer,

  // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
  // and inputs are saved for backward in the post-autocast type.
  AutocastCPU,
  // Naughtily, AutocastCUDA is also being used for XLA.  In the terminal state,
  // it probably should get its own Autocast key
  AutocastCUDA,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // There are a number of alternative modes which may want to handle before
  // autograd; for example, error checking, tracing, profiling or vmap.  They
  // go here.

  FuncTorchBatched, // See Note [Out-of-tree vmap+grad prototype]
  FuncTorchVmapMode, // See Note [Out-of-tree vmap+grad prototype]

  // This is the dispatch key for BatchedTensorImpl, which is used to implement
  // batching rules for vmap.
  Batched,

  // When we are inside a vmap, all tensors dispatch on this key.
  // See Note: [DispatchKey::VmapMode usage] for more details.
  VmapMode,

  FuncTorchGradWrapper, // See Note [Out-of-tree vmap+grad prototype]
  // Alias and mutation removal.
  // If some backends want to opt into only alias removal or only mutation
  // removal,
  // we can consider adding separate keys dedicated to those individual passes.
  // See Note [Functionalization Pass In Core] for details.
  Functionalize,
  FuncTorchDynamicLayerFrontMode, // See Note [Out-of-tree vmap+grad prototype]

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a single
  // process test.  Use it by creating a TensorImpl with this DispatchKey, and
  // then registering operators to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp for a usage example.
  TESTING_ONLY_GenericWrapper,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a ingle
  // process test.  Use it by toggling the mode on and off via
  // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
  // to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp
  // for a usage example
  TESTING_ONLY_GenericMode,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  EndOfFunctionalityKeys, // End of functionality keys.

  // ~~~~~~~~~~~~~~ "Dense" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~~ //
  // Here are backends which you think of as traditionally specifying
  // how to implement operations on some device.

  StartOfDenseBackends,
  CPU, // registered at build/aten/src/ATen/RegisterCPU.cpp
  CUDA, // registered at build/aten/src/ATen/RegisterCUDA.cpp
  HIP, // NB: I think this is not actually used, due to Note [Masquerading as
  // CUDA]
  XLA, // lives out of tree at https://github.com/pytorch/xla
  MLC, // lives out of tree at https://github.com/pytorch/MLCompute
  XPU, // For out of tree Intel's heterogeneous computing plug-in
  HPU, // For out of tree & closed source integration of HPU / Habana
  VE, // For out of tree & closed source integration of SX-Aurora / NEC
  Lazy, // For lazy tensor backends
  NestedTensor, // lives out of tree at https://github.com/pytorch/nestedtensor
  // Here are reserved backends for user-defined backends, see Note [Private use
  // DispatchKey]
  // To see some example about how to use this, check out ORT
  PrivateUse1,
  PrivateUse2,
  PrivateUse3,
  EndOfDenseBackends = PrivateUse3,

  // ~~~~~~~~~~~~~~ "Quantized" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~ //
  // keys starting with an _ are not currently used,
  // but are needed to ensure that every backend is indexed correctly.

  StartOfQuantizedBackends,
  QuantizedCPU, // registered at build/aten/src/ATen/RegisterQuantizedCPU.cpp
  QuantizedCUDA, // registered at build/aten/src/ATen/RegisterQuantizedCUDA.cpp
  _QuantizedHIP,
  _QuantizedXLA,
  _QuantizedMLC,
  QuantizedXPU, // For out of tree Intel's heterogeneous computing plug-in
  _QuantizedHPU,
  _QuantizedVE,
  _QuantizedLazy,
  _QuantizedNestedTensor,
  _QuantizedPrivateUse1,
  _QuantizedPrivateUse2,
  _QuantizedPrivateUse3,
  EndOfQuantizedBackends = _QuantizedPrivateUse3,

  // ~~~~~~~~~~~~~~ "Sparse" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~ //
  // keys starting with an _ are not currently used,
  // but are needed to ensure that every backend is indexed correctly.

  StartOfSparseBackends,
  SparseCPU, // registered at build/aten/src/ATen/RegisterSparseCPU.cpp
  SparseCUDA, // registered at build/aten/src/ATen/RegisterSparseCUDA.cpp
  SparseHIP, // TODO: I think this is not actually used, due to Note
  // [Masquerading as CUDA]
  _SparseXLA,
  _SparseMLC,
  SparseXPU, // For out of tree Intel's heterogeneous computing plug-in
  _SparseHPU,
  SparseVE, // For out of tree & closed source integration of SX-Aurora / NEC
  _SparseLazy,
  _SparseNestedTensor,
  _SparsePrivateUse1,
  _SparsePrivateUse2,
  _SparsePrivateUse3,
  EndOfSparseBackends = _SparsePrivateUse3,

  // ~~~~~~~~~~~~~~ "Autograd" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~ //
  // keys starting with an _ are not currently used,
  // but are needed to ensure that every backend is indexed correctly.

  StartOfAutogradBackends,
  AutogradCPU,
  AutogradCUDA,
  _AutogradHIP,
  AutogradXLA,
  AutogradMLC,
  AutogradXPU,
  AutogradHPU,
  _AutogradVE,
  AutogradLazy,
  AutogradNestedTensor, // lives out of tree at
  // https://github.com/pytorch/nestedtensor
  // Here are some reserved pre-autograd keys for user-defined backends, see
  // Note [Private use DispatchKey]
  AutogradPrivateUse1,
  AutogradPrivateUse2,
  AutogradPrivateUse3,
  EndOfAutogradBackends = AutogradPrivateUse3,

  // ~~~~~~~~~~~~~~~~~~~~~~ Alias Dispatch Keys ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Note [Alias Dispatch Keys]
  // Alias dispatch keys are synthetic dispatch keys which map to multiple
  // runtime dispatch keys. Alisa keys have precedence, but they are always
  // lower precedence than runtime keys. You can register a kernel to an
  // alias key, the kernel might be populated to the mapped runtime keys
  // during dispatch table computation.
  // If a runtime dispatch key has multiple kernels from alias keys, which
  // kernel wins is done based on the precedence of alias keys (but runtime
  // keys always have precedence over alias keys).
  // Alias keys won't be directly called during runtime.

  // See Note [Alias Dispatch Key : Autograd]
  Autograd,
  CompositeImplicitAutograd, // registered at
  // build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
  CompositeExplicitAutograd, // registered at
  // build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp

  // Define an alias key to represent end of alias dispatch keys.
  // If you add new alias keys after Autograd, please also update it here.
  StartOfAliasKeys = Autograd,
  EndOfAliasKeys = CompositeExplicitAutograd, //

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // The aliases exist for backwards compatibility reasons, they shouldn't
  // be used
  CPUTensorId = CPU,
  CUDATensorId = CUDA,
  DefaultBackend = CompositeExplicitAutograd,
  PrivateUse1_PreAutograd = AutogradPrivateUse1,
  PrivateUse2_PreAutograd = AutogradPrivateUse2,
  PrivateUse3_PreAutograd = AutogradPrivateUse3,
  Autocast = AutocastCUDA,
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
// and Autograd DispatchKeys.  DispatchKeys serve the role of ordinary "backend"
// DispatchKeys; if you were adding support for a new type of accelerator, you
// would use a backend DispatchKey, and ideally automatically reuse
// AutogradOther definitions already defined in PyTorch.  AutogradPrivateUse
// DispatchKeys serve as "wrapper" DispatchKeys: they are only necessary for
// tensors that compose multiple internal tensors, and for cases when the
// built-in autograd formulas for operators are not appropriate.

static_assert(
    (static_cast<uint8_t>(BackendBit::EndOfBackendKeys) +
    static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)) <= 64,
    "DispatchKey is used as index into 64-bit bitmask; you must have less than 64 entries");

// Check if a DispatchKey is an alias mapping to other runtime keys.
constexpr bool isAliasDispatchKey(DispatchKey k) {
  return k >= DispatchKey::StartOfAliasKeys && k <= DispatchKey::EndOfAliasKeys;
}

// Check if a DispatchKey is a per-backend functionality key
// [Note: Per-Backend Functionality Dispatch Keys]
// TODO: fill this in.
constexpr bool isPerBackendFunctionalityKey(DispatchKey k) {
  if (k == DispatchKey::Dense ||
      k == DispatchKey::Quantized ||
      k == DispatchKey::Sparse ||
      k == DispatchKey::AutogradFunctionality) {
      return true;
  } else {
      return false;
  }
}

// Note that this includes Undefined in the total count.
// BUT EndOfFunctionalityKeys is its own (placeholder) key.
// e.g. Undefined=0, Dense=1, Sparse=2, EndOfFunctionalityKeys=3.
// In the above example, there are 3 total functionality keys.
constexpr uint8_t num_functionality_keys =
    static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);

// it would be nice if we could cast EndOfBackendKeys to a 4-bit unsigned integer.
// That way we'd automatically get warnings if EndOfBackendKeys happened to grow > 16.
// Sadly, uint4_t is not a C++ type.
constexpr uint8_t num_backends = static_cast<uint8_t>(BackendBit::EndOfBackendKeys);

constexpr uint8_t numPerBackendFunctionalityKeys() {
  uint8_t count = 0;
  for (uint8_t k = 0; k < static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys); ++k) {
    if (isPerBackendFunctionalityKey(static_cast<DispatchKey>(k))) ++count;
  }
  return count;
}

constexpr uint16_t full_backend_mask = (static_cast<uint16_t>(1) << num_backends) - 1;


#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
// See [Note: Trimmed Mobile Dispatch Keys]
constexpr uint16_t num_runtime_entries = 8;
#else
constexpr uint16_t num_runtime_entries =
    num_functionality_keys
  + (numPerBackendFunctionalityKeys() * (num_backends - 1));
#endif

C10_API const char* toString(DispatchKey);
C10_API const char* toString(BackendBit);
C10_API std::ostream& operator<<(std::ostream&, DispatchKey);
C10_API std::ostream& operator<<(std::ostream&, BackendBit);

C10_API DispatchKey getAutogradKeyFromBackend(BackendBit t);

// Parses a string into a dispatch key.
// If the string cannot be correctly parsed, throws an exception.
C10_API c10::DispatchKey parseDispatchKey(const std::string& k);

// These are some convenience identifiers for dispatch keys which are
// shorter to type than their long counterparts.  Note that some of these
// dispatch keys directly correspond to DeviceType; and most APIs that
// accept DispatchKey also accept DeviceType; e.g.,
// torch::dispatch(torch::kCPU, ...) is also valid.
constexpr DispatchKey kAutograd = DispatchKey::Autograd;

// This function relies on the invariant that the dispatch keys between
// StartOfDenseBackends and EndOfAutogradBackends are ordered by backend
// in the same order as `BackendBit`.
// TODO: statically assert this invariant.
constexpr BackendBit toBackendBit(DispatchKey k) {
  if (k >= DispatchKey::StartOfDenseBackends && k <= DispatchKey::EndOfDenseBackends) {
    return static_cast<BackendBit>(static_cast<uint8_t>(k) - static_cast<uint8_t>(DispatchKey::StartOfDenseBackends));
  } else if (k >= DispatchKey::StartOfQuantizedBackends && k <= DispatchKey::EndOfQuantizedBackends) {
    return static_cast<BackendBit>(static_cast<uint8_t>(k) - static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends));
  } else if (k >= DispatchKey::StartOfSparseBackends && k <= DispatchKey::EndOfSparseBackends) {
    return static_cast<BackendBit>(static_cast<uint8_t>(k) - static_cast<uint8_t>(DispatchKey::StartOfSparseBackends));
  } else if (k >= DispatchKey::StartOfAutogradBackends && k <= DispatchKey::EndOfAutogradBackends) {
    return static_cast<BackendBit>(static_cast<uint8_t>(k) - static_cast<uint8_t>(DispatchKey::StartOfAutogradBackends));
  } else {
    return BackendBit::InvalidBit;
  }
}

constexpr DispatchKey toFunctionalityKey(DispatchKey k) {
  if (k <= DispatchKey::EndOfFunctionalityKeys) {
    return k;
  } else if (k <= DispatchKey::EndOfDenseBackends) {
    return DispatchKey::Dense;
  } else if (k <= DispatchKey::EndOfQuantizedBackends) {
    return DispatchKey::Quantized;
  } else if (k <= DispatchKey::EndOfSparseBackends) {
    return DispatchKey::Sparse;
  } else if (k <= DispatchKey::EndOfAutogradBackends) {
    return DispatchKey::AutogradFunctionality;
  } else {
    return DispatchKey::Undefined;
  }
}

// Given (DispatchKey::Dense, DispatchKey::CUDABit), returns DispatchKey::CUDA
// This function relies on the invariant that the dispatch keys between
// StartOfDenseBackends and EndOfAutogradBackends are ordered by backend
// in the same order as `BackendBit`.
// TODO: statically assert this invariant.
constexpr DispatchKey toRuntimePerBackendFunctionalityKey(DispatchKey functionality_k, BackendBit backend_k) {
  if (functionality_k == DispatchKey::Dense) {
    return static_cast<DispatchKey>(
      static_cast<uint8_t>(DispatchKey::StartOfDenseBackends) + static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Sparse) {
    return static_cast<DispatchKey>(
      static_cast<uint8_t>(DispatchKey::StartOfSparseBackends) + static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::Quantized) {
    return static_cast<DispatchKey>(
      static_cast<uint8_t>(DispatchKey::StartOfQuantizedBackends) + static_cast<uint8_t>(backend_k));
  }
  if (functionality_k == DispatchKey::AutogradFunctionality) {
    return static_cast<DispatchKey>(
      static_cast<uint8_t>(DispatchKey::StartOfAutogradBackends) + static_cast<uint8_t>(backend_k));
  }
  return DispatchKey::Undefined;
}

} // namespace c10

namespace torch {
// Expose the constant, but not the TYPE (DispatchKey is an implementation
// detail!)
using c10::kAutograd;
} // namespace torch

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
} // namespace std
