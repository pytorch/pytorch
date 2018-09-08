#pragma once

// The legacy mechanism for dispatching operators in ATen is a Type
// object, which is essentially a giant virtual dispatch table
// for every operation we support dynamically dispatching over.
//
// We intend to deprecate this design for a more extensible one
// that permits addition of extra operators *out-of-band*.  However,
// for the time being, it's the only mechanism which works for
// dispatching PyTorch operators, so we are supporting it for now.
//
// The use of Type in ATen/core poses another problem: on a
// mobile build, we don't want to assume that Type is available.
// But all methods on Tensor which route to PyTorch operators
// need to somehow *get* a Type, and then do a virtual call on it.
// How are we going to get the Type?  Why, by another indirection!
//
// This registry is the mechanism for getting a concrete Type.
// For a regular build, we register all types here; for a
// mobile build, there are no registrations and instead we
// return a stub which errors for all functions.
//
// NB: We don't use Registry for this, because we don't want to
// pay for a hash table lookup every time we do an operation.

#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/Backend.h>
#include <ATen/core/ScalarType.h>

namespace at {

struct Type;

struct AT_CORE_API LegacyTypeDeleter {
  using TypeDeleterFun = void(Type*);
  TypeDeleterFun *fn_ = nullptr;
  LegacyTypeDeleter() {}
  /* implicit */ LegacyTypeDeleter(TypeDeleterFun *fn) : fn_(fn) {}
  void operator()(Type * ptr) {
    if (fn_) {
      (*fn_)(ptr);
    }
  }
};

class AT_CORE_API LegacyTypeDispatch {
public:
  using TypeUniquePtr = std::unique_ptr<Type, LegacyTypeDeleter>;
  Type* getNonVariableTypeRaw(Backend p, ScalarType s) {
    return type_registry[static_cast<int>(p)][static_cast<int>(s)].get();
  }
  void registerType(Backend b, ScalarType s, TypeUniquePtr&& t) {
    type_registry[static_cast<int>(b)][static_cast<int>(s)] = std::move(t);
    detail::getVariableHooks().registerVariableTypeFor(this, b, s);
  }
private:
  // NB: type_registry has nullptr for all CUDA backends until
  // CUDA initialization has occurred
  TypeUniquePtr type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
};

AT_CORE_API LegacyTypeDispatch & globalLegacyTypeDispatch();

} // namespace at
