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

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/VariableHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/core/LegacyDeviceTypeInit.h>
#include <c10/core/TensorImpl.h>

namespace at {

struct Type;

struct CAFFE2_API LegacyTypeDeleter {
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

class CAFFE2_API LegacyTypeDispatch {
 public:
  using TypeUniquePtr = std::unique_ptr<Type, LegacyTypeDeleter>;
  // WARNING: This function has the precondition that you have
  // initialized the type you want to call.  This initialization
  // step is generally done by Context, or assumed because you
  // have a Tensor and thus the Type of that Tensor must already
  // be initialized.
  Type* getNonVariableTypeRaw(Backend p, ScalarType s) {
    return type_registry[static_cast<int>(p)].get();
  }
  Type * getNonVariableTypeOpt(Backend p, ScalarType s) {
    if (p != Backend::Undefined) {
      initForDeviceType(backendToDeviceType(p));
      initForScalarType(s);
    }
    auto type = getNonVariableTypeRaw(p, s);

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        return getNonVariableTypeRaw(Backend::Undefined, ScalarType::Undefined);
      }
    }

    return type;
  }

  Type & getNonVariableType(Backend p, ScalarType s) {
    auto* type = getNonVariableTypeOpt(p, s);
    if (!type) AT_ERROR(toString(p), toString(s), "Type is not enabled.");
    return *type;
  }

  Type* getTypeRaw(Backend p, ScalarType s, bool is_variable) {
    auto baseType = getNonVariableTypeRaw(p, s);
    if (is_variable) {
      return &detail::getVariableHooks().getVariableTypeFromBaseType(*baseType);
    } else {
      return baseType;
    }
  }
  Type & getVariableType(Backend p, ScalarType s) {
    auto& baseType = getNonVariableType(p, s);
    return detail::getVariableHooks().getVariableTypeFromBaseType(baseType);
  }
  Type & getType(Backend p, ScalarType s, bool is_variable) {
    if (is_variable) {
      return getVariableType(p, s);
    } else {
      return getNonVariableType(p, s);
    }
  }
  void registerType(Backend b, TypeUniquePtr&& t) {
    type_registry[static_cast<int>(b)] = std::move(t);
    detail::getVariableHooks().registerVariableTypeFor(this, b);
  }
private:
  void initForDeviceType(DeviceType p) {
    static std::once_flag cpu_once;
    static std::once_flag cuda_once;
    if (p == DeviceType::CPU) {
      std::call_once(cpu_once, [] {
        getLegacyDeviceTypeInit().initCPU();
      });
    } else if (p == DeviceType::CUDA) {
      std::call_once(cuda_once, [] {
        getLegacyDeviceTypeInit().initCUDA();
      });
    } else if (p == DeviceType::HIP) {
      std::call_once(cuda_once, [] {
        getLegacyDeviceTypeInit().initHIP();
      });
    }
  }
  void initForScalarType(ScalarType s) {
    static std::once_flag once;
    // Only complex may need initialization
    if (isComplexType(s)) {
      std::call_once(once, [] {
        getLegacyDeviceTypeInit().initComplex();
      });
    }
  }

  // NB: type_registry has nullptr for all CUDA backends until
  // CUDA initialization has occurred
  TypeUniquePtr type_registry
    [static_cast<int>(Backend::NumOptions)];
};

CAFFE2_API LegacyTypeDispatch& globalLegacyTypeDispatch();

struct CAFFE2_API NonVariableTypeMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that has the following effect:
//
// Upon construction: sets NonVariableTypeMode_enabled for the current thread to
// control whether we are in non-Variable-type mode.
//
// Upon destruction: sets NonVariableTypeMode_enabled back to the original value.
//
// See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
struct CAFFE2_API AutoNonVariableTypeMode {
  AutoNonVariableTypeMode(bool enabled) : prev_mode(NonVariableTypeMode::is_enabled()) {
    NonVariableTypeMode::set_enabled(enabled);
  }
  ~AutoNonVariableTypeMode() {
    NonVariableTypeMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

/**
 * Return the Type object corresponding to this Tensor, which we can
 * use to do dynamic dispatch to operators from.  This method is NOT
 * intended to be used by end-users; it is purely an implementation
 * detail.
 *
 * NOTE: We also check `at::NonVariableTypeMode`, and if it's enabled
 * we always return non-Variable type in this function.
 * See NOTE [ Treating Variables as non-Variables in type dispatch ]
 */
inline Type& legacyTensorType(const TensorImpl& tensor) {
  // NB: It's valid to use getTypeRaw here, because the TensorImpl
  // could not have been created without initializing the Type first.
  // NB: This is not actually true via the Caffe2 codepath! But we call
  // initializeLegacyTypeDispatchFor in the right place.
  return *globalLegacyTypeDispatch().getTypeRaw(
      tensorTypeIdToBackend(tensor.type_id()),
      typeMetaToScalarType(tensor.dtype()),
      tensor.is_variable() && !at::NonVariableTypeMode::is_enabled());
}

inline void initializeLegacyTypeDispatchFor(const TensorImpl& tensor) {
  // getType calls the right initialization
  globalLegacyTypeDispatch().getType(
      tensorTypeIdToBackend(tensor.type_id()),
      typeMetaToScalarType(tensor.dtype()),
      tensor.is_variable() && !at::NonVariableTypeMode::is_enabled());
}

} // namespace at
