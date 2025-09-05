#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <optional>

// use anonymous namespace to avoid collisions between differing
// versions of this file that may be included by different sources
namespace {

// forward declare so that the from/to() implementations in the detail
// namespace of library.h where the real work is done can compile.
template <typename T>
StableIValue from(T val);
template <typename T>
T to(StableIValue val);

// =============================================================================
//  helpers for converting between StableIValue and T
// =============================================================================

// note that the signatures for from and to are forward declared in
// stable/stableivalue_conversions.h but defined below to avoid circular
// dependencies where other headers (like tensor-inl.h) will need to/from.

namespace detail {

// =============================================================================
// FROM CONVERSIONS (T -> StableIValue)
// =============================================================================

// Specialization for general copyable types (catch-all) => StableIValue
template <typename T>
struct FromImpl {
  static StableIValue call(T val) {
    static_assert(
        sizeof(T) <= sizeof(StableIValue),
        "StableLibrary stack does not support parameter types larger than 64 bits.");
    static_assert(std::is_trivially_copyable_v<T>);
    // Initialization should be cheap enough; let's give people well-specified
    // reproducible behavior.
    StableIValue result = 0;
    // NOTE [ -Wclass-memaccess ]: reinterpret_cast to suppress
    // overzealous -Wclass-memaccess. (see
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107361) We have a
    // static_assert above that T is trivially copyable, which should be
    // enough.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(&result, reinterpret_cast<const void*>(&val), sizeof(val));
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // if value has size less than sizeof(StableIValue), then only lowest bytes
    // have to be updated
    std::memcpy(
        reinterpret_cast<unsigned char*>(&result) + sizeof(StableIValue) -
            sizeof(val),
        reinterpret_cast<const void*>(&val),
        sizeof(val));
#else
#error "Unexpected or undefined __BYTE_ORDER__"
#endif
    return result;
  }
};

// Specialization for torch::headeronly::ScalarType => StableIValue
// Note that we call into the shim to translate between the user's
// ScalarType and libtorch's ScalarType, which can be different!
// Also note that the list below is not comprehensive, as it does not
// include types that are no longer really used and should probably be
// deprecated (like qint8).
using torch::headeronly::ScalarType;
template <>
struct FromImpl<ScalarType> {
  static StableIValue call(ScalarType val) {
    switch (val) {
      case ScalarType::Byte:
        return from(aoti_torch_dtype_uint8());
      case ScalarType::Char:
        return from(aoti_torch_dtype_int8());
      case ScalarType::Short:
        return from(aoti_torch_dtype_int16());
      case ScalarType::Int:
        return from(aoti_torch_dtype_int32());
      case ScalarType::Long:
        return from(aoti_torch_dtype_int64());
      case ScalarType::Half:
        return from(aoti_torch_dtype_float16());
      case ScalarType::Float:
        return from(aoti_torch_dtype_float32());
      case ScalarType::Double:
        return from(aoti_torch_dtype_float64());
      case ScalarType::ComplexHalf:
        return from(aoti_torch_dtype_complex32());
      case ScalarType::ComplexFloat:
        return from(aoti_torch_dtype_complex64());
      case ScalarType::ComplexDouble:
        return from(aoti_torch_dtype_complex128());
      case ScalarType::Bool:
        return from(aoti_torch_dtype_bool());
      case ScalarType::BFloat16:
        return from(aoti_torch_dtype_bfloat16());
      case ScalarType::Float8_e5m2:
        return from(aoti_torch_dtype_float8_e5m2());
      case ScalarType::Float8_e4m3fn:
        return from(aoti_torch_dtype_float8_e4m3fn());
      case ScalarType::Float8_e5m2fnuz:
        return from(aoti_torch_dtype_float8_e5m2fnuz());
      case ScalarType::Float8_e4m3fnuz:
        return from(aoti_torch_dtype_float8_e4m3fnuz());
      case ScalarType::UInt16:
        return from(aoti_torch_dtype_uint16());
      case ScalarType::UInt32:
        return from(aoti_torch_dtype_uint32());
      case ScalarType::UInt64:
        return from(aoti_torch_dtype_uint64());
      default:
        throw std::runtime_error(
            "Not yet supported ScalarType, please file an issue describing your use case.");
    }
  }
};

// Specialization for std::nullopt_t => StableIValue
template <>
struct FromImpl<std::nullopt_t> {
  static StableIValue call(std::nullopt_t val) {
    return from(nullptr);
  }
};

// Specialization for std::optional => StableIValue
// [Handling std::optional]
// When the schema is represented by an optional type, say int?, then we
// expect the custom extension representation to be a std::optional<int>
// (critically NOT int!). In order for all parameters to be stably parsed and
// handled by our dispatcher, we liaison custom extension parameters through
// boxed kernels, meaning that every value will make its way to be an IValue:
//
// custom extension value --(from)-> StableIValue --(to_ivalue)-> IValue
//
// When the custom extension value is a literal that can be trivially
// casted to StableIValue, e.g., an int, a float, a pointer, this route is
// ...trivial. The below specialization is for a case when the custom
// extension value would NOT fit within a StableIValue: a std::optional.
//
// If the std::optional has no value, it is treated as std::nullopt,
// whose StableIValue representation is from(nullptr). Otherwise, we:
// 1. unwrap the std::optional<T>
// 2. recursively convert its value of type T to a StableIValue
// 3. allocate heap space for said StableIValue
// 4. convert the resulting StableIValue* into a StableIValue
//
// note that this allocates heap memory! which we expect to be cleaned
// up in the to_ivalue() function defined in shim_common.cpp. We
// purposefully hide this implementation detail from the user so that
// all the user needs to know is:
//
// The schema requests an optional (T?) so I must call `from` on a
// std::optional<T> or a std::nullopt.
template <typename T>
struct FromImpl<std::optional<T>> {
  static StableIValue call(const std::optional<T>& val) {
    if (!val.has_value()) {
      return from(std::nullopt);
    }
    return from(new StableIValue(from(val.value())));
  }
};

// Specialization for torch::stable::Tensor => StableIValue
// Returns a new owning reference of the underlying Tensor.
template <>
struct FromImpl<torch::stable::Tensor> {
  static StableIValue call(const torch::stable::Tensor& val) {
    AtenTensorHandle new_ath;
    TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle(val.get(), &new_ath));
    return from(new_ath);
  }
};

// =============================================================================
// TO CONVERSIONS (StableIValue -> T)
// =============================================================================

// Specialization for StableIValue => general copyable types (catch-all)
template <typename T>
struct ToImpl {
  static T call(StableIValue val) {
    static_assert(std::is_trivially_copyable_v<T>);
    // T may not have a default constructor. (For example, it might be
    // c10::Device.) However, std::memcpy implicitly creates a T at the
    // destination. So, we can use a union to work around this lack of
    // default constructor.
    union Result {
      Result() {}
      T t;
    };
    Result result;
    // See NOTE[ -Wclass-memaccess ] above.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(reinterpret_cast<void*>(&result.t), &val, sizeof(result));
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    static_assert(
        sizeof(T) <= sizeof(StableIValue),
        "StableLibrary stack does not support parameter types larger than 64 bits.");
    // if value has size less than sizeof(StableIValue), then only lowest bytes
    // have to be updated
    std::memcpy(
        reinterpret_cast<void*>(&result.t),
        reinterpret_cast<unsigned char*>(&val) + sizeof(StableIValue) -
            sizeof(result),
        sizeof(result));
#else
#error "Unexpected or undefined __BYTE_ORDER__"
#endif
    return result.t;
  }
};

// Specialization for StableIValue => torch::headeronly::ScalarType
template <>
struct ToImpl<ScalarType> {
  static ScalarType call(StableIValue val) {
    int32_t shim_scalartype = to<int32_t>(val);
    if (shim_scalartype == aoti_torch_dtype_uint8()) {
      return ScalarType::Byte;
    } else if (shim_scalartype == aoti_torch_dtype_int8()) {
      return ScalarType::Char;
    } else if (shim_scalartype == aoti_torch_dtype_int16()) {
      return ScalarType::Short;
    } else if (shim_scalartype == aoti_torch_dtype_int32()) {
      return ScalarType::Int;
    } else if (shim_scalartype == aoti_torch_dtype_int64()) {
      return ScalarType::Long;
    } else if (shim_scalartype == aoti_torch_dtype_float16()) {
      return ScalarType::Half;
    } else if (shim_scalartype == aoti_torch_dtype_float32()) {
      return ScalarType::Float;
    } else if (shim_scalartype == aoti_torch_dtype_float64()) {
      return ScalarType::Double;
    } else if (shim_scalartype == aoti_torch_dtype_complex32()) {
      return ScalarType::ComplexHalf;
    } else if (shim_scalartype == aoti_torch_dtype_complex64()) {
      return ScalarType::ComplexFloat;
    } else if (shim_scalartype == aoti_torch_dtype_complex128()) {
      return ScalarType::ComplexDouble;
    } else if (shim_scalartype == aoti_torch_dtype_bool()) {
      return ScalarType::Bool;
    } else if (shim_scalartype == aoti_torch_dtype_bfloat16()) {
      return ScalarType::BFloat16;
    } else if (shim_scalartype == aoti_torch_dtype_float8_e5m2()) {
      return ScalarType::Float8_e5m2;
    } else if (shim_scalartype == aoti_torch_dtype_float8_e4m3fn()) {
      return ScalarType::Float8_e4m3fn;
    } else if (shim_scalartype == aoti_torch_dtype_float8_e5m2fnuz()) {
      return ScalarType::Float8_e5m2fnuz;
    } else if (shim_scalartype == aoti_torch_dtype_float8_e4m3fnuz()) {
      return ScalarType::Float8_e4m3fnuz;
    } else if (shim_scalartype == aoti_torch_dtype_uint16()) {
      return ScalarType::UInt16;
    } else if (shim_scalartype == aoti_torch_dtype_uint32()) {
      return ScalarType::UInt32;
    } else if (shim_scalartype == aoti_torch_dtype_uint64()) {
      return ScalarType::UInt64;
    } else {
      throw std::runtime_error(
          "Not yet supported ScalarType " + std::to_string(shim_scalartype) +
          ", please file an issue describing your use case.");
    }
  }
};

// Specialization for StableIValue => std::nullopt_t
template <>
struct ToImpl<std::nullopt_t> {
  static std::nullopt_t call(StableIValue val) {
    // val should be equivalent to from(nullptr)
    return std::nullopt;
  }
};

// Specialization for StableIValue => std::optional, see [Handling
// std::optional] as the semantic is the same but in reverse direction as we go
// from IValue --(from_ivalue)-> StableIValue --(to<T>)-> T in custom extension
template <typename T>
struct ToImpl<std::optional<T>> {
  static std::optional<T> call(StableIValue val) {
    auto sivp = to<StableIValue*>(val);

    // sivp is either nullptr or a pointer to a StableIValue
    if (sivp == nullptr) {
      return {};
    }
    auto inner_val = to<T>(*sivp);

    // free the memory associated with StableIValue* sivp
    delete sivp;

    return std::make_optional(inner_val);
  }
};

// Specialization for StableIValue => torch::stable::Tensor
// The resulting stable::Tensor steals ownership of the input's
// underlying AtenTensorHandle.
template <>
struct ToImpl<torch::stable::Tensor> {
  static torch::stable::Tensor call(StableIValue val) {
    return torch::stable::Tensor(to<AtenTensorHandle>(val));
  }
};

} // namespace detail

// Expose the partially templated class functions through single functions
template <typename T>
StableIValue from(T val) {
  return detail::FromImpl<T>::call(val);
}

template <typename T>
StableIValue from(const std::optional<T>& val) {
  return detail::FromImpl<std::optional<T>>::call(val);
}

// The below overload is used! See https://godbolt.org/z/859cshxrW
// We are suppressing the warning for versions clang12- and gcc11-
[[maybe_unused]] StableIValue from(const torch::stable::Tensor& val) {
  return detail::FromImpl<torch::stable::Tensor>::call(val);
}

template <typename T>
T to(StableIValue val) {
  return detail::ToImpl<T>::call(val);
}

// =============================================================================
//  end to helpers for converting between StableIValue and T
// =============================================================================

} // namespace
