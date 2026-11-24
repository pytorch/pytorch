#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device_struct.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

// Header-only helpers converting between Python objects (passed as raw
// PyObject* / void*) and their torch::stable equivalents. These are
// libtorch-only to link against, but require libtorch_python to be loaded at
// runtime (see the Python interop shims section in torch/csrc/stable/c/shim.h).
// The GIL must be held by the caller.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// Wrap a Python torch.Tensor (PyObject* passed as void*) as a stable Tensor
// that shares its underlying TensorImpl.
inline Tensor tensor_from_pyobject(void* py_obj) {
  AtenTensorHandle ath = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_tensor_from_pyobject(py_obj, &ath));
  return Tensor(ath);
}

// Wrap a stable Tensor as a new-reference Python torch.Tensor. py_type is an
// optional PyTypeObject* (passed as void*) used as the result's exact Python
// type; nullptr means default torch.Tensor.
inline void* tensor_to_pyobject(const Tensor& t, void* py_type = nullptr) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_tensor_to_pyobject(t.get(), py_type, &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_15_0

// Whether py_obj is a Python torch.Tensor (or a subclass). A probe for callers
// that want to type-check before tensor_from_pyobject (which errors on
// non-tensors).
inline bool is_tensor_pyobject(void* py_obj) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_is_tensor_pyobject(py_obj, &ret));
  return ret;
}

// The dtype/device helpers below translate codes through the stable enum
// mappings (torch::stable::detail::from/to), which cover a subset of all
// ScalarTypes / DeviceTypes; a valid torch.dtype / torch.device outside that
// subset errors even though the C shim itself can represent it.

// Read the ScalarType out of a Python torch.dtype (PyObject* passed as void*).
inline torch::headeronly::ScalarType dtype_from_pyobject(void* py_obj) {
  int32_t dtype = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_dtype_from_pyobject(py_obj, &dtype));
  return torch::stable::detail::to<torch::headeronly::ScalarType>(
      torch::stable::detail::from(dtype));
}

// Wrap a ScalarType as a new-reference Python torch.dtype.
inline void* dtype_to_pyobject(torch::headeronly::ScalarType dtype) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_dtype_to_pyobject(
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype)),
      &raw));
  return raw;
}

// Read the Device out of a Python torch.device (PyObject* passed as void*).
inline Device device_from_pyobject(void* py_obj) {
  int32_t device_type = 0;
  int32_t device_index = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_device_from_pyobject(py_obj, &device_type, &device_index));
  DeviceType extension_device_type = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  return Device(extension_device_type, static_cast<DeviceIndex>(device_index));
}

// Wrap a Device as a new-reference Python torch.device.
inline void* device_to_pyobject(const Device& device) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_device_to_pyobject(
      torch::stable::detail::to<int32_t>(
          torch::stable::detail::from(device.type())),
      static_cast<int32_t>(device.index()),
      &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_15_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// Like the dtype/device helpers, the layout and memory format helpers translate
// codes through the stable enum mappings.

// Read the Layout out of a Python torch.layout (PyObject* passed as void*).
inline torch::headeronly::Layout layout_from_pyobject(void* py_obj) {
  int32_t layout = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_layout_from_pyobject(py_obj, &layout));
  return torch::stable::detail::to<torch::headeronly::Layout>(
      torch::stable::detail::from(layout));
}

// Wrap a Layout as a new-reference Python torch.layout.
inline void* layout_to_pyobject(torch::headeronly::Layout layout) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_layout_to_pyobject(
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(layout)),
      &raw));
  return raw;
}

// Read the MemoryFormat out of a Python torch.memory_format (PyObject* passed
// as void*).
inline torch::headeronly::MemoryFormat memory_format_from_pyobject(
    void* py_obj) {
  int32_t memory_format = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_memory_format_from_pyobject(py_obj, &memory_format));
  return torch::stable::detail::to<torch::headeronly::MemoryFormat>(
      torch::stable::detail::from(memory_format));
}

// Wrap a MemoryFormat as a new-reference Python torch.memory_format.
inline void* memory_format_to_pyobject(
    torch::headeronly::MemoryFormat memory_format) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_memory_format_to_pyobject(
      torch::stable::detail::to<int32_t>(
          torch::stable::detail::from(memory_format)),
      &raw));
  return raw;
}

// Read a Python bool (PyObject* passed as void*). Only True and False are
// accepted.
inline bool bool_from_pyobject(void* py_obj) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_bool_from_pyobject(py_obj, &ret));
  return ret;
}

// Wrap a bool as a new-reference Python bool.
inline void* bool_to_pyobject(bool value) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_bool_to_pyobject(value, &raw));
  return raw;
}

// Read a Python int (PyObject* passed as void*) as an int64_t. bool is
// rejected, and so are values outside the int64_t range.
inline int64_t int64_from_pyobject(void* py_obj) {
  int64_t ret = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_int64_from_pyobject(py_obj, &ret));
  return ret;
}

// Wrap an int64_t as a new-reference Python int.
inline void* int64_to_pyobject(int64_t value) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_int64_to_pyobject(value, &raw));
  return raw;
}

// Read a Python float or int (PyObject* passed as void*) as a double.
inline double double_from_pyobject(void* py_obj) {
  double ret = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_double_from_pyobject(py_obj, &ret));
  return ret;
}

// Wrap a double as a new-reference Python float.
inline void* double_to_pyobject(double value) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_double_to_pyobject(value, &raw));
  return raw;
}

// Copy a Python str (as UTF-8) or bytes (as is) (PyObject* passed as void*)
// into an owned std::string. Embedded NUL bytes are preserved.
inline std::string string_from_pyobject(void* py_obj) {
  StringHandle raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_string_from_pyobject(py_obj, &raw));
  return torch::stable::detail::to<std::string>(
      torch::stable::detail::from(raw));
}

// Decode UTF-8 bytes as a new-reference Python str. The bytes are copied, so
// str only needs to stay alive for the call; invalid UTF-8 errors.
inline void* string_to_pyobject(std::string_view str) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_string_to_pyobject(str.data(), str.size(), &raw));
  return raw;
}

// Generic conversions, composing the helpers above. See from_pyobject /
// to_pyobject below.

namespace detail {

// PyObjectConverter<T>::from(void*) reads a T out of a borrowed PyObject* and
// PyObjectConverter<T>::to(const T&) returns a new reference. Only the types
// specialized here are supported; any other T fails to compile.
template <typename T>
struct PyObjectConverter;

template <>
struct PyObjectConverter<Tensor> {
  static Tensor from(void* py_obj) {
    return tensor_from_pyobject(py_obj);
  }
  static void* to(const Tensor& value) {
    return tensor_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<torch::headeronly::ScalarType> {
  static torch::headeronly::ScalarType from(void* py_obj) {
    return dtype_from_pyobject(py_obj);
  }
  static void* to(torch::headeronly::ScalarType value) {
    return dtype_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<Device> {
  static Device from(void* py_obj) {
    return device_from_pyobject(py_obj);
  }
  static void* to(const Device& value) {
    return device_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<torch::headeronly::Layout> {
  static torch::headeronly::Layout from(void* py_obj) {
    return layout_from_pyobject(py_obj);
  }
  static void* to(torch::headeronly::Layout value) {
    return layout_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<torch::headeronly::MemoryFormat> {
  static torch::headeronly::MemoryFormat from(void* py_obj) {
    return memory_format_from_pyobject(py_obj);
  }
  static void* to(torch::headeronly::MemoryFormat value) {
    return memory_format_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<bool> {
  static bool from(void* py_obj) {
    return bool_from_pyobject(py_obj);
  }
  static void* to(bool value) {
    return bool_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<int64_t> {
  static int64_t from(void* py_obj) {
    return int64_from_pyobject(py_obj);
  }
  static void* to(int64_t value) {
    return int64_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<double> {
  static double from(void* py_obj) {
    return double_from_pyobject(py_obj);
  }
  static void* to(double value) {
    return double_to_pyobject(value);
  }
};

template <>
struct PyObjectConverter<std::string> {
  static std::string from(void* py_obj) {
    return string_from_pyobject(py_obj);
  }
  static void* to(const std::string& value) {
    return string_to_pyobject(value);
  }
};

// C++ -> Python only: a view cannot own what it reads from Python.
template <>
struct PyObjectConverter<std::string_view> {
  static void* to(std::string_view value) {
    return string_to_pyobject(value);
  }
};

template <typename T>
struct is_std_optional : std::false_type {};
template <typename T>
struct is_std_optional<std::optional<T>> : std::true_type {};

template <typename T>
struct PyObjectConverter<std::optional<T>> {
  static_assert(
      !is_std_optional<T>::value,
      "Nested std::optional is not supported: Python None cannot tell an "
      "empty outer optional from an empty inner one");

  static std::optional<T> from(void* py_obj) {
    bool is_none = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_is_none_pyobject(py_obj, &is_none));
    if (is_none) {
      return std::nullopt;
    }
    return PyObjectConverter<T>::from(py_obj);
  }

  static void* to(const std::optional<T>& value) {
    if (value.has_value()) {
      return PyObjectConverter<T>::to(*value);
    }
    void* raw = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_new_none_pyobject(&raw));
    return raw;
  }
};

} // namespace detail

// Convert a Python object (borrowed PyObject* passed as void*) to T. T is any
// type with a *_from_pyobject helper above, or std::optional<T>, which maps
// None to std::nullopt and otherwise converts the object as T. A null py_obj
// is an error, not None.
template <typename T>
T from_pyobject(void* py_obj) {
  return detail::PyObjectConverter<T>::from(py_obj);
}

// Convert value to a new-reference Python object. T is any type with a
// *_to_pyobject helper above (std::string_view included), or std::optional<T>,
// which maps std::nullopt to None.
template <typename T>
void* to_pyobject(const T& value) {
  return detail::PyObjectConverter<T>::to(value);
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
