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

#include <string>
#include <string_view>

// Header-only helpers converting between Python objects (passed as raw
// PyObject* / void*) and their torch::stable equivalents. These are
// libtorch-only to link against, but require libtorch_python to be loaded at
// runtime (see the Python interop shims section in torch/csrc/stable/c/shim.h).
// The GIL must be held by the caller.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

/**
 * Borrow a non-null Python torch.Tensor or actual subclass; overridden
 * `__class__` attributes do not affect the type check. The returned stable
 * Tensor owns a native handle sharing TensorImpl, without copying tensor data.
 *
 * @param py_obj Borrowed PyObject* passed as void*, kept alive for the call.
 * @return An owning stable Tensor sharing the input tensor state.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.14 or newer.
 */
inline Tensor tensor_from_pyobject(void* py_obj) {
  AtenTensorHandle ath = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_tensor_from_pyobject(py_obj, &ath));
  return Tensor(ath);
}

/**
 * Return an owning Python reference sharing the native tensor state. An
 * existing Python wrapper may be reused; a new reference does not imply a new
 * allocation. A requested type must be a Tensor subclass, and a cached wrapper
 * must be compatible with it. The default route uses the runtime default tensor
 * wrapper type.
 *
 * @param t Borrowed stable Tensor, kept alive for the call.
 * @param py_type Optional borrowed PyTypeObject* passed as void*; nullptr
 * selects the default route.
 * @return A new owning PyObject* reference passed as void*. An undefined native
 * tensor maps to Python None. Release or transfer the reference through Python
 * APIs.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.14 or newer.
 */
inline void* tensor_to_pyobject(const Tensor& t, void* py_type = nullptr) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_tensor_to_pyobject(t.get(), py_type, &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_15_0

/**
 * Check the actual Python type against torch.Tensor or its subclasses;
 * overridden `__class__` attributes are ignored. Non-tensors return false, but
 * a null pointer or missing Python backend is an error, not false.
 *
 * @param py_obj Borrowed, non-null PyObject* passed as void*, kept alive for
 * the call.
 * @return Whether the actual object type is a Tensor or subclass.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.15 or newer.
 */
inline bool is_tensor_pyobject(void* py_obj) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_is_tensor_pyobject(py_obj, &ret));
  return ret;
}

// The dtype/device helpers below translate codes through the stable enum
// mappings (torch::stable::detail::from/to), which cover a subset of all
// ScalarTypes / DeviceTypes; a valid torch.dtype / torch.device outside that
// subset errors even though the C shim itself can represent it.

/**
 * Require an actual torch.dtype. The stable runtime enum mapping covers only
 * supported ScalarTypes, so a valid Python dtype outside that subset fails. The
 * by-value result retains no Python storage.
 *
 * @param py_obj Borrowed, non-null Python dtype object passed as void*.
 * @return A ScalarType value translated through the runtime stable mapping.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.15 or newer.
 */
inline torch::headeronly::ScalarType dtype_from_pyobject(void* py_obj) {
  int32_t dtype = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_dtype_from_pyobject(py_obj, &dtype));
  return torch::stable::detail::to<torch::headeronly::ScalarType>(
      torch::stable::detail::from(dtype));
}

/**
 * Translate a supported ScalarType through the runtime stable enum mapping and
 * acquire a reference to the registered Python dtype singleton; no extension
 * enum encoding is frozen by this conversion.
 *
 * @param dtype ScalarType supported by the stable mapping.
 * @return A new owning reference to the dtype singleton, passed as void*.
 * Release or transfer it through Python APIs.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.15 or newer.
 */
inline void* dtype_to_pyobject(torch::headeronly::ScalarType dtype) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_dtype_to_pyobject(
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype)),
      &raw));
  return raw;
}

/**
 * Require an actual torch.device. DeviceType uses the supported runtime stable
 * mapping; valid devices outside its subset fail. The device index is
 * preserved, including -1 for an unspecified index; no Python input storage is
 * retained.
 *
 * @param py_obj Borrowed, non-null Python device object passed as void*.
 * @return An independent stable Device value with the mapped type and index.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.15 or newer.
 */
inline Device device_from_pyobject(void* py_obj) {
  int32_t device_type = 0;
  int32_t device_index = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_device_from_pyobject(py_obj, &device_type, &device_index));
  DeviceType extension_device_type = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  return Device(extension_device_type, static_cast<DeviceIndex>(device_index));
}

/**
 * Translate the DeviceType through the supported runtime stable mapping and
 * construct a Python device with the same index. The result does not borrow the
 * C++ Device or its storage.
 *
 * @param device Borrowed stable Device, kept alive for the call.
 * @return A new owning Python device reference passed as void*. Release or
 * transfer it through Python APIs.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.15 or newer.
 */
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

/**
 * Require an actual torch.layout and translate its runtime code through the
 * stable Layout mapping. All currently exposed torch.layout values are
 * supported; the result retains no Python storage.
 *
 * @param py_obj Borrowed, non-null Python layout object passed as void*.
 * @return A by-value Layout selected through the runtime mapping.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline torch::headeronly::Layout layout_from_pyobject(void* py_obj) {
  int32_t layout = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_layout_from_pyobject(py_obj, &layout));
  return torch::stable::detail::to<torch::headeronly::Layout>(
      torch::stable::detail::from(layout));
}

/**
 * Translate Layout through the runtime stable mapping and acquire a reference
 * to the registered Python layout singleton; it does not allocate a duplicate
 * singleton.
 *
 * @param layout Layout value supported by the stable mapping.
 * @return A new owning layout reference passed as void*. Release or transfer it
 * through Python APIs.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline void* layout_to_pyobject(torch::headeronly::Layout layout) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_layout_to_pyobject(
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(layout)),
      &raw));
  return raw;
}

/**
 * Require an actual torch.memory_format and translate its runtime code through
 * the stable mapping. The supported formats are contiguous, preserve,
 * channels-last and channels-last-3d; the result retains no Python storage.
 *
 * @param py_obj Borrowed, non-null Python memory-format object passed as void*.
 * @return An independent by-value MemoryFormat selected through the runtime
 * mapping.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline torch::headeronly::MemoryFormat memory_format_from_pyobject(
    void* py_obj) {
  int32_t memory_format = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_memory_format_from_pyobject(py_obj, &memory_format));
  return torch::stable::detail::to<torch::headeronly::MemoryFormat>(
      torch::stable::detail::from(memory_format));
}

/**
 * Translate MemoryFormat through the runtime stable mapping and acquire a
 * reference to the registered singleton; no C++ input storage is retained.
 *
 * @param memory_format Contiguous, preserve, channels-last or channels-last-3d
 * format.
 * @return A new owning memory-format reference passed as void*. Release or
 * transfer it through Python APIs.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline void* memory_format_to_pyobject(
    torch::headeronly::MemoryFormat memory_format) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_memory_format_to_pyobject(
      torch::stable::detail::to<int32_t>(
          torch::stable::detail::from(memory_format)),
      &raw));
  return raw;
}

/**
 * Read a Python bool (PyObject* passed as void*). Only True and False are
 * accepted; NumPy booleans and general truthiness conversions are rejected.
 *
 * @param py_obj Borrowed, non-null PyObject* passed as void*, kept alive for
 * the call.
 * @return The bool value, with no retained Python reference.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline bool bool_from_pyobject(void* py_obj) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_bool_from_pyobject(py_obj, &ret));
  return ret;
}

/**
 * Return a reference to the Python bool singleton for value; no new bool
 * allocation is implied.
 *
 * @param value Bool value to convert.
 * @return A new owning Python bool reference passed as void*. Release or
 * transfer it through Python APIs.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline void* bool_to_pyobject(bool value) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_bool_to_pyobject(value, &raw));
  return raw;
}

/**
 * Read a Python int or subclass (PyObject* passed as void*) as an int64_t.
 * NumPy integer scalars are also accepted when runtime NumPy support is
 * available and unpacking succeeds. bool and values outside the int64_t range
 * are rejected; an unrelated `__index__` method alone does not pass the type
 * gate.
 *
 * @param py_obj Borrowed, non-null PyObject* passed as void*, kept alive for
 * the call.
 * @return The checked int64_t value, with no retained Python reference.
 * @note Conversion failures throw. CPython failures additionally remain
 * pending, and their diagnostic is available to a handling C++ caller.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline int64_t int64_from_pyobject(void* py_obj) {
  int64_t ret = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_int64_from_pyobject(py_obj, &ret));
  return ret;
}

/**
 * Convert the full int64_t value into a Python integer; the result does not
 * borrow native input storage.
 *
 * @param value Integer value to convert.
 * @return A new owning Python int reference passed as void*. Release or
 * transfer it through Python APIs.
 * @note Allocation/conversion failures throw; a CPython error remains pending.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline void* int64_to_pyobject(int64_t value) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_int64_to_pyobject(value, &raw));
  return raw;
}

/**
 * Read a Python float, int or subclass (PyObject* passed as void*) as a double,
 * including bool (True becomes 1.0). With runtime NumPy support, NumPy boolean,
 * integer, floating and complex scalars are accepted if float conversion
 * succeeds. NumPy complex conversion discards the imaginary part and can emit
 * ComplexWarning; promoting that warning to an exception makes conversion fail.
 * This uses PythonArgParser's builtin/NumPy scalar check-and-unpack policy, not
 * its Tensor or symbolic-number conveniences. Builtin complex and unrelated
 * objects with only `__float__` or `__index__` methods are rejected.
 *
 * @param py_obj Borrowed, non-null PyObject* passed as void*, kept alive for
 * the call.
 * @return A double value, preserving signed zero, infinities and NaN without
 * retaining the input.
 * @note Conversion failures throw. CPython failures additionally remain
 * pending, and their diagnostic is available to a handling C++ caller.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline double double_from_pyobject(void* py_obj) {
  double ret = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_double_from_pyobject(py_obj, &ret));
  return ret;
}

/**
 * Convert the double value into a Python float; the result does not borrow
 * native input storage.
 *
 * @param value Double value to convert.
 * @return A new owning Python float reference passed as void*. Release or
 * transfer it through Python APIs.
 * @note Allocation/conversion failures throw; a CPython error remains pending.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline void* double_to_pyobject(double value) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_double_to_pyobject(value, &raw));
  return raw;
}

/**
 * Copy a Python string or bytes object into an owned std::string.
 *
 * Python str and bytes objects, including subclasses, are accepted without
 * coercing other objects with str(). A str is encoded as strict UTF-8; bytes
 * are copied as-is, including invalid UTF-8. Explicit lengths preserve embedded
 * NUL bytes. A str containing a lone surrogate, such as the Python literal
 * @code{.py}
 * "\ud800"
 * @endcode
 * cannot be encoded as UTF-8 and fails with UnicodeEncodeError pending.
 *
 * The runtime copies the bytes into an owning StringHandle. This wrapper copies
 * the handle's bytes into std::string and releases the handle on success or
 * exceptional exit; neither the result nor the handle borrows Python storage.
 * Conversion failures throw a C++ exception; CPython errors remain pending.
 *
 * @param py_obj Borrowed, non-null PyObject* passed as void*. The caller keeps
 * it alive for the call.
 * @return An owning std::string, independent of the input object's lifetime.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline std::string string_from_pyobject(void* py_obj) {
  StringHandle raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_string_from_pyobject(py_obj, &raw));
  return torch::stable::detail::to<std::string>(
      torch::stable::detail::from(raw));
}

/**
 * Decode a borrowed UTF-8 byte range into an owned Python str.
 *
 * The view's pointer and length are borrowed only for this call. CPython copies
 * and decodes the bytes into Python-owned Unicode storage; the result does not
 * retain the view or its buffer. Embedded NUL bytes are preserved. Invalid
 * UTF-8 fails with UnicodeDecodeError pending, and conversion failures throw a
 * C++ exception. No temporary std::string is needed for this output conversion.
 *
 * @param str Borrowed UTF-8 bytes. The caller keeps the buffer alive for the
 * call. A null data pointer is allowed only when the size is zero, as with a
 * default-constructed empty std::string_view; nonempty views must be non-null.
 * @return A new owning reference to Python str, passed as void*. The caller
 * releases or transfers it through the appropriate Python reference API.
 * @pre The caller holds the GIL and libtorch_python is loaded at runtime.
 * @note Requires a TORCH_TARGET_VERSION and runtime of PyTorch 2.16 or newer.
 */
inline void* string_to_pyobject(std::string_view str) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_string_to_pyobject(str.data(), str.size(), &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
