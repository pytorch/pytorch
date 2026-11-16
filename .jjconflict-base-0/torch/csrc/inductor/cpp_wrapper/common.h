#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>

#include <Python.h>
#if __has_include(<pybind11/gil_simple.h>)
#include <pybind11/gil_simple.h>
#else
// pybind11 < 3.0: gil_simple.h does not exist yet.
#define PYBIND11_SIMPLE_GIL_MANAGEMENT
#include <pybind11/gil.h>
// Provide the _simple aliases so generated code works with either version.
namespace pybind11 {
using gil_scoped_acquire_simple = gil_scoped_acquire;
using gil_scoped_release_simple = gil_scoped_release;
} // namespace pybind11
#endif

// Required for custom op dispatch via the stable ABI
#include <torch/csrc/stable/library.h>

#ifdef TORCH_INDUCTOR_PRECOMPILE_HEADERS
// Include some often-used cpp_wrapper headers, for precompiling.
#include <c10/util/BFloat16.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#endif

namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

class RAIIPyObject {
 public:
  RAIIPyObject() = default;
  // steals a reference to a PyObject
  RAIIPyObject(PyObject* obj) : obj_{obj} {}
  RAIIPyObject(const RAIIPyObject& other) : obj_{other.obj_} {
    Py_XINCREF(obj_);
  }
  RAIIPyObject(RAIIPyObject&& other) noexcept {
    // refcount doesn't change, and obj_ is currently nullptr
    std::swap(obj_, other.obj_);
  }
  ~RAIIPyObject() {
    Py_XDECREF(obj_);
  }
  RAIIPyObject& operator=(const RAIIPyObject& other) {
    if (this != &other) {
      Py_XDECREF(obj_);
      obj_ = other.obj_;
      Py_XINCREF(obj_);
    }
    return *this;
  }
  RAIIPyObject& operator=(RAIIPyObject&& other) noexcept {
    // refcount to the current object decreases, but refcount to other.obj_ is
    // the same
    Py_XDECREF(obj_);
    obj_ = std::exchange(other.obj_, nullptr);
    return *this;
  }
  operator bool() const noexcept {
    return obj_;
  }
  operator PyObject*() {
    return obj_;
  }
  PyObject* get() {
    return obj_;
  }

 private:
  PyObject* obj_{nullptr};
};

#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
using namespace torch::aot_inductor;

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

// Helpers backed by function pointers exported from the torch._C._dynamo.guards
// Python module. Defined at the end of this header so they can use RAIIPyObject
// and AOTI_TORCH_CHECK, both established above.
//
// Inductor-generated wrappers cannot always link the libtorch_python THP
// symbols directly (notably in fbcode, where libtorch_python symbols are absent
// from the executable's dynamic symbol table; see torch_c_dynamo_guards_init in
// torch/csrc/dynamo/guards.cpp). Instead we resolve the corresponding
// _torchinductor_* function pointers from the guards module at runtime, lazily
// on first use, and expose typed torch_python_* wrappers around them.

// Forward-declared rather than including the heavy <ATen/Tensor.h>:
// at::TensorBase is used below only in reference parameters and
// function-pointer types.
namespace at {
class TensorBase;
}

namespace torch::aot_inductor {

static PyObject* (*_torchinductor_thp_device_new)(int, int) = nullptr;
static PyObject* (*_torchinductor_get_thp_dtype)(int) = nullptr;
static PyObject* (*_torchinductor_get_thp_layout)(int) = nullptr;
static PyObject* (*_torchinductor_get_thp_memory_format)(int) = nullptr;
static PyObject* (*_torchinductor_thp_variable_wrap)(const at::TensorBase&) =
    nullptr;
static int32_t (*_torchinductor_thputils_unpack_int)(PyObject*) = nullptr;

// Guards the lazy init below; set true only once all pointers resolve.
static bool _torchinductor_pointers_initialized = false;

static inline void* torch_python_get_pointer_attr(
    PyObject* module,
    const char* attr) {
  RAIIPyObject value(PyObject_GetAttrString(module, attr));
  AOTI_TORCH_CHECK(
      value, "Failed to load torch._C._dynamo.guards function pointer");
  void* ptr = PyLong_AsVoidPtr(value.get());
  AOTI_TORCH_CHECK(
      ptr && !PyErr_Occurred(),
      "Failed to parse torch._C._dynamo.guards function pointer");
  return ptr;
}

// Resolves every _torchinductor_* pointer in one shot; the guards module
// registers them all together. _torchinductor_pointers_initialized is set only
// after all resolutions succeed, so a partial failure leaves it false and the
// next call retries instead of leaving some pointers null behind a satisfied
// early-return.
static inline void torch_python_init_guards_pointers() {
  if (_torchinductor_pointers_initialized) {
    return;
  }

  RAIIPyObject guards_mod(PyImport_ImportModule("torch._C._dynamo.guards"));
  AOTI_TORCH_CHECK(guards_mod, "Failed to import torch._C._dynamo.guards");

  _torchinductor_thp_device_new =
      reinterpret_cast<decltype(_torchinductor_thp_device_new)>(
          torch_python_get_pointer_attr(
              guards_mod.get(), "_torchinductor_thp_device_new"));
  _torchinductor_get_thp_dtype =
      reinterpret_cast<decltype(_torchinductor_get_thp_dtype)>(
          torch_python_get_pointer_attr(
              guards_mod.get(), "_torchinductor_get_thp_dtype"));
  _torchinductor_get_thp_layout =
      reinterpret_cast<decltype(_torchinductor_get_thp_layout)>(
          torch_python_get_pointer_attr(
              guards_mod.get(), "_torchinductor_get_thp_layout"));
  _torchinductor_get_thp_memory_format =
      reinterpret_cast<decltype(_torchinductor_get_thp_memory_format)>(
          torch_python_get_pointer_attr(
              guards_mod.get(), "_torchinductor_get_thp_memory_format"));
  _torchinductor_thp_variable_wrap =
      reinterpret_cast<decltype(_torchinductor_thp_variable_wrap)>(
          torch_python_get_pointer_attr(
              guards_mod.get(), "_torchinductor_thp_variable_wrap"));
  _torchinductor_thputils_unpack_int =
      reinterpret_cast<decltype(_torchinductor_thputils_unpack_int)>(
          torch_python_get_pointer_attr(
              guards_mod.get(), "_torchinductor_thputils_unpack_int"));

  _torchinductor_pointers_initialized = true;
}

static inline PyObject* torch_python_thp_device_new(
    int device_type,
    int device_index) {
  torch_python_init_guards_pointers();
  return _torchinductor_thp_device_new(device_type, device_index);
}

static inline PyObject* torch_python_get_thp_dtype(int dtype) {
  torch_python_init_guards_pointers();
  return _torchinductor_get_thp_dtype(dtype);
}

static inline PyObject* torch_python_get_thp_layout(int layout) {
  torch_python_init_guards_pointers();
  return _torchinductor_get_thp_layout(layout);
}

static inline PyObject* torch_python_get_thp_memory_format(int memory_format) {
  torch_python_init_guards_pointers();
  return _torchinductor_get_thp_memory_format(memory_format);
}

static inline PyObject* torch_python_thp_variable_wrap(
    const at::TensorBase& tensor) {
  torch_python_init_guards_pointers();
  return _torchinductor_thp_variable_wrap(tensor);
}

static inline int32_t torch_python_thputils_unpack_int(PyObject* obj) {
  torch_python_init_guards_pointers();
  return _torchinductor_thputils_unpack_int(obj);
}

} // namespace torch::aot_inductor
