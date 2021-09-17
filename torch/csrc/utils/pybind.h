#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_tuples.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/Generator.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;

// This makes intrusive_ptr to be available as a custom pybind11 holder type,
// see
// https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::intrusive_ptr<T>, true);

namespace pybind11 { namespace detail {

// torch.Tensor <-> at::Tensor conversions (without unwrapping)
template <>
struct type_caster<at::Tensor> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Tensor, _("at::Tensor"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPVariable_Check(obj)) {
      value = THPVariable_Unpack(obj);
      return true;
    }
    return false;
  }

  static handle
  cast(const at::Tensor& src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPVariable_Wrap(src));
  }
};

// torch._StorageBase <-> at::Storage
template <>
struct type_caster<at::Storage> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Storage, _("at::Storage"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (torch::isStorage(obj)) {
      value = torch::createStorage(obj);
      return true;
    }
    return false;
  }

  static handle
  cast(const at::Storage& src, return_value_policy /* policy */, handle /* parent */) {
    TORCH_CHECK(
        false,
        "NotImplementedError: pybind conversion of at::Storages from C++ to python not supported.");
    // Storages are untyped, see: https://github.com/pytorch/pytorch/issues/47442
    return handle(torch::createPyObject(src, caffe2::TypeMeta()));
  }
};

template <>
struct type_caster<at::Generator> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Generator, _("at::Generator"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPGenerator_Check(obj)) {
      value = reinterpret_cast<THPGenerator*>(obj)->cdata;
      return true;
    }
    return false;
  }

  static handle
  cast(const at::Generator& src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPGenerator_Wrap(src));
  }
};

template<> struct type_caster<at::IntArrayRef> {
public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::IntArrayRef, _("at::IntArrayRef"));

  bool load(handle src, bool) {
    PyObject *source = src.ptr();
    auto tuple = PyTuple_Check(source);
    if (tuple || PyList_Check(source)) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      const auto size = tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
      v_value.resize(size);
      for(const auto idx : c10::irange(size)) {
        PyObject* obj = tuple ? PyTuple_GET_ITEM(source, idx) : PyList_GET_ITEM(source, idx);
        if (THPVariable_Check(obj)) {
          v_value[idx] = THPVariable_Unpack(obj).item<int64_t>();
        } else if (PyLong_Check(obj)) {
          // use THPUtils_unpackLong after it is safe to include python_numbers.h
          v_value[idx] = THPUtils_unpackLong(obj);
        } else {
          return false;
        }
      }
      value = v_value;
      return true;
    }
    return false;
  }
  static handle cast(at::IntArrayRef src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPUtils_packInt64Array(src.size(), src.data()));
  }
private:
  std::vector<int64_t> v_value;
};

template <>
struct type_caster<at::Device> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Device, _("at::Device"));

  // PYBIND11_TYPE_CASTER defines a member field called value. Since at::Device
  // cannot be default-initialized, we provide this constructor to explicitly
  // initialize that field. The value doesn't matter as it will be overwritten
  // after a successful call to load.
  type_caster() : value(c10::kCPU) {}

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPDevice_Check(obj)) {
      value = reinterpret_cast<THPDevice*>(obj)->device;
      return true;
    }
    return false;
  }

  static handle
  cast(const at::Device& src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPDevice_New(src));
  }
};

// Pybind11 bindings for our optional type.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
template <typename T>
struct type_caster<c10::optional<T>> : optional_caster<c10::optional<T>> {};
}} // namespace pybind11::detail

namespace torch {
namespace impl {

// Use this function if you have a C++ object that is used from both C++
// and Python contexts, and you need its GIL to be released when you
// destruct it in the Python context.
//
// This function is a valid shared_ptr destructor and can be used to
// conveniently allocate a shared_ptr to an object whose destructor will be run
// without the GIL.  Pass it as the second argument to shared_ptr, e.g.,
//
//    shared_ptr<T>(new T(), destroy_without_gil<T>)
//
// Attaching the GIL release logic to the holder pointer rather than the
// actual destructor of T is helpful when T is Python-agnostic and
// shouldn't refer to the PYthon API.
//
// Note there are limitations to the correctness of code that makes use of this.
// In particular, if a shared_ptr is constructed from C++ code without this
// destructor and then passed to pybind11, pybind11 will happily take ownership
// of the shared_ptr (and be willing to destruct it from a context where it is
// holding the GIL).  unique_ptr with a type branded deleter is less prone to
// this problem, because a stock deleter unique_ptr is not convertible with it.
// I plan to mitigate this problem by adding DEBUG-only asserts to the true C++
// destructors that the GIL is not held (using a virtual call to get to the
// Python interpreter); alternately, we could use a virtual call to simply
// ensure we release the GIL in the C++ destructor, however, this is a layering
// violation (why does code that is ostensibly Python agnostic calling into the
// GIL).
//
// Adapted from https://github.com/pybind/pybind11/issues/1446#issuecomment-406341510
template <typename T> inline void destroy_without_gil(T *ptr) {
  // Because the ownership of a shared_ptr is diffuse, it's not possible to
  // necessarily predict whether or not the last reference to an object will
  // be destructed from Python or C++.  This means that in the destructor here,
  // we don't necessarily know if we actually have the GIL or not; in fact,
  // we don't even know if the Python interpreter still exists!  Thus, we have
  // to test for it before releasing the GIL.
  //
  // PyGILState_Check is hopefully self explanatory.  But Py_IsInitialized or
  // _PyIsFinalizing?  Both get set at the same time during the Python
  // destruction process:
  // https://github.com/python/cpython/blob/d92513390a1a0da781bb08c284136f4d7abea36d/Python/pylifecycle.c#L1716-L1717
  // so the operant question is whether or not you want to release the GIL after
  // finalization has completed (and there is just no Python interpreter).
  // Clearly there is no need to release GIL in that state, so we want
  // Py_IsInitialized.
  if (Py_IsInitialized() && PyGILState_Check()) {
    pybind11::gil_scoped_release nogil;
    delete ptr;
  } else {
    delete ptr;
  }
}

} // namespace impl
} // namespace torch
