#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/jit_type_base.h>
#include <c10/util/irange.h>
#include <c10/util/variant.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/utils/tensor_memoryformats.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;

// This makes intrusive_ptr to be available as a custom pybind11 holder type,
// see
// https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::intrusive_ptr<T>, true);

PYBIND11_DECLARE_HOLDER_TYPE(T, c10::SingletonOrSharedTypePtr<T>);
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::SingletonTypePtr<T>, true);

namespace pybind11 {
namespace detail {

// torch.Tensor <-> at::Tensor conversions (without unwrapping)
template <>
struct TORCH_PYTHON_API type_caster<at::Tensor> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Tensor, _("torch.Tensor"));

  bool load(handle src, bool);

  static handle cast(
      const at::Tensor& src,
      return_value_policy /* policy */,
      handle /* parent */);
};

// torch._StorageBase <-> at::Storage
template <>
struct type_caster<at::Storage> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Storage, _("torch.StorageBase"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (torch::isStorage(obj)) {
      value = torch::createStorage(obj);
      return true;
    }
    return false;
  }

  static handle cast(
      const at::Storage& src,
      return_value_policy /* policy */,
      handle /* parent */) {
    // See Note [Python Storages for Tensor Subclasses]
    // Unfortunately, we don't know at this point whether or not our storage
    // came from a traceable python subclass, so we'll (dangerously)
    // return a python storage here.
    return handle(torch::createPyObject(src, /*always_create_storage=*/true));
  }
};

template <>
struct type_caster<at::Generator> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Generator, _("torch.Generator"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPGenerator_Check(obj)) {
      value = reinterpret_cast<THPGenerator*>(obj)->cdata;
      return true;
    }
    return false;
  }

  static handle cast(
      const at::Generator& src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return handle(THPGenerator_Wrap(src));
  }
};

template <>
struct TORCH_PYTHON_API type_caster<at::IntArrayRef> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::IntArrayRef, _("Tuple[int, ...]"));

  bool load(handle src, bool);
  static handle cast(
      at::IntArrayRef src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  std::vector<int64_t> v_value;
};

template <>
struct TORCH_PYTHON_API type_caster<at::SymIntArrayRef> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::SymIntArrayRef, _("List[int]"));

  bool load(handle src, bool);
  static handle cast(
      at::SymIntArrayRef src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  std::vector<c10::SymInt> v_value;
};

template <>
struct TORCH_PYTHON_API type_caster<at::ArrayRef<c10::SymNode>> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::ArrayRef<c10::SymNode>, _("List[SymNode]"));

  bool load(handle src, bool);
  static handle cast(
      at::ArrayRef<c10::SymNode> src,
      return_value_policy /* policy */,
      handle /* parent */);

 private:
  std::vector<c10::SymNode> v_value;
};

template <>
struct type_caster<at::MemoryFormat> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::MemoryFormat, _("torch.memory_format"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPMemoryFormat_Check(obj)) {
      value = reinterpret_cast<THPMemoryFormat*>(obj)->memory_format;
      return true;
    }
    return false;
  }
  static handle cast(
      at::MemoryFormat src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return handle(torch::utils::getTHPMemoryFormat(src));
  }
};

template <>
struct type_caster<at::Device> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::Device, _("torch.device"));

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

  static handle cast(
      const at::Device& src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return handle(THPDevice_New(src));
  }
};

template <>
struct type_caster<c10::Stream> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(c10::Stream, _("torch.Stream"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPStream_Check(obj)) {
      value = c10::Stream::unpack3(
          ((THPStream*)obj)->stream_id,
          ((THPStream*)obj)->device_index,
          static_cast<c10::DeviceType>(((THPStream*)obj)->device_type));
      return true;
    }
    return false;
  }

  static handle cast(
      const c10::Stream& src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return handle(THPStream_Wrap(src));
  }
};

template <>
struct type_caster<c10::DispatchKey>
    : public type_caster_base<c10::DispatchKey> {
  using base = type_caster_base<c10::DispatchKey>;
  c10::DispatchKey tmp;

 public:
  bool load(handle src, bool convert) {
    if (base::load(src, convert)) {
      return true;
    } else if (py::isinstance(
                   src, py::module_::import("builtins").attr("str"))) {
      tmp = c10::parseDispatchKey(py::cast<std::string>(src));
      value = &tmp;
      return true;
    }
    return false;
  }

  static handle cast(
      c10::DispatchKey src,
      return_value_policy policy,
      handle parent) {
    return base::cast(src, policy, parent);
  }
};

template <>
struct TORCH_PYTHON_API type_caster<c10::Scalar> {
 public:
  PYBIND11_TYPE_CASTER(
      c10::Scalar,
      _("Union[Number, torch.SymInt, torch.SymFloat, torch.SymBool]"));
  bool load(py::handle src, bool);

  static py::handle cast(
      const c10::Scalar& si,
      return_value_policy /* policy */,
      handle /* parent */);
};

template <>
struct TORCH_PYTHON_API type_caster<c10::SymInt> {
 public:
  PYBIND11_TYPE_CASTER(c10::SymInt, _("Union[int, torch.SymInt]"));
  bool load(py::handle src, bool);

  static py::handle cast(
      c10::SymInt si,
      return_value_policy /* policy */,
      handle /* parent */);
};

template <>
struct TORCH_PYTHON_API type_caster<c10::SymFloat> {
 public:
  PYBIND11_TYPE_CASTER(c10::SymFloat, _("float"));
  bool load(py::handle src, bool);

  static py::handle cast(
      c10::SymFloat si,
      return_value_policy /* policy */,
      handle /* parent */);
};

template <>
struct TORCH_PYTHON_API type_caster<c10::SymBool> {
 public:
  PYBIND11_TYPE_CASTER(c10::SymBool, _("Union[bool, torch.SymBool]"));
  bool load(py::handle src, bool);

  static py::handle cast(
      c10::SymBool si,
      return_value_policy /* policy */,
      handle /* parent */);
};

template <typename T>
struct type_caster<c10::complex<T>> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(c10::complex<T>, _("complex"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();

    // Refered from `THPUtils_unpackComplexDouble`
    Py_complex py_complex = PyComplex_AsCComplex(obj);
    if (py_complex.real == -1.0 && PyErr_Occurred()) {
      return false;
    }

    // Python's Complex is always double precision.
    value = c10::complex<double>(py_complex.real, py_complex.imag);
    return true;
  }

  static handle cast(
      const c10::complex<T>& complex,
      return_value_policy /* policy */,
      handle /* parent */) {
    // Python only knows double precision complex.
    return handle(PyComplex_FromDoubles(complex.real(), complex.imag()));
  }
};

// Pybind11 bindings for our optional and variant types.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
template <typename T>
struct type_caster<c10::optional<T>> : optional_caster<c10::optional<T>> {};

template <typename... Ts>
struct C10_MPARK_VISIBILITY_HIDDEN type_caster<c10::variant<Ts...>>
    : variant_caster<c10::variant<Ts...>> {};
} // namespace detail
} // namespace pybind11

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
// Adapted from
// https://github.com/pybind/pybind11/issues/1446#issuecomment-406341510
template <typename T>
inline void destroy_without_gil(T* ptr) {
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
