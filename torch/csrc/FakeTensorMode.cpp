#include <torch/csrc/FakeTensorMode.h>
#include <torch/csrc/PyInterpreter.h>

#include <mutex>
#include <string_view>

namespace torch::fake_tensor {

static constexpr const char* kCppFakeModeCapsuleName =
    "torch._C._CppFakeTensorMode";

static py::capsule make_cpp_fake_mode_capsule(
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  return py::capsule(
      new std::shared_ptr<c10::FakeTensorMode>(mode),
      kCppFakeModeCapsuleName,
      [](PyObject* capsule) {
        delete static_cast<std::shared_ptr<c10::FakeTensorMode>*>(
            PyCapsule_GetPointer(capsule, kCppFakeModeCapsuleName));
      });
}

std::shared_ptr<c10::FakeTensorMode> cpp_fake_mode_from_capsule(
    const py::capsule& capsule) {
  TORCH_CHECK_TYPE(
      capsule.name() != nullptr &&
          std::string_view(capsule.name()) == kCppFakeModeCapsuleName,
      "expected a C++ FakeTensorMode capsule");
  return *capsule.get_pointer<std::shared_ptr<c10::FakeTensorMode>>();
}

// Whether py_mode is an initialized _CppFakeTensorMode holding mode
static bool holds_mode(
    const py::object& py_mode,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (!py::isinstance<c10::FakeTensorMode>(py_mode)) {
    return false;
  }
  try {
    return py::cast<std::shared_ptr<c10::FakeTensorMode>>(py_mode) == mode;
  } catch (const py::cast_error&) {
    // An instance whose __init__ has not run
    return false;
  }
}

// The referent of a weakref, or a null py::object if it was collected.
static py::object weakref_referent(const c10::SafePyObject& weakref) {
  PyObject* obj = nullptr;
  if (PyWeakref_GetRef(weakref.ptr(getPyInterpreter()), &obj) < 0) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(obj);
}

// Note [C++ FakeTensorMode Python object lifetime]
// Fakes, TLS and saved ThreadLocalState own the C++ mode through shared_ptr.
// CppFakeTensorMode's pybind11 holder also owns it; the reverse reference is
// weak, so Python can collect the wrapper while C++ still uses the mode.
// Every C++-to-Python lookup reuses the live wrapper or recreates it around
// the same mode. All persistent state must therefore live on the C++ mode.
// A returned py::object keeps the wrapper alive for the duration of a callback.
//
// ShapeEnv and the converter must not strongly hold the mode's fakes: Python
// GC cannot see the references through the C++ mode. Only the exact slotted
// CppFakeTensorMode is supported, so wrappers cannot introduce extra state.
//
// The per-mode mutex protects the weak cache on free-threaded Python. Python
// reconstruction runs outside the lock; racing lookups return the first live
// wrapper installed, releasing any unused wrapper after unlocking.
static py::handle get_cpp_fake_tensor_mode_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch._subclasses.fake_tensor")
            .attr("CppFakeTensorMode");
      })
      .get_stored();
#else
  static py::handle cls =
      py::object(py::module::import("torch._subclasses.fake_tensor")
                     .attr("CppFakeTensorMode"))
          .release();
  return cls;
#endif
}

std::shared_ptr<c10::FakeTensorMode> getCppFakeTensorMode(py::handle py_mode) {
  TORCH_CHECK_TYPE(
      Py_TYPE(py_mode.ptr()) ==
          reinterpret_cast<PyTypeObject*>(
              get_cpp_fake_tensor_mode_class().ptr()),
      "Expected a CppFakeTensorMode (not a subclass, nor a bare ",
      "_CppFakeTensorMode), got ",
      Py_TYPE(py_mode.ptr())->tp_name);
  return py::cast<std::shared_ptr<c10::FakeTensorMode>>(py_mode);
}

py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (mode == nullptr) {
    return py::none();
  }
  std::shared_ptr<c10::SafePyObject> cached;
  {
    std::lock_guard<std::mutex> lock(mode->fake_mode_pyobj_mutex_);
    cached = mode->fake_mode_pyobj_;
  }
  if (cached != nullptr) {
    if (py::object py_mode = weakref_referent(*cached)) {
      return py_mode;
    }
  }
  // The object is gone; see Note [C++ FakeTensorMode Python object lifetime].
  py::object py_mode =
      py::module::import("torch._subclasses.fake_tensor")
          .attr("CppFakeTensorMode")
          .attr("_from_cpp_mode")(make_cpp_fake_mode_capsule(mode));
  TORCH_CHECK(
      Py_TYPE(py_mode.ptr()) ==
              reinterpret_cast<PyTypeObject*>(
                  get_cpp_fake_tensor_mode_class().ptr()) &&
          holds_mode(py_mode, mode),
      "CppFakeTensorMode._from_cpp_mode returned an object that does not hold "
      "the C++ FakeTensorMode it was given");
  // SafePyObject owns the weakref with interpreter-safe lifetime management,
  // and the shared_ptr owns the SafePyObject.
  PyObject* weakref = PyWeakref_NewRef(py_mode.ptr(), nullptr);
  if (weakref == nullptr) {
    throw py::error_already_set();
  }
  auto fresh = std::make_shared<c10::SafePyObject>(weakref, getPyInterpreter());
  std::lock_guard<std::mutex> lock(mode->fake_mode_pyobj_mutex_);
  // Another thread may have cached an object while this one ran Python
  if (mode->fake_mode_pyobj_ != cached) {
    if (py::object installed = weakref_referent(*mode->fake_mode_pyobj_)) {
      return installed;
    }
  }
  mode->fake_mode_pyobj_.swap(fresh);
  return py_mode;
}

} // namespace torch::fake_tensor
