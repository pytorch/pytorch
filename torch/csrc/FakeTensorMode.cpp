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
// A C++ FakeTensorMode is exposed to Python as one object, a CppFakeTensorMode,
// which subclasses the mode's binding (torch._C._CppFakeTensorMode). It is the
// fake mode that the Python parts of fake dispatch (op implementations, the
// fake tensor converter) see. TODO: it does not yet have all of the
// FakeTensorMode interface they use (e.g. is_our_fake, allow_fallback_kernels,
// in_kernel_invocation, which is thread-local for a C++ mode); it gains it as
// C++ fake dispatch comes to call them. Ownership:
//
//   fake tensor       --shared_ptr--> C++ mode
//   CppFakeTensorMode --shared_ptr (pybind11 holder)--> C++ mode
//   C++ mode          --fake_mode_pyobj_ (weakref)--> CppFakeTensorMode
//
// The C++ mode only weakly references its Python object; a strong reference
// would form a cycle through the holder. So fake tensors can outlive the
// object: once Python drops it, it is collected while the C++ mode lives on
// through its fakes.
//
// getCppFakeTensorModePyObj returns the object while it is alive. Otherwise it
// builds a new CppFakeTensorMode around the same C++ mode with the classmethod
// CppFakeTensorMode._from_cpp_mode, passing the mode in a capsule that
// _CppFakeTensorMode's constructor adopts, and checks the object holds the
// mode. It caches a weakref to that object, so lookups return the same object
// only while it remains alive. The returned py::object keeps it alive while a
// callback runs; afterwards it can be collected and rebuilt again.
//
// A rebuilt object is equivalent for fake dispatch because everything dispatch
// reads is owned by the C++ mode: the ShapeEnv, the fake tensor converter, the
// epoch and the mode's flags. It is still a new object, so Python identity is
// not preserved, and it is always a plain CppFakeTensorMode, so subclasses are
// not supported (the class is marked final, which only type checkers enforce).
//
// Under free-threaded Python, threads can look up the same mode's object at
// once, so fake_mode_pyobj_ is guarded by the mode's fake_mode_pyobj_mutex_.
// The lock is never held while running Python, so two threads may both rebuild
// the object; one already cached by another thread is kept while it is alive.
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
