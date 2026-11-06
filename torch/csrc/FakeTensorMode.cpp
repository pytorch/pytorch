#include <torch/csrc/FakeTensorMode.h>
#include <torch/csrc/PyInterpreter.h>

#include <mutex>

namespace torch::fake_tensor {

// Guards fake_mode_pyobj_ of every C++ mode once the mode is shared: under
// free-threaded Python, threads can look up the same mode's wrapper at once. It
// is never held while running Python, so two threads may both rebuild a
// wrapper; one already cached by another thread is kept while it is alive.
static std::mutex& fake_mode_pyobj_mutex() {
  static std::mutex mutex;
  return mutex;
}

// The referent of a weakref, or a null py::object if it was collected.
static py::object weakref_referent(const c10::SafePyObject& weakref) {
  PyObject* obj = nullptr;
  if (PyWeakref_GetRef(weakref.ptr(getPyInterpreter()), &obj) < 0) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(obj);
}

// Note [C++ FakeTensorMode Python wrapper lifetime]
// A C++ FakeTensorMode is exposed to Python through a CppFakeTensorMode
// wrapper. Ownership:
//
//   fake tensor    --shared_ptr--> C++ mode
//   Python wrapper --_cpp_mode--> _CppFakeTensorMode --shared_ptr--> C++ mode
//   C++ mode       --fake_mode_pyobj_ (weakref)--> Python wrapper
//
// The wrapper must hold the bound _CppFakeTensorMode strongly in its _cpp_mode
// attribute. The wrapper is the fake mode that the Python parts of fake
// dispatch (op implementations, the fake tensor converter) run against, so it
// must provide the FakeTensorMode interface they use. The C++ mode only weakly
// references its wrapper; a strong reference would form a cycle through the
// bound _CppFakeTensorMode object. So fake tensors can outlive the wrapper:
// once Python drops it, it is collected while the C++ mode lives on through its
// fakes.
//
// getCppFakeTensorModePyObj returns the wrapper while it is alive. Otherwise it
// builds a new CppFakeTensorMode around the same C++ mode, with the classmethod
// CppFakeTensorMode._from_cpp_mode(cpp_mode), which must wrap the given mode
// rather than build one through _CppFakeTensorMode's constructor. It caches a
// weakref to that wrapper, so lookups return the same object only while that
// wrapper remains alive. The returned py::object keeps the wrapper alive while
// a callback runs; afterwards it can be collected and rebuilt again.
//
// A rebuilt wrapper is equivalent for fake dispatch because everything
// dispatch reads is owned by the C++ mode: the ShapeEnv, the fake tensor
// converter, the epoch and the mode's flags. Retracing code bumps
// fake_mode.epoch on the wrapper, so the wrapper must expose epoch as a
// property forwarding to _cpp_mode.epoch (FakeTensorMode.epoch is a class
// attribute, so assigning it on an instance would not reach the C++ mode). It
// is still a new object, so Python identity is not preserved, and it is always
// a plain CppFakeTensorMode, so subclasses are not supported (the class is
// marked final, which only type checkers enforce).
py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (mode == nullptr) {
    return py::none();
  }
  std::shared_ptr<c10::SafePyObject> cached;
  {
    std::lock_guard<std::mutex> lock(fake_mode_pyobj_mutex());
    cached = mode->fake_mode_pyobj_;
  }
  if (cached != nullptr) {
    if (py::object wrapper = weakref_referent(*cached)) {
      return wrapper;
    }
  }
  // The wrapper is gone or was never made; see
  // Note [C++ FakeTensorMode Python wrapper lifetime].
  py::object wrapper = py::module::import("torch._subclasses.fake_tensor")
                           .attr("CppFakeTensorMode")
                           .attr("_from_cpp_mode")(py::cast(mode));
  // SafePyObject owns the weakref with interpreter-safe lifetime management,
  // and the shared_ptr owns the SafePyObject.
  PyObject* weakref = PyWeakref_NewRef(wrapper.ptr(), nullptr);
  if (weakref == nullptr) {
    throw py::error_already_set();
  }
  auto fresh = std::make_shared<c10::SafePyObject>(weakref, getPyInterpreter());
  std::lock_guard<std::mutex> lock(fake_mode_pyobj_mutex());
  // Another thread may have cached a wrapper while this one ran Python
  if (mode->fake_mode_pyobj_ != cached) {
    if (py::object installed = weakref_referent(*mode->fake_mode_pyobj_)) {
      return installed;
    }
  }
  mode->fake_mode_pyobj_.swap(fresh);
  return wrapper;
}

} // namespace torch::fake_tensor
