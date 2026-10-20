#include <torch/csrc/FakeTensorMode.h>
#include <torch/csrc/PyInterpreter.h>

namespace torch::fake_tensor {

// Note [C++ FakeTensorMode Python wrapper lifetime]
// A C++ FakeTensorMode is exposed to Python through a CppFakeTensorMode
// wrapper. Ownership:
//
//   fake tensor        --shared_ptr-------------> C++ FakeTensorMode
//   Python wrapper     --shared_ptr (capsule)---> C++ FakeTensorMode
//   C++ FakeTensorMode --fake_mode_pyobj_ (weakref)--> Python wrapper
//
// The C++ mode only weakly references its wrapper; a strong reference would
// form a cycle through the capsule. So fake tensors can outlive the wrapper:
// once Python drops it, it is collected while the C++ mode lives on through
// its fakes.
//
// getCppFakeTensorModePyObj returns the wrapper while it is alive. Otherwise it
// builds a new CppFakeTensorMode around the same C++ mode and caches a weakref
// to that one, so lookups return the same object only while that wrapper
// remains alive. The returned py::object keeps the wrapper alive while a
// callback runs; afterwards it can be collected and rebuilt again.
//
// A rebuilt wrapper is equivalent for fake dispatch because everything
// dispatch reads is owned by the C++ mode: the ShapeEnv, the fake tensor
// converter and the mode's flags. It is still a new object: Python identity,
// attributes set on the old wrapper instance and its creation stack trace are
// not preserved, and it is always a plain CppFakeTensorMode, so subclasses are
// not supported.
py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (mode == nullptr) {
    return py::none();
  }
  if (mode->fake_mode_pyobj_ != nullptr) {
    PyObject* obj = nullptr;
    int alive =
        PyWeakref_GetRef(mode->fake_mode_pyobj_->ptr(getPyInterpreter()), &obj);
    if (alive < 0) {
      throw py::error_already_set();
    }
    if (alive > 0) {
      return py::reinterpret_steal<py::object>(obj);
    }
  }
  // The wrapper is gone or was never made; see
  // Note [C++ FakeTensorMode Python wrapper lifetime].
  auto converter = mode->fake_tensor_converter_
      ? py::reinterpret_borrow<py::object>(
            mode->fake_tensor_converter_->ptr(getPyInterpreter()))
      : py::none();
  auto shape_env = mode->shape_env_
      ? py::reinterpret_borrow<py::object>(
            mode->shape_env_->ptr(getPyInterpreter()))
      : py::none();
  py::object cls = py::module::import("torch._subclasses.fake_tensor")
                       .attr("CppFakeTensorMode");
  py::object wrapper =
      cls.attr("_from_cpp_mode")(py::cast(mode), converter, shape_env);
  // SafePyObject owns the weakref with interpreter-safe lifetime management,
  // and the shared_ptr owns the SafePyObject.
  PyObject* weakref = PyWeakref_NewRef(wrapper.ptr(), nullptr);
  if (weakref == nullptr) {
    throw py::error_already_set();
  }
  mode->fake_mode_pyobj_ =
      std::make_shared<c10::SafePyObject>(weakref, getPyInterpreter());
  return wrapper;
}

} // namespace torch::fake_tensor
