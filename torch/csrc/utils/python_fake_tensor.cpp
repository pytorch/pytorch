#include <torch/csrc/utils/python_fake_tensor.h>

#include <torch/csrc/PyInterpreter.h>

namespace torch {

py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (mode == nullptr) {
    return py::none();
  }
  // fake_mode_pyobj_ weakly references the python CppFakeTensorMode: the python
  // object owns this C++ mode through the capsule, so a strong ref would cycle.
  if (mode->fake_mode_pyobj_ != nullptr) {
    PyObject* obj = nullptr;
    if (PyWeakref_GetRef(
            mode->fake_mode_pyobj_->ptr(getPyInterpreter()), &obj) > 0) {
      return py::reinterpret_steal<py::object>(obj);
    }
  }
  // The wrapper is gone (or was never made). Mint one around this same C++
  // mode: all mode state lives in C++, so the new wrapper is equivalent. Cache
  // a weakref to it so every later lookup returns the same object.
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
  // The Python wrapper owns the C++ mode, so retain only a Python weakref to
  // avoid a cycle. SafePyObject owns that weakref with interpreter-safe
  // lifetime management, and the shared_ptr owns the SafePyObject.
  PyObject* weakref = PyWeakref_NewRef(wrapper.ptr(), nullptr);
  TORCH_CHECK(weakref != nullptr, "failed to weakref CppFakeTensorMode");
  mode->fake_mode_pyobj_ =
      std::make_shared<c10::SafePyObject>(weakref, getPyInterpreter());
  return wrapper;
}

} // namespace torch
