#include <torch/csrc/FakeTensorMode.h>
#include <torch/csrc/PyInterpreter.h>

#include <mutex>

namespace torch::fake_tensor {

// Note [C++ FakeTensorMode handle]
// A C++ FakeTensorMode is exposed to Python as one object, a CppFakeTensorMode,
// which subclasses the mode's binding (torch._C._CppFakeTensorMode). It is the
// fake mode that the Python parts of fake dispatch (op implementations, the
// fake tensor converter) see. TODO: it does not yet have all of the
// FakeTensorMode interface they use (e.g. is_our_fake, allow_fallback_kernels,
// in_kernel_invocation, which is thread-local for a C++ mode); it gains it as
// C++ fake dispatch comes to call them.
//
// Fakes, the fake mode slot (FakeTensorModeTLS) and the ThreadLocalStates that
// capture it hold a mode through a c10::FakeTensorModeHandle, which owns both
// the mode and its Python object:
//
//   fake tensor / slot --shared_ptr--> handle --mode--> C++ mode
//                                             --py_mode--> CppFakeTensorMode
//   CppFakeTensorMode --shared_ptr (pybind11 holder)--> C++ mode
//   C++ mode --handle_ (weak_ptr)--> handle
//
// So, like a Python FakeTensorMode, which its FakeTensors and the FAKE slot
// hold, the Python object lives at least as long as anything uses the mode,
// and lookups (getCppFakeTensorModePyObj) return that object. The mode holds
// its handle only weakly, so there is no cycle. A cycle back through Python
// would need an attribute on the object holding a fake, which GC could not
// see through the handle; so CppFakeTensorMode has __slots__ that hold no
// fakes, and getCppFakeTensorModeHandle accepts no other type, including a
// subclass.
//
// A mode has at most one handle at a time: getCppFakeTensorModeHandle reuses
// the live one, and makes a new one only once every holder of the last is
// gone. C++ code copies a handle from a fake or the slot, and a mode made in
// C++ (which has no Python object) makes its one handle itself. As a mode may
// get a new handle over time, code compares and keys modes by handle->mode,
// never by handle.
//
// Copying or dropping a handle reference takes no GIL. When the last one
// drops, the handle releases its Python object through SafePyObject, which
// takes the GIL, as the mode's shape_env_ and fake_tensor_converter_ do.
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

std::shared_ptr<c10::FakeTensorModeHandle> getCppFakeTensorModeHandle(
    py::handle py_mode) {
  TORCH_CHECK_TYPE(
      Py_TYPE(py_mode.ptr()) ==
          reinterpret_cast<PyTypeObject*>(
              get_cpp_fake_tensor_mode_class().ptr()),
      "Expected a CppFakeTensorMode, got ",
      Py_TYPE(py_mode.ptr())->tp_name);
  auto mode = py::cast<std::shared_ptr<c10::FakeTensorMode>>(py_mode);
  // Under free-threaded Python, threads can get the same mode's handle at
  // once. Nothing under the lock runs Python code or releases a handle.
  std::lock_guard<std::mutex> lock(mode->handle_mutex_);
  auto handle = mode->handle_.lock();
  if (handle == nullptr) {
    handle =
        std::make_shared<c10::FakeTensorModeHandle>(c10::FakeTensorModeHandle{
            mode,
            std::make_shared<c10::SafePyObject>(
                py_mode.inc_ref().ptr(), getPyInterpreter())});
    mode->handle_ = handle;
  }
  return handle;
}

py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorModeHandle>& handle) {
  if (handle == nullptr) {
    return py::none();
  }
  TORCH_CHECK(
      handle->py_mode != nullptr,
      "This C++ FakeTensorMode was made in C++ and has no Python object");
  return py::reinterpret_borrow<py::object>(
      handle->py_mode->ptr(getPyInterpreter()));
}

} // namespace torch::fake_tensor
