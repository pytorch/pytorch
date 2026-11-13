#include <torch/csrc/Exceptions.h>
#include <torch/csrc/FakeTensorMode.h>
#include <torch/csrc/PyInterpreter.h>

#include <mutex>
#include <new>
#include <type_traits>

namespace torch::fake_tensor {

// Note [C++ FakeTensorMode Python object lifetime]
// Fakes, TLS and saved ThreadLocalState own the C++ mode through shared_ptr.
// CppFakeTensorMode's native wrapper also owns it; the reverse reference is
// weak, so Python can collect the wrapper while C++ still uses the mode.
// Every C++-to-Python lookup reuses the live wrapper or recreates it around
// the same mode. All persistent state must therefore live on the C++ mode.
// Reconstruction does not create a converter for a mode made without one.
// A returned py::object keeps the wrapper alive for the duration of a callback.
//
// ShapeEnv and the converter must not strongly hold the mode's fakes: Python
// GC cannot see the references through the C++ mode. Only the exact slotted
// CppFakeTensorMode is supported, so wrappers cannot introduce extra state.
//
// The per-mode mutex protects the weak cache on free-threaded Python. Wrapper
// allocation runs outside the lock; racing lookups return the first live
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

struct PyCppFakeTensorMode {
  PyObject_HEAD
  std::shared_ptr<c10::FakeTensorMode> mode;
  PyObject* weakreflist;
};

static PyObject* cpp_fake_tensor_mode_new(
    PyTypeObject* type,
    PyObject*,
    PyObject*) {
  auto* self = reinterpret_cast<PyCppFakeTensorMode*>(type->tp_alloc(type, 0));
  if (self == nullptr) {
    return nullptr;
  }
  new (&self->mode) std::shared_ptr<c10::FakeTensorMode>();
  self->weakreflist = nullptr;
  return reinterpret_cast<PyObject*>(self);
}

static void cpp_fake_tensor_mode_dealloc(PyObject* obj) {
  auto* self = reinterpret_cast<PyCppFakeTensorMode*>(obj);
  if (self->weakreflist != nullptr) {
    PyObject_ClearWeakRefs(obj);
  }
  self->mode.~shared_ptr();
  Py_TYPE(obj)->tp_free(obj);
}

std::shared_ptr<c10::FakeTensorMode> getCppFakeTensorMode(py::handle py_mode) {
  TORCH_CHECK_TYPE(
      Py_TYPE(py_mode.ptr()) ==
          reinterpret_cast<PyTypeObject*>(
              get_cpp_fake_tensor_mode_class().ptr()),
      "Expected a CppFakeTensorMode (not a subclass, nor a bare ",
      "_CppFakeTensorMode), got ",
      Py_TYPE(py_mode.ptr())->tp_name);
  auto& mode = reinterpret_cast<PyCppFakeTensorMode*>(py_mode.ptr())->mode;
  TORCH_CHECK(mode != nullptr, "CppFakeTensorMode is not initialized");
  return mode;
}

static int cpp_fake_tensor_mode_init(
    PyObject* obj,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  PyObject* converter;
  PyObject* shape_env;
  PyObject* creation_traceback = Py_None;
  static const char* names[] = {
      "converter", "shape_env", "creation_traceback", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|O:_CppFakeTensorMode",
          const_cast<char**>(names),
          &converter,
          &shape_env,
          &creation_traceback)) {
    return -1;
  }
  TORCH_CHECK(converter != Py_None, "Expected a FakeTensorConverter");
  TORCH_CHECK_TYPE(
      Py_TYPE(obj) ==
          reinterpret_cast<PyTypeObject*>(
              get_cpp_fake_tensor_mode_class().ptr()),
      "Expected a CppFakeTensorMode (not a subclass, nor a bare ",
      "_CppFakeTensorMode), got ",
      Py_TYPE(obj)->tp_name);
  auto* self = reinterpret_cast<PyCppFakeTensorMode*>(obj);
  TORCH_CHECK(
      self->mode == nullptr, "CppFakeTensorMode is already initialized");
  std::shared_ptr<c10::SafePyObject> shape_env_obj;
  if (shape_env != Py_None) {
    shape_env_obj = std::make_shared<c10::SafePyObject>(
        Py_NewRef(shape_env), getPyInterpreter());
  }
  auto config = py::module::import("torch._functorch.config");
  auto mode = std::make_shared<c10::FakeTensorMode>(
      std::move(shape_env_obj),
      std::make_shared<c10::SafePyObject>(
          Py_NewRef(converter), getPyInterpreter()),
      config.attr("fake_tensor_allow_meta").cast<bool>(),
      config.attr("fake_tensor_allow_unsafe_data_ptr_access").cast<bool>());
  if (creation_traceback != Py_None) {
    mode->creation_traceback_ = std::make_shared<c10::SafePyObject>(
        Py_NewRef(creation_traceback), getPyInterpreter());
  }
  auto weakref =
      py::reinterpret_steal<py::object>(PyWeakref_NewRef(obj, nullptr));
  if (!weakref) {
    throw py::error_already_set();
  }
  mode->fake_mode_pyobj_ =
      std::make_shared<c10::SafePyObject>(weakref.ptr(), getPyInterpreter());
  weakref.release();
  TORCH_CHECK(
      self->mode == nullptr, "CppFakeTensorMode is already initialized");
  self->mode = std::move(mode);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

template <auto member>
static PyObject* get_mode_field(PyObject* self, void*) {
  HANDLE_TH_ERRORS
  return py::cast(getCppFakeTensorMode(self).get()->*member).release().ptr();
  END_HANDLE_TH_ERRORS
}

template <auto member>
static int set_mode_field(PyObject* self, PyObject* value, void*) {
  HANDLE_TH_ERRORS
  if (value == nullptr) {
    PyErr_SetString(
        PyExc_AttributeError, "Cannot delete a CppFakeTensorMode field");
    return -1;
  }
  auto mode = getCppFakeTensorMode(self);
  using T = std::remove_reference_t<decltype(mode.get()->*member)>;
  try {
    mode.get()->*member = py::cast<T>(py::handle(value));
  } catch (const py::cast_error& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return -1;
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

template <auto member>
static PyObject* get_mode_pyobject(PyObject* self, void*) {
  HANDLE_TH_ERRORS
  auto mode = getCppFakeTensorMode(self);
  const auto& value = mode.get()->*member;
  if constexpr (member == &c10::FakeTensorMode::fake_tensor_converter_) {
    TORCH_CHECK(value, "This C++ FakeTensorMode has no FakeTensorConverter");
  }
  return Py_NewRef(value ? value->ptr(getPyInterpreter()) : Py_None);
  END_HANDLE_TH_ERRORS
}

static PyGetSetDef mode_properties[] = {
    {"_creation_traceback",
     get_mode_pyobject<&c10::FakeTensorMode::creation_traceback_>,
     nullptr,
     nullptr,
     nullptr},
    {"static_shapes",
     get_mode_field<&c10::FakeTensorMode::static_shapes_>,
     set_mode_field<&c10::FakeTensorMode::static_shapes_>,
     nullptr,
     nullptr},
    {"allow_meta",
     get_mode_field<&c10::FakeTensorMode::allow_meta_>,
     set_mode_field<&c10::FakeTensorMode::allow_meta_>,
     nullptr,
     nullptr},
    {"_allow_unsafe_data_ptr_access",
     get_mode_field<&c10::FakeTensorMode::allow_unsafe_data_ptr_access_>,
     set_mode_field<&c10::FakeTensorMode::allow_unsafe_data_ptr_access_>,
     nullptr,
     nullptr},
    {"allow_non_fake_inputs",
     get_mode_field<&c10::FakeTensorMode::allow_non_fake_inputs_>,
     set_mode_field<&c10::FakeTensorMode::allow_non_fake_inputs_>,
     nullptr,
     nullptr},
    {"epoch",
     get_mode_field<&c10::FakeTensorMode::epoch_>,
     set_mode_field<&c10::FakeTensorMode::epoch_>,
     nullptr,
     nullptr},
    {"fake_tensor_converter",
     get_mode_pyobject<&c10::FakeTensorMode::fake_tensor_converter_>,
     nullptr,
     nullptr,
     nullptr},
    {"shape_env",
     get_mode_pyobject<&c10::FakeTensorMode::shape_env_>,
     nullptr,
     nullptr,
     nullptr},
    {nullptr}};

PyTypeObject PyCppFakeTensorModeType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._CppFakeTensorMode", /* tp_name */
    sizeof(PyCppFakeTensorMode), /* tp_basicsize */
    0, /* tp_itemsize */
    cpp_fake_tensor_mode_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    offsetof(PyCppFakeTensorMode, weakreflist), /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    mode_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    cpp_fake_tensor_mode_init, /* tp_init */
    nullptr, /* tp_alloc */
    cpp_fake_tensor_mode_new, /* tp_new */
};

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
    PyObject* py_mode = nullptr;
    if (PyWeakref_GetRef(cached->ptr(getPyInterpreter()), &py_mode) < 0) {
      throw py::error_already_set();
    }
    if (py_mode != nullptr) {
      return py::reinterpret_steal<py::object>(py_mode);
    }
  }
  // No live wrapper; see Note [C++ FakeTensorMode Python object lifetime].
  auto py_mode = py::reinterpret_steal<py::object>(cpp_fake_tensor_mode_new(
      reinterpret_cast<PyTypeObject*>(get_cpp_fake_tensor_mode_class().ptr()),
      nullptr,
      nullptr));
  if (!py_mode) {
    throw py::error_already_set();
  }
  reinterpret_cast<PyCppFakeTensorMode*>(py_mode.ptr())->mode = mode;
  // SafePyObject owns the weakref with interpreter-safe lifetime management,
  // and the shared_ptr owns the SafePyObject.
  PyObject* weakref = PyWeakref_NewRef(py_mode.ptr(), nullptr);
  if (weakref == nullptr) {
    throw py::error_already_set();
  }
  auto fresh = std::make_shared<c10::SafePyObject>(weakref, getPyInterpreter());
  std::lock_guard<std::mutex> lock(mode->fake_mode_pyobj_mutex_);
  // Another thread may have cached an object while this one allocated its
  // wrapper
  if (mode->fake_mode_pyobj_ != cached) {
    PyObject* installed = nullptr;
    if (PyWeakref_GetRef(
            mode->fake_mode_pyobj_->ptr(getPyInterpreter()), &installed) < 0) {
      throw py::error_already_set();
    }
    if (installed != nullptr) {
      return py::reinterpret_steal<py::object>(installed);
    }
  }
  mode->fake_mode_pyobj_.swap(fresh);
  return py_mode;
}

} // namespace torch::fake_tensor
