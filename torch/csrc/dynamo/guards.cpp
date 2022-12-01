#define PY_SSIZE_T_CLEAN
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/extension.h>
#include <sstream>

namespace {

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  bool grad_mode_enabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    return (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        grad_mode_enabled(at::GradMode::is_enabled()) {}
};

class TensorCheck {
 public:
  TensorCheck(
      const LocalState& state,
      PyTypeObject* pt,
      const at::Tensor& v,
      bool dynamic_shapes)
      : pytype(pt),
        dispatch_key_(state.apply(v.key_set()).raw_repr()),
        dtype_(v.dtype().toScalarType()),
        requires_grad_(state.grad_mode_enabled && v.requires_grad()),
        dynamic_shapes_(dynamic_shapes) {
    auto ndim = v.ndimension();
    const auto& sizes = v.sizes();
    const auto& strides = v.strides();
    sizes_.reserve(ndim);
    strides_.reserve(ndim);
    for (auto i : c10::irange(ndim)) {
      sizes_.emplace_back(sizes[i]);
      strides_.emplace_back(strides[i]);
    }
  }

  bool check(const LocalState& state, const at::Tensor& v) {
    if (dispatch_key_ != state.apply(v.key_set()).raw_repr() ||
        dtype_ != v.dtype().toScalarType() ||
        requires_grad_ != (state.grad_mode_enabled && v.requires_grad())) {
      return false;
    }
    auto ndim = static_cast<size_t>(v.ndimension());
    if (ndim != sizes_.size()) {
      return false;
    }
    if (!dynamic_shapes_) {
      const auto& sizes = v.sizes();
      const auto& strides = v.strides();
      for (auto i : c10::irange(ndim)) {
        if (sizes_[i] != sizes[i] || strides_[i] != strides[i]) {
          return false;
        }
      }
    }
    return true;
  }

  std::string check_verbose(
      const LocalState& state,
      const at::Tensor& v,
      std::string tensor_name) {
    std::stringstream fail_reason;
    fail_reason << "tensor '" << tensor_name << "' ";
    if (dispatch_key_ != state.apply(v.key_set()).raw_repr()) {
      // return fmt::format("tensor dispatch key mismatch. expected {}, actual
      // {}", dispatch_key_, state.apply(v.key_set()).raw_repr());
      fail_reason << "dispatch key set mismatch. expected "
                  << c10::DispatchKeySet(
                         c10::DispatchKeySet::RAW, dispatch_key_)
                  << ", actual " << state.apply(v.key_set());
      return fail_reason.str();
    } else if (dtype_ != v.dtype().toScalarType()) {
      // return fmt::format("tensor dtype mismatch. expected {}, actual {}",
      // dtype_, v.dtype().toScalarType());
      fail_reason << "dtype mismatch. expected " << dtype_ << ", actual "
                  << v.dtype().toScalarType();
      return fail_reason.str();
    } else if (
        requires_grad_ != (state.grad_mode_enabled && v.requires_grad())) {
      // return fmt::format("tensor requires_grad mismatch. expected {}",
      // requires_grad_);
      fail_reason << "requires_grad mismatch. expected requires_grad="
                  << requires_grad_;
      return fail_reason.str();
    }
    size_t ndim = static_cast<size_t>(v.ndimension());
    if (ndim != sizes_.size()) {
      // return fmt::format("tensor rank mismatch. expected {}, actual {}",
      // sizes_.size(), ndim);
      fail_reason << "rank mismatch. expected " << sizes_.size() << ", actual "
                  << ndim;
      return fail_reason.str();
    }
    if (!dynamic_shapes_) {
      const auto& sizes = v.sizes();
      const auto& strides = v.strides();
      for (auto i : c10::irange(ndim)) {
        if (sizes_[i] != sizes[i]) {
          // return fmt::format("tensor size mismatch at index {}. expected {},
          // actual {}", i, sizes_[i], sizes[i]);
          fail_reason << "size mismatch at index " << i << ". expected "
                      << sizes_[i] << ", actual " << sizes[i];
          return fail_reason.str();
        } else if (strides_[i] != strides[i]) {
          // return fmt::format("tensor strides mismatch at index {}. expected
          // {}, actual {}", i, strides_[i]);
          fail_reason << "strides mismatch at index " << i << ". expected "
                      << strides_[i] << ", actual " << strides[i];
          return fail_reason.str();
        }
      }
    }
    return "";
  }

  PyTypeObject* pytype;

 private:
  uint64_t dispatch_key_; // DispatchKeySet includes device/layout
  at::ScalarType dtype_;
  bool requires_grad_;
  bool dynamic_shapes_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

typedef std::vector<TensorCheck> ChecksList;

typedef struct {
  PyObject_HEAD;
  ChecksList* checks;
} TensorGuards;

static void TensorGuards_dealloc(TensorGuards* self) {
  if (self->checks != NULL) {
    delete self->checks;
    self->checks = NULL;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TensorGuards_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  TensorGuards* self = (TensorGuards*)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->checks = new ChecksList();
  }
  return (PyObject*)self;
}

static int TensorGuards_init(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwds) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return -1;
  }
  PyObject* dynamic_shapes_py = PyDict_GetItemString(kwds, "dynamic_shapes");
  if (dynamic_shapes_py == NULL) {
    PyErr_SetString(PyExc_TypeError, "missing dynamic_shapes=...");
    return -1;
  }
  bool dynamic_shapes = PyObject_IsTrue(dynamic_shapes_py);

  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);
  checks.reserve(len);
  LocalState state;
  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "expected Tensor()");
      return -1;
    }
    checks.emplace_back(TensorCheck(
        state, Py_TYPE(item), THPVariable_Unpack(item), dynamic_shapes));
  }
  return 0;
}

PyObject* TensorGuards_check(TensorGuards* self, PyObject* args) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return NULL;
  }
  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);

  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return NULL;
  }

  LocalState state;

  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (Py_TYPE(item) != checks[i].pytype) {
      Py_RETURN_FALSE;
    }
    if (!checks[i].check(state, THPVariable_Unpack(item))) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

PyObject* TensorGuards_check_verbose(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwargs) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return NULL;
  }
  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);

  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return NULL;
  }

  PyObject* tensor_check_names_py =
      PyDict_GetItemString(kwargs, "tensor_check_names");
  if (tensor_check_names_py == NULL) {
    PyErr_SetString(PyExc_TypeError, "missing tensor_check_names kwarg");
    return NULL;
  }

  if (!PyList_Check(tensor_check_names_py)) {
    PyErr_SetString(PyExc_TypeError, "tensor_check_names kwarg must be a list");
    return NULL;
  }

  auto names_size = PyList_Size(tensor_check_names_py);
  if (names_size != static_cast<decltype(names_size)>(checks.size())) {
    PyErr_SetString(
        PyExc_TypeError,
        "tensor_check_names should be the same size as # tensors");
    return NULL;
  }

  std::vector<std::string> tensor_check_names;
  tensor_check_names.reserve(names_size);
  for (auto i : c10::irange(names_size)) {
    PyObject* value = PyList_GetItem(tensor_check_names_py, i);
    if (!PyUnicode_Check(value)) {
      PyErr_SetString(
          PyExc_TypeError, "tensor_check_names must only contain strings");
      return NULL;
    }
    tensor_check_names.emplace_back(PyUnicode_AsUTF8(value));
  }

  LocalState state;
  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (Py_TYPE(item) != checks[i].pytype) {
      std::stringstream fail_reason;
      PyObject* type_str = PyObject_Str(PyObject_Type(item));
      fail_reason << "expected type of '" << tensor_check_names[i]
                  << "' to be a tensor type, ";
      if (!type_str) {
        fail_reason << "but found a different type";
      } else {
        fail_reason << "' but found " << PyUnicode_AsUTF8(type_str);
      }
      return Py_BuildValue("s", fail_reason.str().c_str());
    }
    std::string fail_reason = checks[i].check_verbose(
        state, THPVariable_Unpack(item), tensor_check_names[i]);
    if (fail_reason.length() > 0) {
      return Py_BuildValue("s", fail_reason.c_str());
    }
  }

  Py_RETURN_TRUE;
}

static PyMethodDef TensorGuards_methods[] = {
    {"check", (PyCFunction)TensorGuards_check, METH_VARARGS, ""},
    {"check_verbose",
     (PyCFunction)(void*)TensorGuards_check_verbose,
     METH_VARARGS | METH_KEYWORDS,
     "verbose fail reasons for failed checks"},
    {NULL} /* Sentinel */
};

static PyTypeObject TensorGuardsType = {
    // NOLINTNEXTLINE
    PyVarObject_HEAD_INIT(NULL, 0)};

static PyObject* check_type_id(PyObject* dummy, PyObject* args) {
  // faster `lambda obj, expected: id(type(obj)) == expected`
  PyObject* obj;
  unsigned long expected;
  if (!PyArg_ParseTuple(args, "Ok", &obj, &expected)) {
    return NULL;
  }
  if (Py_TYPE(obj) == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* check_obj_id(PyObject* dummy, PyObject* args) {
  // faster `lambda obj, expected: id(obj) == expected`
  PyObject* obj;
  unsigned long expected;
  if (!PyArg_ParseTuple(args, "Ok", &obj, &expected)) {
    return NULL;
  }
  if (obj == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* assert_size_stride(PyObject* dummy, PyObject* args) {
  /*
   Assert that a given tensor has a given size/stride, but ignore strides
   of size==1 dimensions.  Implemented in C++ as this is on the hot path.
  */
  PyObject* item;
  PyObject* size;
  PyObject* stride;
  if (!PyArg_ParseTuple(args, "OOO", &item, &size, &stride)) {
    return NULL;
  }
  if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor()");
    return NULL;
  }
  if (!PyTuple_CheckExact(size) || !PyTuple_CheckExact(stride)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return NULL;
  }
  at::Tensor tensor = THPVariable_Unpack(item);
  int64_t ndim = tensor.ndimension();
  if (PyTuple_GET_SIZE(size) != ndim || PyTuple_GET_SIZE(stride) != ndim) {
    PyErr_SetString(PyExc_AssertionError, "wrong number of dimensions");
    return NULL;
  }
  for (auto i : c10::irange(ndim)) {
    int64_t want_size = THPUtils_unpackLong(PyTuple_GET_ITEM(size, i));
    int64_t want_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(stride, i));
    int64_t actual_size = tensor.size(i);
    int64_t actual_stride = tensor.stride(i);
    if (want_size != actual_size ||
        // ignore stride differences when size is 1
        (want_stride != actual_stride && actual_size > 1)) {
      std::stringstream msg;
      msg << "expected size " << actual_size << "==" << want_size << ", stride "
          << actual_stride << "==" << want_stride << " at dim=" << i;
      PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
      return NULL;
    }
  }
  Py_RETURN_TRUE;
}

static PyMethodDef _methods[] = {
    {"check_type_id", check_type_id, METH_VARARGS, NULL},
    {"check_obj_id", check_obj_id, METH_VARARGS, NULL},
    {"assert_size_stride", assert_size_stride, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.guards",
    "Module containing checks on tensors",
    -1,
    _methods};

} // namespace

PyObject* torch_c_dynamo_guards_init() {
  // initialize TensorGuardsType
  TensorGuardsType.tp_name = "torch._C._dynamo.guards.TensorGuards";
  TensorGuardsType.tp_basicsize = sizeof(TensorGuards);
  TensorGuardsType.tp_itemsize = 0;
  TensorGuardsType.tp_dealloc = (destructor)TensorGuards_dealloc;
  TensorGuardsType.tp_flags = Py_TPFLAGS_DEFAULT;
  TensorGuardsType.tp_doc = "Check properties of a torch.Tensor";
  TensorGuardsType.tp_methods = TensorGuards_methods;
  TensorGuardsType.tp_init = (initproc)TensorGuards_init;
  TensorGuardsType.tp_new = TensorGuards_new;

  PyObject* m;
  if (PyType_Ready(&TensorGuardsType) < 0)
    return NULL;

  m = PyModule_Create(&_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&TensorGuardsType);
  if (PyModule_AddObject(m, "TensorGuards", (PyObject*)&TensorGuardsType) < 0) {
    Py_DECREF(&TensorGuardsType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
