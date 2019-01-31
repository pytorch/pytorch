#include <Python.h>
#include <torch/csrc/np.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/THP.h>


PyObject* THNPArray_NewWithTensor(PyTypeObject* type, torch::autograd::Variable v)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto nd = (THNPArray*) obj;
    new (&nd->cdata) torch::autograd::Variable{make_variable_view (v, v.data())};
    nd->cdata.set_pyobj(obj);
  }
  return obj;
}

static PyObject *THNPArray_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  auto& default_type = torch::tensors::get_default_tensor_type();
  auto default_scalar_type = torch::tensors::get_default_scalar_type(); // XXX fix
  auto tensor = torch::utils::legacy_tensor_ctor(default_type, default_scalar_type, args, kwargs);
  return THNPArray_NewWithTensor(type, std::move(tensor));
}

static void THNPArray_dealloc(THNPArray* self)
{
  self->cdata.~Variable();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

namespace torch { namespace autograd {
  extern PyMethodDef np_compat_methods[];
}}

static PyObject *THNPArray_get_shape(THNPArray *self)
{
  HANDLE_TH_ERRORS
  auto& v = self->cdata;
  int dims = v.dim();
  PyObject* tup = PyTuple_New(dims);
  for (int64_t i = 0; i < dims; ++i) {
    PyTuple_SET_ITEM(tup, i, PyLong_FromLong(v.size(i)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static int THNPArray_set_shape(THNPArray *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  const int MAX_DIMS = 32;
  std::array<int64_t, MAX_DIMS> dims;
  int dim_idx = 0;
  _PyObject_Dump(obj);
  THPUtils_assertRet(-1, obj, "Cannot delete array shape");
  if (PyLong_Check(obj)) {
    dims[dim_idx++] = PyLong_AsLong(obj);
  } else { 
    PyObject *iterator = PyObject_GetIter(obj);
    THPUtils_assertRet(-1, iterator, "shape must be an integer or iterable of integers");
    PyObject *item = nullptr;
    while ((item = PyIter_Next(iterator))) {
      _PyObject_Dump(item);
      // XXX decref
      THPUtils_assertRet(-1, PyLong_Check(item), "shape must be an integer or iterable of integers");
      dims[dim_idx++] = PyLong_AsLong(item);
      Py_DECREF(item);
      THPUtils_assertRet(-1, dim_idx < MAX_DIMS, "cannot set shape to more than 32 dimensions");
    }
    Py_DECREF(iterator);
  }
  c10::IntArrayRef shape {&dims[0], dim_idx};
  
  self->cdata = self->cdata.view(shape);

  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}


static struct PyGetSetDef THNPArray_properties[] = {
  {"shape", (getter)THNPArray_get_shape, (setter)THNPArray_set_shape, nullptr, nullptr},
  {nullptr}
};

PyTypeObject THNPArrayBaseType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._np_compat._ndarray_base",                /* tp_name */
  sizeof(THNPArray),                   /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THNPArray_dealloc,   //FIXME    /* tp_dealloc */
  nullptr,                                     /* tp_print */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr, // FIXME &THPVariable_as_mapping,               /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                               /* tp_doc */
  nullptr, // FIXME (traverseproc)THPVariable_traverse,    /* tp_traverse */
  nullptr, // FIXME (inquiry)THPVariable_clear,            /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  torch::autograd::np_compat_methods,                                     /* tp_methods */
  nullptr,                                     /* tp_members */
  THNPArray_properties, // FIXME THPVariable_properties,                /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THNPArray_pynew,                      /* tp_new */
};

PyObject *THNPArrayClass = nullptr;

namespace torch { namespace autograd {
extern PyMethodDef np_compat_functions[];
PyObject *_np_compat_init(PyObject *, PyObject *) {
    auto np_module = THPObjectPtr(PyImport_ImportModule("torch.numpy"));
    if (!np_module) throw python_error();

    THNPArrayClass = PyObject_GetAttrString(np_module, "ndarray");
    if (!THNPArrayClass) throw python_error();
    Py_RETURN_NONE;
}
}}

static std::vector<PyMethodDef> functions;

PyObject* initNpModule() {
  THPUtils_addPyMethodDefs(functions, torch::autograd::np_compat_functions);
#if PY_MAJOR_VERSION == 2
  auto module = Py_InitModule("torch._np_compat", functions.data());
  assert(module);
#else
  static struct PyModuleDef npmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._np_compat",
     nullptr,
     -1,
     functions.data(),
  };
  auto module = PyModule_Create(&npmodule);
  assert(module);
#endif
    assert(PyType_Ready(&THNPArrayBaseType) == 0);
    Py_INCREF(&THNPArrayBaseType);
    PyModule_AddObject(module, "_ndarray_base", (PyObject *)&THNPArrayBaseType);
    return module;
}
