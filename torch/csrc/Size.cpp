#include "Size.h"

#include <string>
#include "torch/csrc/utils/python_strings.h"
#include "THP.h"

PyObject* THPSizeClass = NULL;

struct THPSize {
  PyTupleObject tuple;
};

PyObject * THPSize_New(int dim, long *sizes)
{
  PyTypeObject* type = (PyTypeObject*)THPSizeClass;
  PyObject* self = type->tp_alloc(type, dim);
  if (!self) {
    return NULL;
  }
  for (int i = 0; i < dim; ++i) {
    PyTuple_SET_ITEM(self, i, PyLong_FromLong(sizes[i]));
  }
  return self;
}

static PyObject * THPSize_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  THPObjectPtr self(PyTuple_Type.tp_new(type, args, kwargs));
  if (self) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(self); ++i) {
      PyObject *item = PyTuple_GET_ITEM(self.get(), i);
      if (!THPUtils_checkLong(item)) {
        return PyErr_Format(PyExc_TypeError, "torch.Size() takes an iterable of 'int' (item %zd is '%s')",
            i, Py_TYPE(item)->tp_name);
      }
    }
  }
  return self.release();
}

static PyObject * THPSize_repr(THPSize *self)
{
  std::string repr("torch.Size([");
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    if (i != 0) {
      repr += ", ";
    }
    repr += std::to_string(PyLong_AsLong(PyTuple_GET_ITEM(self, i)));
  }
  repr += "])";
  return THPUtils_packString(repr);
}

extern PyTypeObject THPSizeType;

template<typename FnType, FnType fn, typename ...Args>
static PyObject* wrap_tuple_fn(Args ... args)
{
  PyObject *result = (*fn)(std::forward<Args>(args)...);
  if (!result) return NULL;
  if (PyTuple_Check(result)) {
    return PyObject_CallFunctionObjArgs((PyObject*)&THPSizeType, result, NULL);
  }
  Py_INCREF(result);
  return result;
}

static auto sq_concat = PyTuple_Type.tp_as_sequence->sq_concat;
static auto sq_repeat = PyTuple_Type.tp_as_sequence->sq_repeat;
#if PY_MAJOR_VERSION == 2
static auto sq_slice = PyTuple_Type.tp_as_sequence->sq_slice;
#endif
static auto mp_subscript = PyTuple_Type.tp_as_mapping->mp_subscript;


static PySequenceMethods THPSize_as_sequence = {
  PyTuple_Type.tp_as_sequence->sq_length,
  wrap_tuple_fn<decltype(&sq_concat), &sq_concat>,
  wrap_tuple_fn<decltype(&sq_repeat), &sq_repeat>,
  PyTuple_Type.tp_as_sequence->sq_item,
#if PY_MAJOR_VERSION == 2
  wrap_tuple_fn<decltype(&sq_slice), &sq_slice>,
#else
  0,                                          /* sq_slice */
#endif
  0,                                          /* sq_ass_item */
  0,                                          /* sq_ass_slice */
  PyTuple_Type.tp_as_sequence->sq_contains
};

static PyMappingMethods THPSize_as_mapping = {
    PyTuple_Type.tp_as_mapping->mp_length,
    wrap_tuple_fn<decltype(&mp_subscript), &mp_subscript>,
    0
};


PyTypeObject THPSizeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch.Size",                          /* tp_name */
  sizeof(THPSize),                       /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPSize_repr,                /* tp_repr */
  0,                                     /* tp_as_number */
  &THPSize_as_sequence,                  /* tp_as_sequence */
  &THPSize_as_mapping,                   /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  &PyTuple_Type,                         /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPSize_pynew,                         /* tp_new */
};

bool THPSize_init(PyObject *module)
{
  THPSizeClass = (PyObject*)&THPSizeType;
  if (PyType_Ready(&THPSizeType) < 0)
    return false;
  Py_INCREF(&THPSizeType);
  PyModule_AddObject(module, "Size", (PyObject *)&THPSizeType);
  return true;
}
