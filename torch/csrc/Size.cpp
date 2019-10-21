#include <torch/csrc/Size.h>

#include <string>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_tuples.h>

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/tracer.h>

struct THPSize {
  PyTupleObject tuple;
};

PyObject * THPSize_New(const torch::autograd::Variable& var)
{
  if (!torch::jit::tracer::isTracing()) {
    auto sizes = var.sizes();
    return THPSize_NewFromSizes(var.dim(), sizes.data());
  }
  auto self = THPObjectPtr(THPSizeType.tp_alloc(&THPSizeType, var.dim()));
  if (!self) throw python_error();

  for (int64_t i = 0; i < var.dim(); ++i) {
    PyObject *py_size_tensor = THPVariable_Wrap(torch::jit::tracer::getSizeOf(var, i));
    if (!py_size_tensor) throw python_error();
    PyTuple_SET_ITEM(self.get(), i, py_size_tensor);
  }

  return self.release();
}

PyObject * THPSize_NewFromSizes(int dim, const int64_t *sizes)
{
  auto self = THPObjectPtr(THPSizeType.tp_alloc(&THPSizeType, dim));
  if (!self) throw python_error();
  THPUtils_packInt64Array(self, dim, sizes);
  return self.release();
}

static bool isTracedZeroDimVar(PyObject *item) {
  if (!THPVariable_Check(item)) return false;
  auto & var = reinterpret_cast<THPVariable*>(item)->cdata;
  return var.dim() == 0 && torch::jit::tracer::getValueTrace(var);
}

static PyObject * THPSize_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  THPObjectPtr self(PyTuple_Type.tp_new(type, args, kwargs));
  if (self) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(self); ++i) {
      PyObject *item = PyTuple_GET_ITEM(self.get(), i);
      if (THPUtils_checkLong(item)) {
        continue;
      }
      if (torch::jit::tracer::isTracing() && isTracedZeroDimVar(item)) {
        continue;
      }
      // item.__index__() works with 0-dim tensors and tensors with one element
      THPObjectPtr number(PyNumber_Index(item));
      if (number && THPUtils_checkLong(number.get())) {
        Py_INCREF(number.get());
        auto status = PyTuple_SetItem(self, i, number.get());
        if (status != 0) {
          throw python_error();
        }
        continue;
      }
      return PyErr_Format(PyExc_TypeError,
                          "torch.Size() takes an iterable of 'int' (item %zd is '%s')",
                          i, Py_TYPE(item)->tp_name);
    }
  }
  return self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPSize_repr(THPSize *self)
{
  HANDLE_TH_ERRORS
  std::string repr("torch.Size([");
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    if (i != 0) {
      repr += ", ";
    }
    repr += std::to_string(PyLong_AsLong(PyTuple_GET_ITEM(self, i)));
  }
  repr += "])";
  return THPUtils_packString(repr);
  END_HANDLE_TH_ERRORS
}

extern PyTypeObject THPSizeType;

template<typename FnType, FnType fn, typename ...Args>
static PyObject* wrap_tuple_fn(Args ... args)
{
  THPObjectPtr result((*fn)(std::forward<Args>(args)...));
  if (!result) return nullptr;
  if (PyTuple_Check(result.get())) {
    return PyObject_CallFunctionObjArgs((PyObject*)&THPSizeType, result.get(), nullptr);
  }
  return result.release();
}

// We use an anonymous namespace instead of static to work around
// (what @peterjc123 think is) a bug in Visual Studio
namespace {
  auto sq_concat = PyTuple_Type.tp_as_sequence->sq_concat;
  auto sq_repeat = PyTuple_Type.tp_as_sequence->sq_repeat;
  #if PY_MAJOR_VERSION == 2
  auto sq_slice = PyTuple_Type.tp_as_sequence->sq_slice;
  #endif
  binaryfunc mp_subscript = PyTuple_Type.tp_as_mapping->mp_subscript;
}


static PySequenceMethods THPSize_as_sequence = {
  nullptr,                                          /* sq_length */
  wrap_tuple_fn<decltype(&sq_concat), &sq_concat>,
  wrap_tuple_fn<decltype(&sq_repeat), &sq_repeat>,
  nullptr,                                          /* sq_item */
#if PY_MAJOR_VERSION == 2
  wrap_tuple_fn<decltype(&sq_slice), &sq_slice>,
#else
  nullptr,                                          /* sq_slice */
#endif
  nullptr,                                          /* sq_ass_item */
  nullptr,                                          /* sq_ass_slice */
  nullptr                                           /* sq_contains */
};

static PyMappingMethods THPSize_as_mapping = {
    nullptr,                                        /* mp_length */
    wrap_tuple_fn<decltype(&mp_subscript), &mp_subscript>,
    nullptr
};

static PyObject *THPSize_numel(THPSize *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  int64_t numel = 1;
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    numel *= PyLong_AsLong(PyTuple_GET_ITEM(self, i));
  }
  return THPUtils_packInt64(numel);
  END_HANDLE_TH_ERRORS
}

static PyObject *THPSize_reduce(THPSize *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto ret = THPObjectPtr{PyTuple_New(2)};
  if (!ret) throw python_error();

  auto obj = (PyObject*)(&THPSizeType);
  Py_INCREF(&THPSizeType);
  PyTuple_SET_ITEM(ret.get(), 0, obj);

  THPObjectPtr t(PyTuple_New(PyTuple_Size((PyObject*)self)));
  if (!t) throw python_error();
  for (Py_ssize_t i = 0; i < PyTuple_Size((PyObject*)self); ++i) {
    auto d = PyTuple_GET_ITEM(self, i);
    Py_INCREF(d);
    PyTuple_SET_ITEM(t.get(), i, d);
  }

  THPObjectPtr dims(Py_BuildValue("(O)", t.get()));
  if (!dims) throw python_error();
  PyTuple_SET_ITEM(ret.get(), 1, dims.release());

  return ret.release();
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THPSize_methods[] = {
  {"numel",       (PyCFunction)THPSize_numel,       METH_NOARGS,  nullptr},
  {"__reduce__",  (PyCFunction)THPSize_reduce,      METH_NOARGS,  nullptr},
  {nullptr}
};


PyTypeObject THPSizeType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
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
  0,                               /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPSize_methods,                       /* tp_methods */
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

void THPSize_init(PyObject *module)
{
  if (PyType_Ready(&THPSizeType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPSizeType);
  if (PyModule_AddObject(module, "Size", (PyObject*)&THPSizeType) < 0) {
    throw python_error();
  }
}
