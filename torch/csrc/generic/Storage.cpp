#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.cpp"
#else

/* A pointer to RealStorage class defined later in Python */
extern PyObject *THPStorageClass;

PyObject * THPStorage_(newObject)(THStorage *ptr)
{
  // TODO: error checking
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));
  PyObject *instance = PyObject_Call(THPStorageClass, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  return instance;
}

bool THPStorage_(IsSubclass)(PyObject *storage)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(storage), (PyObject*)&THPStorageType);
}

static void THPStorage_(dealloc)(THPStorage* self)
{
  THStorage_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPStorage_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *cdata_ptr = NULL;     // keyword-only arg - cdata pointer value
  THPStorage *storage_arg = NULL; // storage to be viewed on
  long storage_arg_offset = 0;    // offset for storage view
  long storage_arg_size = -1;     // size for storage view
  THPObjectPtr iterator;          // not null iff got a single iterable
  long size = -1;                 // non-negative iff got a number - new storage size
  bool args_ok = true;

  if (kwargs != NULL && PyDict_Size(kwargs) == 1) {
    cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
    args_ok = cdata_ptr != NULL;
  } else if (args != NULL && PyTuple_Size(args) == 1) {
    PyObject *arg = PyTuple_GET_ITEM(args, 0);
    if (THPUtils_checkLong(arg)) {
      args_ok = THPUtils_getLong(PyTuple_GET_ITEM(args, 0), &size);
    } else {
      iterator = PyObject_GetIter(arg);
      args_ok = iterator != nullptr;
      if (args_ok) {
        size = PyObject_Length(arg);
        args_ok = size != -1;
      }
    }
  // Storage view
  } else if (args != NULL && PyTuple_Size(args) >= 1 && THPStorage_(IsSubclass)(PyTuple_GET_ITEM(args, 0))) {
    storage_arg = (THPStorage *)PyTuple_GET_ITEM(args, 0);
    if (PyTuple_Size(args) >= 2 && !THPUtils_getLong(PyTuple_GET_ITEM(args, 1), &storage_arg_offset))
        return NULL;
    storage_arg_size = storage_arg->cdata->size - storage_arg_offset;
    if (PyTuple_Size(args) >= 3 && !THPUtils_getLong(PyTuple_GET_ITEM(args, 2), &storage_arg_size))
        return NULL;
    if (storage_arg_offset < 0 || storage_arg_offset >= storage_arg->cdata->size) {
      THPUtils_setError("Invalid storage offset (%ld)!\n", storage_arg_offset);
      return NULL;
    }
    if (storage_arg_size < 1 || storage_arg_size > storage_arg->cdata->size - storage_arg_offset) {
      THPUtils_setError("Invalid storage size (got %ld, but should be between 0 and %ld)!\n",
          storage_arg_size);
      return NULL;
    }
    if (PyTuple_Size(args) >= 4)
      args_ok = false;
  } else if (args && PyTuple_Size(args) != 0) {
    args_ok = false;
  }
  if (!args_ok) {
    // TODO: nice error mossage
    THPUtils_setError("invalid arguments");
    return NULL;
  }

  THPStoragePtr self = (THPStorage *)type->tp_alloc(type, 0);
  if (self != nullptr) {
    if (cdata_ptr) {
      THStorage *ptr = (THStorage*)PyLong_AsVoidPtr(cdata_ptr);
      self->cdata = ptr;
    } else if (storage_arg) {
      real *data_ptr = storage_arg->cdata->data + storage_arg_offset;
      THStoragePtr storage = THStorage_(newWithData)(data_ptr, storage_arg_size);
      storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
      storage->view = storage_arg->cdata;
      THStorage_(retain)(LIBRARY_STATE storage_arg->cdata);
      self->cdata = storage.release();
    } else if (iterator == nullptr && size >= 0) {
      self->cdata = THStorage_(newWithSize)(LIBRARY_STATE size);
    } else if (iterator != nullptr) {
      self->cdata = THStorage_(newWithSize)(LIBRARY_STATE size);
      long items_processed = 0;
      THPObjectPtr item;
      real v;
      while ((item = PyIter_Next(iterator))) {
        if (!THPUtils_(parseReal)(item, &v)) {
          THPUtils_setError("expected a numeric type, but got %s", Py_TYPE(item)->tp_name);
          return NULL;
        }
        if (items_processed == size) {
          // TODO: error - iterator has too many items
          return NULL;
        }
#ifndef THC_GENERIC_FILE
        self->cdata->data[items_processed++] = v;
#else
        // TODO: this might be slow - consider batched updates?
        THCStorage_(set)(LIBRARY_STATE self->cdata, items_processed++, v);
#endif
      }
      // Iterator raised an exception
      if (PyErr_Occurred()) {
        return NULL;
      }
      // Iterator was too short
      if (items_processed < size) {
        // TODO; error message
        return NULL;
      }
    } else {
      self->cdata = THStorage_(new)(LIBRARY_STATE_NOARGS);
    }

    if (self->cdata == NULL)
      return NULL;
  }
  return (PyObject *)self.release();
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPStorage_(length)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return THStorage_(size)(LIBRARY_STATE self->cdata);
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject * THPStorage_(get)(THPStorage *self, PyObject *index)
{
  HANDLE_TH_ERRORS
  /* Integer index */
  long nindex;
  if ((PyLong_Check(index) || PyInt_Check(index))
      && THPUtils_getLong(index, &nindex) == 1 ) {
    if (nindex < 0)
      nindex += THStorage_(size)(LIBRARY_STATE self->cdata);
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || \
    defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
    return PyFloat_FromDouble(THStorage_(get)(LIBRARY_STATE self->cdata, nindex));
#elif defined(THC_REAL_IS_HALF)
    return PyFloat_FromDouble(THC_half2float(THStorage_(get)(LIBRARY_STATE self->cdata, nindex)));
#else
    return PyLong_FromLong(THStorage_(get)(LIBRARY_STATE self->cdata, nindex));
#endif
  /* Slice index */
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, slicelength, len;
    len = THStorage_(size)(LIBRARY_STATE self->cdata);
    if (!THPUtils_(parseSlice)(index, len, &start, &stop, &slicelength))
      return NULL;

    real *data = THStorage_(data)(LIBRARY_STATE self->cdata);
#ifndef THC_GENERIC_FILE
    // TODO: this can leak memory if newWithData fails
    real *new_data = (real*)THAlloc(slicelength * sizeof(real));
    memcpy(new_data, data + start, slicelength * sizeof(real));
    THStoragePtr new_storage = THStorage_(newWithData)(LIBRARY_STATE new_data, slicelength);
#else
    THStoragePtr new_storage = THStorage_(newWithSize)(LIBRARY_STATE slicelength);
    THStoragePtr view = THStorage_(newWithData)(LIBRARY_STATE data + start, slicelength);
    THStorage_(clearFlag)(LIBRARY_STATE view, TH_STORAGE_FREEMEM);
    THStorage_(copy)(LIBRARY_STATE new_storage, view);
#endif
    PyObject *_ret = THPStorage_(newObject)(new_storage);
    new_storage.release();
    return _ret;
  }
  char err_string[512];
  snprintf (err_string, 512,
      "%s %s", "Only indexing with integers and slices supported, but got type: ",
      index->ob_type->tp_name);
  PyErr_SetString(PyExc_RuntimeError, err_string);
  return NULL;
  END_HANDLE_TH_ERRORS
}

static int THPStorage_(set)(THPStorage *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
  real rvalue;
  if (!THPUtils_(parseReal)(value, &rvalue))
    return -1;

  long nindex;
  if ((PyLong_Check(index) || PyInt_Check(index))
      && THPUtils_getLong(index, &nindex) == 1) {
    THStorage_(set)(LIBRARY_STATE self->cdata, nindex, rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, len;
    len = THStorage_(size)(LIBRARY_STATE self->cdata);
    if (!THPUtils_(parseSlice)(index, len, &start, &stop, NULL))
      return -1;
    // TODO: check the bounds only once
    for (;start < stop; start++)
      THStorage_(set)(LIBRARY_STATE self->cdata, start, rvalue);
    return 0;
  }
  char err_string[512];
  snprintf (err_string, 512, "%s %s",
      "Only indexing with integers and slices supported, but got type: ",
      index->ob_type->tp_name);
  PyErr_SetString(PyExc_RuntimeError, err_string);
  return -1;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyMappingMethods THPStorage_(mappingmethods) = {
  (lenfunc)THPStorage_(length),
  (binaryfunc)THPStorage_(get),
  (objobjargproc)THPStorage_(set)
};

// TODO: implement equality
PyTypeObject THPStorageType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPStorageBaseStr,         /* tp_name */
  sizeof(THPStorage),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPStorage_(dealloc),      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  &THPStorage_(mappingmethods),          /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,   /* will be assigned in init */    /* tp_methods */
  0,   /* will be assigned in init */    /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPStorage_(pynew),                    /* tp_new */
};

static struct PyMemberDef THPStorage_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPStorage, cdata), READONLY, NULL},
  {NULL}
};

#include "StorageMethods.cpp"

bool THPStorage_(init)(PyObject *module)
{
  THPStorageType.tp_methods = THPStorage_(methods);
  THPStorageType.tp_members = THPStorage_(members);
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject *)&THPStorageType);
  return true;
}

#endif
