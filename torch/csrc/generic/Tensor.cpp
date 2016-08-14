#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.cpp"
#else

#ifdef WITH_NUMPY

#ifdef TH_REAL_IS_DOUBLE
#define NUMPY_TYPE_ENUM NPY_DOUBLE
#endif
#ifdef TH_REAL_IS_FLOAT
#define NUMPY_TYPE_ENUM NPY_FLOAT
#endif
#ifdef TH_REAL_IS_LONG
#define NUMPY_TYPE_ENUM NPY_INT64
#endif
#ifdef TH_REAL_IS_INT
#define NUMPY_TYPE_ENUM NPY_INT32
#endif
#ifdef TH_REAL_IS_BYTE
#define NUMPY_TYPE_ENUM NPY_UINT8
#endif

#endif

#include "TensorMethods.cpp"

extern PyObject *THPTensorClass;
PyObject * THPTensor_(newObject)(THTensor *ptr)
{
  // TODO: error checking
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = Py_BuildValue("{s:K}", "cdata", (unsigned long long) ptr);
  PyObject *instance = PyObject_Call(THPTensorClass, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  return instance;
}

bool THPTensor_(IsSubclass)(PyObject *tensor)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(tensor), (PyObject*)&THPTensorType);
}

static void THPTensor_(dealloc)(THPTensor* self)
{
  THTensor_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *cdata_arg = NULL;                 // keyword-only arg - cdata pointer value
  THLongStorage *sizes_arg = NULL;            // a storage with sizes for a new tensor
  THTensor *tensor_arg = NULL;                // a tensor to be viewed on
  // TODO: constructor from storage
  PyObject *iterable_arg = NULL;              // an iterable, with new tensor contents
  std::vector<size_t> iterator_lengths;       // a queue storing lengths of iterables at each depth
  bool args_ok = true;
#ifdef NUMPY_TYPE_ENUM
  THPObjectPtr numpy_array = NULL;
#endif

  if (kwargs && PyDict_Size(kwargs) == 1) {
    cdata_arg = PyDict_GetItemString(kwargs, "cdata");
    args_ok = cdata_arg != NULL;
  } else if (args && PyTuple_Size(args) == 1) {
    PyObject *arg = PyTuple_GET_ITEM(args, 0);
    if (THPTensor_(IsSubclass)(arg)) {
      tensor_arg = ((THPTensor*)arg)->cdata;
    } else if (THPLongStorage_IsSubclass(arg)) {
      sizes_arg = ((THPLongStorage*)arg)->cdata;
    } else if (THPUtils_checkLong(arg)) {
      sizes_arg = THPUtils_getLongStorage(args);
      args_ok = sizes_arg != nullptr;
#ifdef NUMPY_TYPE_ENUM
    } else if (PyArray_Check(arg) && PyArray_TYPE((PyArrayObject*)arg) == NUMPY_TYPE_ENUM) {
      numpy_array = PyArray_FromArray((PyArrayObject*)arg, nullptr, NPY_ARRAY_BEHAVED);
      args_ok = numpy_array != nullptr;
#endif
    } else {
      iterable_arg = arg;
      Py_INCREF(arg);
      THPObjectPtr item = arg;
      THPObjectPtr iter;
      while ((iter = PyObject_GetIter(item)) != nullptr) {
        Py_ssize_t length = PyObject_Length(item);
        iterator_lengths.push_back(length);
        // TODO length == 0 is an error too
        if (length == -1) {
          // TODO: error
          return NULL;
        }
        if (length > 0) {
          item = PyIter_Next(iter);
          if (item == nullptr) {
            // TODO: set error
            return NULL;
          }
        } else {
          break;
        }
      }
      if (iterator_lengths.size() > 1) {
        for (auto length: iterator_lengths) {
          if (length <= 0) {
            // TODO: error message
            THPUtils_setError("invalid size");
            return NULL;
          }
        }
      }
      args_ok = iterator_lengths.size() > 0;
      // We have accumulated some errors along the way.
      // Since we did all checking and ignored only the non-important
      // ones it's safe to clear them here.
      PyErr_Clear();
    }
  } else if (args && PyTuple_Size(args) > 0) {
    sizes_arg = THPUtils_getLongStorage(args);
    args_ok = sizes_arg != nullptr;
  }

  if (!args_ok) {
    // TODO: nice error mossage
    THPUtils_setError("invalid arguments");
    return NULL;
  }

  THPTensorPtr self = (THPTensor *)type->tp_alloc(type, 0);
  if (self != nullptr) {
    if (cdata_arg) {
      self->cdata = (THTensor*)PyLong_AsVoidPtr(cdata_arg);
    } else if (sizes_arg) {
      self->cdata = THTensor_(newWithSize)(LIBRARY_STATE sizes_arg, nullptr);
    } else if (tensor_arg) {
      self->cdata = THTensor_(newWithTensor)(LIBRARY_STATE tensor_arg);
#ifdef NUMPY_TYPE_ENUM
    } else if (numpy_array) {
      self->cdata = THPTensor_(fromNumpy)(numpy_array.get());
#endif
    } else if (iterable_arg && iterator_lengths.size() == 1 && iterator_lengths[0] == 0) {
      self->cdata = THTensor_(new)(LIBRARY_STATE_NOARGS);
    } else if (iterable_arg) {
      size_t iter_depth = iterator_lengths.size();
      std::stack<THPObjectPtr> iterator_stack;
      std::vector<size_t> items_processed(iter_depth);
      Py_INCREF(iterable_arg);
      THPObjectPtr item = iterable_arg;
      PyObject *iter;
      while (iterator_stack.size() != iter_depth) {
        iter = PyObject_GetIter(item);
        if (!iter) {
          THPUtils_setError("inconsistent iterator depth");
          return NULL;
        }
        iterator_stack.emplace(iter);
        item = PyIter_Next(iter);
        if (item == nullptr) {
          THPUtils_setError("error or empty iter");
          return NULL;
        }
      }
      THLongStoragePtr sizes = THLongStorage_newWithSize(iter_depth);
      long *sizes_data = sizes->data;
      for (size_t s: iterator_lengths) {
        *sizes_data++ = s;
      }
      THTensorPtr tensor = THTensor_(newWithSize)(LIBRARY_STATE sizes, NULL);

      // TODO CUDA
#ifndef THC_GENERIC_FILE
#define SET_ITEM if (!THPUtils_(parseReal)(item, data++)) return NULL
      real *data = tensor->storage->data;
#else
#define SET_ITEM if (!THPUtils_(parseReal)(item, &item_value)) return NULL; THStorage_(set)(LIBRARY_STATE storage, item_nr++, item_value)
      real item_value;
      size_t item_nr = 0;
      THStorage *storage = tensor->storage;
#endif
      SET_ITEM;
      items_processed[iter_depth-1]++;

      while (!iterator_stack.empty()) {
        PyObject *iter = iterator_stack.top().get();
        // Parse items
        if (iterator_stack.size() == iter_depth) {
          while ((item = PyIter_Next(iter))) {
            SET_ITEM;
            items_processed[iter_depth-1]++;
          }
          if (items_processed[iter_depth-1] != iterator_lengths[iter_depth-1]) {
            THPUtils_setError("inconsistent size");
            return NULL;
          }
          iterator_stack.pop(); // this deallocates the iter
        // Iterate on lower depths
        } else {
          item = PyIter_Next(iter);
          if (item == nullptr) {
            if (PyErr_Occurred())
              return NULL;
            if (items_processed[iterator_stack.size()-1]) {
              THPUtils_setError("inconsistent size");
              return NULL;
            }
            iterator_stack.pop(); // this deallocates the iter
          } else {
            PyObject *new_iter = PyObject_GetIter(item);
            if (!new_iter) {
              THPUtils_setError("non-iterable item");
              return NULL;
            }
            items_processed[iterator_stack.size()] = 0;
            iterator_stack.emplace(new_iter);
          }
        }
      }
      self->cdata = tensor.release();
    } else {
      self->cdata = THTensor_(new)(LIBRARY_STATE_NOARGS);
    }

    if (self->cdata == NULL)
      return NULL;
  }
  return (PyObject *)self.release();
  END_HANDLE_TH_ERRORS
}

#define INDEX_LONG(DIM, IDX_VARIABLE, TENSOR_VARIABLE, CASE_1D, CASE_MD)       \
  long idx;                                                                    \
  THPUtils_getLong(IDX_VARIABLE, &idx);                                        \
  long dimsize = THTensor_(size)(LIBRARY_STATE TENSOR_VARIABLE, DIM);          \
  idx = (idx < 0) ? dimsize + idx : idx;                                       \
                                                                               \
  THArgCheck(dimsize > 0, 1, "empty tensor");                                  \
  THArgCheck(idx >= 0 && idx < dimsize, 2, "out of range");                    \
                                                                               \
  if(THTensor_(nDimension)(LIBRARY_STATE TENSOR_VARIABLE) == 1) {              \
    CASE_1D;                                                                   \
  } else {                                                                     \
    CASE_MD;                                                                   \
  }

#define GET_OFFSET(t, idx)                                                     \
  t->storageOffset + t->stride[0] * idx;

static bool THPTensor_(_index)(THPTensor *self, PyObject *index,
    THTensor * &tresult, THStorage * &sresult, long &storage_offset)
{
  tresult = NULL;
  sresult = NULL;
  try {
    // Indexing with an integer
    if(PyLong_Check(index) || PyInt_Check(index)) {
      THTensor *self_t = self->cdata;
      INDEX_LONG(0, index, self_t,
        // 1D tensor
        sresult = self_t->storage;
        storage_offset = GET_OFFSET(self_t, idx),
        // >1D tensor
        tresult = THTensor_(newWithTensor)(LIBRARY_STATE self_t);
        THTensor_(select)(LIBRARY_STATE tresult, NULL, 0, idx)
      )
    // Indexing with a single element tuple
    } else if (PyTuple_Check(index) &&
           PyTuple_Size(index) == 1 &&
           (PyLong_Check(PyTuple_GET_ITEM(index, 0))
        || PyInt_Check(PyTuple_GET_ITEM(index, 0)))) {
      PyObject *index_obj = PyTuple_GET_ITEM(index, 0);
      tresult = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
      INDEX_LONG(0, index_obj, tresult,
          THTensor_(narrow)(LIBRARY_STATE tresult, NULL, 0, idx, 1),
          THTensor_(narrow)(LIBRARY_STATE tresult, NULL, 0, idx, 1)
        )
    // Indexing with a slice
    } else if (PySlice_Check(index)) {
      tresult = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
      Py_ssize_t start, end, length;
      if (!THPUtils_(parseSlice)(index, THTensor_(size)(LIBRARY_STATE tresult, 0), &start, &end, &length))
        return false;
      THTensor_(narrow)(LIBRARY_STATE tresult, NULL, 0, start, length);
    // Indexing multiple dimensions
    } else if(PyTuple_Check(index)) {
      THArgCheck(PyTuple_Size(index) <= THTensor_(nDimension)(LIBRARY_STATE self->cdata), 2,
              "Indexing too many dimensions");
      tresult = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
      int t_dim = 0;

      for(int dim = 0; dim < PyTuple_Size(index); dim++) {
        PyObject *dimidx = PyTuple_GET_ITEM(index, dim);
        if(THPUtils_checkLong(dimidx)) {
          INDEX_LONG(t_dim, dimidx, tresult,
              // 1D tensor
              sresult = tresult->storage;
              storage_offset = GET_OFFSET(tresult, idx);
              THTensor_(free)(LIBRARY_STATE tresult);
              tresult = NULL;
              return true,
              // >1D tensor
              THTensor_(select)(LIBRARY_STATE tresult, NULL, t_dim, idx)
            )
        } else if (PyTuple_Check(dimidx)) {
          long length = 1;
          if (PyTuple_Size(dimidx) == 0 || PyTuple_Size(dimidx) > 2 || !THPUtils_checkLong(PyTuple_GET_ITEM(dimidx, 0))) {
            PyErr_SetString(PyExc_RuntimeError, "Expected one or two integers");
            return false;
          }
          PyObject *index_obj = PyTuple_GET_ITEM(dimidx, 0);
          if (PyTuple_Size(dimidx) == 2) {
            long idx;
            if (!THPUtils_checkLong(PyTuple_GET_ITEM(dimidx, 1))) {
              THPUtils_setError("Expected one or two intetegers");
              return false;
            }
            THPUtils_getLong(index_obj, &idx);
            THPUtils_getLong(PyTuple_GET_ITEM(dimidx, 1), &length);
            length -= idx;
          }
          INDEX_LONG(t_dim, index_obj, tresult,
              THTensor_(narrow)(LIBRARY_STATE tresult, NULL, t_dim++, idx, length),
              THTensor_(narrow)(LIBRARY_STATE tresult, NULL, t_dim++, idx, length)
            )
        } else if (PySlice_Check(dimidx)) {
          Py_ssize_t start, end, length;
          if (!THPUtils_(parseSlice)(dimidx, THTensor_(size)(LIBRARY_STATE tresult, t_dim), &start, &end, &length))
            return false;
          THTensor_(narrow)(LIBRARY_STATE tresult, NULL, t_dim++, start, length);
        } else {
          PyErr_SetString(PyExc_RuntimeError, "Slicing with an unsupported type");
          return false;
        }
      }
    }
    return true;
  } catch(...) {
    THTensor_(free)(LIBRARY_STATE tresult);
    throw;
  }
}
#undef INDEX_LONG
#undef GET_PTR_1D

static PyObject * THPTensor_(getValue)(THPTensor *self, PyObject *index)
{
  HANDLE_TH_ERRORS
#ifndef THC_GENERIC_FILE
  if(THPByteTensor_IsSubclass(index)) {
    THTensor *t = THTensor_(new)(LIBRARY_STATE_NOARGS);
    THTensor_(maskedSelect)(LIBRARY_STATE t, self->cdata, ((THPByteTensor*)index)->cdata);
    return THPTensor_(newObject)(t);
#elif defined(THC_REAL_IS_FLOAT)
  if(THCPByteTensor_IsSubclass(index)) {
    THTensor *t = THTensor_(new)(LIBRARY_STATE_NOARGS);
    THTensor_(maskedSelect)(LIBRARY_STATE t, self->cdata, ((THCPByteTensor*)index)->cdata);
    return THPTensor_(newObject)(t);
#else
  if (false) {
#endif
  } else {
    THTensor *tresult; // TODO: free on error
    THStorage *sresult;
    long storage_offset;
    if (!THPTensor_(_index)(self, index, tresult, sresult, storage_offset))
      return NULL;
    if (tresult)
      return THPTensor_(newObject)(tresult);
    if (sresult)
      return THPUtils_(newReal)(THStorage_(get)(LIBRARY_STATE sresult, storage_offset));
    char err_string[512];
    snprintf (err_string, 512,
          "%s %s", "Unknown exception in THPTensor_(getValue). Index type is: ",
          index->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}

//extern PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other);
int THPTensor_(setValue)(THPTensor *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
#if !defined(THC_GENERIC_FILE) || defined(THC_REAL_IS_FLOAT)
#ifdef THC_REAL_IS_FLOAT
  if (THCPByteTensor_IsSubclass(index)) {
    THCPByteTensor *mask = (THCPByteTensor*)index;
#else
  if (THPByteTensor_IsSubclass(index)) {
    THPByteTensor *mask = (THPByteTensor*)index;
#endif
    if (THPUtils_(checkReal)(value)) {
      real v;
      if (!THPUtils_(parseReal)(value, &v))
        return -1;
      THTensor_(maskedFill)(LIBRARY_STATE self->cdata, mask->cdata, v);
    } else if (THPTensor_(IsSubclass)(value)) {
      THTensor_(maskedCopy)(LIBRARY_STATE self->cdata, mask->cdata, ((THPTensor*)value)->cdata);
    } else {
      THError("number or Tensor expected");
    }
#else
  if (false) {
#endif
  } else {
    THTensor *tresult;
    THStorage *sresult;
    long storage_offset;
    real v;
    if (!THPTensor_(_index)(self, index, tresult, sresult, storage_offset))
      return -1;

    THTensorPtr tresult_ptr = tresult;
    if (sresult) {
      if (!THPUtils_(parseReal)(value, &v))
        return -1;
      THStorage_(set)(LIBRARY_STATE sresult, storage_offset, v);
    } else if (tresult) {
      if (THPUtils_(checkReal)(value)) {
        if (!THPUtils_(parseReal)(value, &v))
          return -1;
        THTensor_(fill)(LIBRARY_STATE tresult, v);
      } else {
        // TODO: try to do this without creating a temporary object
        THPTensorPtr tmp = (THPTensor*)THPTensor_(newObject)(tresult_ptr.get());
        tresult_ptr.release();
        if (!THPModule_tensorCopy((PyObject*)tmp.get(), value))
          return -1;
      }
    } else {
      // TODO: error message
      THPUtils_setError("error");
      return -1;
    }
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyMappingMethods THPTensor_(mappingmethods) = {
  NULL,
  (binaryfunc)THPTensor_(getValue),
  (objobjargproc)THPTensor_(setValue)
};

// TODO: implement equality
PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPTensorBaseStr,          /* tp_name */
  sizeof(THPTensor),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTensor_(dealloc),       /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  &THPTensor_(mappingmethods),           /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
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
  THPTensor_(pynew),                     /* tp_new */
};

static struct PyMemberDef THPTensor_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPTensor, cdata), READONLY, NULL},
  {NULL}
};

typedef struct {
  PyObject_HEAD
} THPTensorStateless;

PyTypeObject THPTensorStatelessType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPTensorBaseStr ".stateless", /* tp_name */
  sizeof(THPTensorStateless),            /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved / tp_compare */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPTensor_stateless_(methods),         /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0,                                     /* tp_new */
  0,                                     /* tp_free */
  0,                                     /* tp_is_gc */
  0,                                     /* tp_bases */
  0,                                     /* tp_mro */
  0,                                     /* tp_cache */
  0,                                     /* tp_subclasses */
  0,                                     /* tp_weaklist */
};

bool THPTensor_(init)(PyObject *module)
{
  THPTensorType.tp_methods = THPTensor_(methods);
  THPTensorType.tp_members = THPTensor_(members);
  if (PyType_Ready(&THPTensorType) < 0)
    return false;
  THPTensorStatelessType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&THPTensorStatelessType) < 0)
    return false;

  PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
  return true;
}

#undef NUMPY_TYPE_ENUM

#endif
