#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.cpp"
#else

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
  THTensor_(free)(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *cdata_arg = NULL;                 // keyword-only arg - cdata pointer value
  THLongStorage *sizes_arg = NULL;            // a storage with sizes for a new tensor
  THTensor *tensor_arg = NULL;                // a tensor to be viewed on
  PyObject *iterable_arg = NULL;              // an iterable, with new tensor contents
  std::vector<size_t> iterator_lengths;       // a queue storing lengths of iterables at each depth
  bool args_ok = true;

  if (kwargs && PyDict_Size(kwargs) == 1) {
    cdata_arg = PyDict_GetItemString(kwargs, "cdata");
    args_ok = cdata_arg != NULL;
  } else if (args && PyTuple_Size(args) == 1) {
    PyObject *arg = PyTuple_GET_ITEM(args, 0);
    if (THPTensor_(IsSubclass)(arg)) {
      tensor_arg = ((THPTensor*)arg)->cdata;
    } else if (THPLongStorage_IsSubclass(arg)) {
      sizes_arg = ((THPLongStorage*)arg)->cdata;
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
        item = PyIter_Next(iter);
        if (item == nullptr) {
          // TODO: set error
          return NULL;
        }
      }
      args_ok = iterator_lengths.size() > 0;
      // We have accumulated some errors along the way
      // Since, we did all checking and ignored only the non-important
      // ones it's safe to clear them here
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
      THTensor *ptr = (THTensor*)PyLong_AsVoidPtr(cdata_arg);
      THTensor_(retain)(ptr);
      self->cdata = ptr;
    } else if (sizes_arg) {
      self->cdata = THTensor_(newWithSize)(sizes_arg, nullptr);
    } else if (tensor_arg) {
      self->cdata = THTensor_(newWithTensor)(tensor_arg);
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
      THTensorPtr tensor = THTensor_(newWithSize)(sizes, NULL);

      real *data = tensor->storage->data;
      if (!THPUtils_(parseReal)(item, data++))
        return NULL;
      items_processed[iter_depth-1]++;

      while (!iterator_stack.empty()) {
        PyObject *iter = iterator_stack.top().get();
        // Parse items
        if (iterator_stack.size() == iter_depth) {
          while ((item = PyIter_Next(iter))) {
            if (!THPUtils_(parseReal)(item, data++))
              return NULL;
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
      self->cdata = THTensor_(new)();
    }

    if (self->cdata == NULL)
      return NULL;
  }
  return (PyObject *)self.release();
  END_HANDLE_TH_ERRORS
}

#define INDEX_LONG(DIM, IDX_VARIABLE, TENSOR_VARIABLE, CASE_1D, CASE_MD) \
  long idx;								\
  THPUtils_getLong(IDX_VARIABLE, &idx);					\
  long dimsize = THTensor_(size)(TENSOR_VARIABLE, DIM);			\
  idx = (idx < 0) ? dimsize + idx : idx;				\
									\
  THArgCheck(dimsize > 0, 1, "empty tensor");				\
  THArgCheck(idx >= 0 && idx < dimsize, 2, "out of range");		\
									\
  if(THTensor_(nDimension)(TENSOR_VARIABLE) == 1) {			\
    CASE_1D;								\
  } else {								\
    CASE_MD;								\
  }

#define GET_PTR_1D(t, idx)					\
  t->storage->data + t->storageOffset + t->stride[0] * idx;

static bool THPTensor_(_index)(THPTensor *self, PyObject *index,
    THTensor * &tresult, real * &rresult)
{
  tresult = NULL;
  rresult = NULL;
  try {
    // Indexing with an integer
    if(PyLong_Check(index) || PyInt_Check(index)) {
      THTensor *self_t = self->cdata;
      INDEX_LONG(0, index, self_t,
        // 1D tensor
        rresult = GET_PTR_1D(self_t, idx),
        // >1D tensor
        tresult = THTensor_(newWithTensor)(self_t);
        THTensor_(select)(tresult, NULL, 0, idx)
      )
    // Indexing with a single element tuple
    } else if (PyTuple_Check(index) &&
	       PyTuple_Size(index) == 1 &&
	       (PyLong_Check(PyTuple_GET_ITEM(index, 0))
		|| PyInt_Check(PyTuple_GET_ITEM(index, 0)))) {
      PyObject *index_obj = PyTuple_GET_ITEM(index, 0);
      tresult = THTensor_(newWithTensor)(self->cdata);
      INDEX_LONG(0, index_obj, tresult,
          THTensor_(narrow)(tresult, NULL, 0, idx, 1),
          THTensor_(narrow)(tresult, NULL, 0, idx, 1)
        )
    // Indexing with a slice
    } else if (PySlice_Check(index)) {
      tresult = THTensor_(newWithTensor)(self->cdata);
      Py_ssize_t start, end, length;
      if (!THPUtils_(parseSlice)(index, THTensor_(size)(tresult, 0), &start, &end, &length))
        return false;
      THTensor_(narrow)(tresult, NULL, 0, start, length);
    // Indexing multiple dimensions
    } else if(PyTuple_Check(index)) {
      THArgCheck(PyTuple_Size(index) <= THTensor_(nDimension)(self->cdata), 2,
              "Indexing too many dimensions");
      tresult = THTensor_(newWithTensor)(self->cdata);
      int t_dim = 0;

      for(int dim = 0; dim < PyTuple_Size(index); dim++) {
        PyObject *dimidx = PyTuple_GET_ITEM(index, dim);
        if(PyLong_Check(dimidx) || PyInt_Check(dimidx)) {
          INDEX_LONG(t_dim, dimidx, tresult,
              // 1D tensor
              rresult = GET_PTR_1D(tresult, idx);
              THTensor_(free)(tresult);
              tresult = NULL;
              return true,
              // >1D tensor
              THTensor_(select)(tresult, NULL, t_dim, idx)
            )
        } else if (PyTuple_Check(dimidx)) {
          if (PyTuple_Size(dimidx) != 1
	      || !(PyLong_Check(PyTuple_GET_ITEM(dimidx, 0))
		   || PyInt_Check(PyTuple_GET_ITEM(dimidx, 0)))) {
            PyErr_SetString(PyExc_RuntimeError, "Expected a single integer");
            return false;
          }
          PyObject *index_obj = PyTuple_GET_ITEM(dimidx, 0);
          INDEX_LONG(t_dim, index_obj, tresult,
              THTensor_(narrow)(tresult, NULL, t_dim++, idx, 1),
              THTensor_(narrow)(tresult, NULL, t_dim++, idx, 1)
            )
        } else if (PySlice_Check(dimidx)) {
          Py_ssize_t start, end, length;
          if (!THPUtils_(parseSlice)(dimidx, THTensor_(size)(tresult, t_dim), &start, &end, &length))
            return false;
          THTensor_(narrow)(tresult, NULL, t_dim++, start, length);
        } else {
          PyErr_SetString(PyExc_RuntimeError, "Slicing with an unsupported type");
          return false;
        }
      }
    }
    return true;
  } catch(...) {
    THTensor_(free)(tresult);
    throw;
  }
}
#undef INDEX_LONG
#undef GET_PTR_1D

static PyObject * THPTensor_(getValue)(THPTensor *self, PyObject *index)
{
  HANDLE_TH_ERRORS
  if(THPByteTensor_IsSubclass(index)) {
    THTensor *t = THTensor_(new)();
    THTensor_(maskedSelect)(t, self->cdata, ((THPByteTensor*)index)->cdata);
    return THPTensor_(newObject)(t);
  } else {
    THTensor *tresult;
    real *rresult;
    if (!THPTensor_(_index)(self, index, tresult, rresult))
      return NULL;
    if (tresult)
      return THPTensor_(newObject)(tresult);
    if (rresult)
      return THPUtils_(newReal)(*rresult);
    char err_string[512];
    snprintf (err_string, 512,
	      "%s %s", "Unknown exception in THPTensor_(get). Index type is: ",
	      index->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return NULL;
  }
  END_HANDLE_TH_ERRORS
}

int THPTensor_(setValue)(THPTensor *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
  if (THPByteTensor_IsSubclass(index)) {
    THPByteTensor *byte_index = (THPByteTensor*)index;
    if (THPUtils_(checkReal)(value)) {
      real v;
      if (!THPUtils_(parseReal)(value, &v))
        return -1;
      THTensor_(maskedFill)(self->cdata, byte_index->cdata, v);
    } else if (THPTensor_(IsSubclass)(index)) {
      THTensor_(maskedCopy)(self->cdata, byte_index->cdata, ((THPTensor*)value)->cdata);
    }
    THError("number or Tensor expected");
  } else {
    THTensor *tresult;
    real *rresult;
    real v;
    if (!THPTensor_(_index)(self, index, tresult, rresult))
      return -1;

    if (rresult) {
      if (!THPUtils_(parseReal)(value, &v))
        return -1;
      *rresult = v;
    } else {
      try {
        if (THPUtils_(checkReal)(value)) {
          if (!THPUtils_(parseReal)(value, &v))
            return -1;
          THTensor_(fill)(tresult, v);
        } else {
          if (THPByteTensor_IsSubclass(value))
            THTensor_(copyByte)(tresult, ((THPByteTensor*)value)->cdata);
          else if (THPCharTensor_IsSubclass(value))
            THTensor_(copyChar)(tresult, ((THPCharTensor*)value)->cdata);
          else if (THPShortTensor_IsSubclass(value))
            THTensor_(copyShort)(tresult, ((THPShortTensor*)value)->cdata);
          else if (THPIntTensor_IsSubclass(value))
            THTensor_(copyInt)(tresult, ((THPIntTensor*)value)->cdata);
          else if (THPLongTensor_IsSubclass(value))
            THTensor_(copyLong)(tresult, ((THPLongTensor*)value)->cdata);
          else if (THPFloatTensor_IsSubclass(value))
            THTensor_(copyFloat)(tresult, ((THPFloatTensor*)value)->cdata);
          else if (THPDoubleTensor_IsSubclass(value))
            THTensor_(copyDouble)(tresult, ((THPDoubleTensor*)value)->cdata);
          else {
            PyErr_SetString(PyExc_RuntimeError, "Expected a number or tensor");
            return -1;
          }
        }
      } catch(...) {
        THTensor_(free)(tresult);
        throw;
      }
      THTensor_(free)(tresult);
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
  "torch.C." THPTensorBaseStr,           /* tp_name */
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

#include "TensorMethods.cpp"

typedef struct {
  PyObject_HEAD
} THPTensorStateless;

PyTypeObject THPTensorStatelessType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch.C." THPTensorBaseStr ".stateless", /* tp_name */
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
  THPTensorStatelessMethods,             /* tp_methods */
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

#endif
