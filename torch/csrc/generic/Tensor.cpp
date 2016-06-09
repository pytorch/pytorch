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
  THPLongStorage *storage_obj = NULL;
  long long sizes[] = {-1, -1, -1, -1};
  // Check if it's a long storage
  if (PyTuple_Size(args) == 1) {
    PyObject *arg = PyTuple_GetItem(args, 0);
    if (THPLongStorage_IsSubclass(arg)) {
      storage_obj = (THPLongStorage*)arg;
    }
  }
  const char *keywords[] = {"dim0", "dim1", "dim2", "dim3", "cdata", NULL};
  THTensor *cdata_ptr = NULL;
  // If not, try to parse integers
#define ERRMSG ";Expected torch.LongStorage or up to 4 integers as arguments"
  // TODO: check that cdata_ptr is a keyword arg
  if (!storage_obj &&
      !PyArg_ParseTupleAndKeywords(args, kwargs, "|LLLLL" ERRMSG, (char**)keywords,
          &sizes[0], &sizes[1], &sizes[2], &sizes[3], &cdata_ptr))
#undef ERRMSG
    return NULL;

  THPTensor *self = (THPTensor *)type->tp_alloc(type, 0);
  if (self != NULL) {
    if (storage_obj)
        self->cdata = THTensor_(newWithSize)(storage_obj->cdata, NULL);
    else if (cdata_ptr)
        self->cdata = cdata_ptr;
    else
        self->cdata = THTensor_(newWithSize4d)(sizes[0], sizes[1], sizes[2], sizes[3]);
    if (self->cdata == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject *)self;
  // TODO: cleanup on error
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

static PyObject * THPTensor_(get)(THPTensor *self, PyObject *index)
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

int THPTensor_(set)(THPTensor *self, PyObject *index, PyObject *value)
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
  (binaryfunc)THPTensor_(get),
  (objobjargproc)THPTensor_(set)
};

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
