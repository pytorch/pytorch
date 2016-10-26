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

PyObject *THPTensorClass = NULL;

PyObject * THPTensor_(New)(THTensor *ptr)
{
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = NULL;
  if (!args) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create a new storage object - "
        "failed to allocate argument tuple");
    return NULL;
  }
  if (ptr) {
    kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));
    if (!kwargs) {
      PyErr_SetString(PyExc_RuntimeError, "Could not create a new storage object - "
          "failed to allocate keyword argument dictionary");
      Py_DECREF(args);
      return NULL;
    }
  }
  PyObject *result = PyObject_Call(THPTensorClass, args, kwargs);
  Py_DECREF(args);
  Py_XDECREF(kwargs);
  return result;
}

#include "TensorMethods.cpp"

static void THPTensor_(dealloc)(THPTensor* self)
{
  THTensor_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static std::string THPTensor_(indicesToString)(std::vector<size_t> &indices,
    size_t depth)
{
  std::string index = "(";
  for (size_t i = 0; i <= depth; ++i) {
    index += std::to_string(indices[i]);
    index += ", ";
  }
  index.erase(index.length()-2);  // Remove trailing ", "
  index += ")";
  return index;
}

static void THPTensor_(setInconsistentDepthError)(std::vector<size_t> &sizes,
    std::vector<size_t> &indices, size_t depth, size_t length)
{
  std::string error = "inconsistent sequence length at index ";
  error += THPTensor_(indicesToString)(indices, depth);
  error += " - expected ";
  error += std::to_string(sizes[depth]);
  error += " but got ";
  error += std::to_string(length);
  THPUtils_setError(error.c_str());
}

#ifdef NUMPY_TYPE_ENUM
THTensor* THPTensor_(fromNumpy)(PyObject *numpy_array) {
  PyArrayObject *array = (PyArrayObject*)numpy_array;
  THStoragePtr storage = THStorage_(newWithDataAndAllocator)(
      (real*)PyArray_DATA(array),
      PyArray_NBYTES(array) / sizeof(real),
      &THNumpyArrayAllocator,
      new NumpyArrayAllocator(numpy_array));

  // Numpy and Torch disagree on empty tensors. In Torch, an empty
  // tensor is a tensor with zero dimensions. In Numpy, an empty tensor
  // keeps its shape, but has 0 as the size of one of the dimensions.
  // So we'll convert all Numpy tensors of 0 elements to empty Torch tensors.
  if (PyArray_SIZE(array) != 0) {
    auto ndim = PyArray_NDIM(array);
    THLongStoragePtr sizes = THLongStorage_newWithSize(ndim);
    long *sizes_data = sizes->data;
    for (int i = 0; i < ndim; ++i) {
      sizes_data[i] = PyArray_DIM(array, i);
    }

    THLongStoragePtr strides = THLongStorage_newWithSize(ndim);
    long *strides_data = strides->data;
    for (int i = 0; i < ndim; ++i) {
      strides_data[i] = PyArray_STRIDE(array, i) / sizeof(real);   // numpy uses bytes, torch uses elements
    }

    THTensor *result = THTensor_(newWithStorage)(storage, 0, sizes, strides);
    return result;
  } else {
    THTensor *result = THTensor_(newWithStorage)(storage, 0, NULL, NULL);
    return result;
  }
}
#endif

static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

  THPTensorPtr self = (THPTensor *)type->tp_alloc(type, 0);
  THPUtils_assert(self, "failed to allocate a " THPTensorStr " object");
  self->cdata = NULL;
#ifdef THC_GENERIC_FILE
  THCPAutoGPU gpu_guard;
#endif

  // Internally we allow constructing with a keywoard only argument cdata
  if (kwargs != NULL) {
    Py_ssize_t num_kwargs = PyDict_Size(kwargs);
#ifdef THC_GENERIC_FILE
    PyObject *device_id = PyDict_GetItemString(kwargs, "device");
    if (device_id) {
      THPUtils_assert(THPUtils_checkLong(device_id), "device argument "
          " has to be an int, but got %s", THPUtils_typename(device_id));
      gpu_guard.setDevice(THPUtils_unpackLong(device_id));
      // simulate pop() and pretend this key was never there
      num_kwargs--;
    }
#endif
    if (num_args == 0) {
      PyObject *cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
      if (num_kwargs == 1 && cdata_ptr && THPUtils_checkLong(cdata_ptr)) {
        THTensor *ptr = (THTensor*)PyLong_AsVoidPtr(cdata_ptr);
        self->cdata = ptr;
        return (PyObject*)self.release();
      }
    }
    // This is an internal option, so we don't want to advertise it.
#ifdef THC_GENERIC_FILE
    THPUtils_assert(num_kwargs == 0, THPTensorStr " constructor only "
        "accepts a 'device' keyword argument")
#else
    THPUtils_assert(num_kwargs == 0, THPTensorStr " constructor doesn't "
        "accept any keyword arguments");
#endif
  }

  // torch.Tensor()
  if (num_args == 0) {
    self->cdata = THTensor_(new)(LIBRARY_STATE_NOARGS);
    return (PyObject*)self.release();
  }

  PyObject *first_arg = PyTuple_GET_ITEM(args, 0);

  // torch.Tensor(torch.Tensor tensor)
  if (num_args == 1 && THPTensor_(Check)(first_arg)) {
    THTensor *tensor = ((THPTensor*)first_arg)->cdata;
    self->cdata = THTensor_(newWithTensor)(LIBRARY_STATE tensor);
    return (PyObject*)self.release();
  }

  // torch.Tensor(torch.Size sizes)
  if (num_args == 1 && THPSize_Check(first_arg)) {
    THLongStoragePtr sizes = THPUtils_unpackSize(first_arg);
    self->cdata = THTensor_(newWithSize)(LIBRARY_STATE sizes.get(), nullptr);
    return (PyObject *)self.release();
  }

  // TODO: implement storageOffset, sizes and strides
  // torch.Tensor(torch.Storage data)
  if (num_args == 1 && THPStorage_(Check)(first_arg)) {
    THStorage *storage = ((THPStorage*)first_arg)->cdata;
    self->cdata = THTensor_(newWithStorage1d)(LIBRARY_STATE storage, 0, storage->size, -1);
    return (PyObject *)self.release();
  }

#ifdef NUMPY_TYPE_ENUM
  // torch.Tensor(np.ndarray array)
  if (num_args == 1 && PyArray_Check(first_arg) &&
      PyArray_TYPE((PyArrayObject*)first_arg) == NUMPY_TYPE_ENUM) {
    THPObjectPtr numpy_array =
      PyArray_FromArray((PyArrayObject*)first_arg, nullptr, NPY_ARRAY_BEHAVED);
    self->cdata = THPTensor_(fromNumpy)(numpy_array.get());
    return (PyObject*)self.release();
  }
#endif

  // torch.Tensor(Sequence data)
  if (num_args == 1 && PySequence_Check(first_arg)) {
    Py_ssize_t length = PySequence_Length(first_arg);
    THPUtils_assert(length >= 0, "couldn't obtain the length of %s",
        THPUtils_typename(first_arg));
    if (length == 0) {
      self->cdata = THTensor_(new)(LIBRARY_STATE_NOARGS);
      return (PyObject*)self.release();
    }

    Py_INCREF(first_arg);
    THPObjectPtr item = first_arg;
    std::vector<size_t> sizes;
    while ((length = PySequence_Length(item)) >= 0) {
      sizes.push_back(length);
      // TODO: check for string in this case
      THPUtils_assert(sizes.size() < 1000000, "already counted a million "
          "dimensions in a given sequence. Most likely your items are also "
          "sequences and there's no way to infer how many dimension should "
          "the tensor have");
      THPUtils_assert(length > 0, "given sequence has an invalid size of "
          "dimension %ld: %ld", (long)sizes.size(), (long)length);
      item = PySequence_GetItem(item, 0);
      if (!item)
        return NULL;
    }
    // Last length check has set an error flag, so we need to clear it.
    PyErr_Clear();

    THLongStoragePtr sizes_storage = THLongStorage_newWithSize(sizes.size());
    long *sizes_data = sizes_storage->data;
    for (auto size: sizes)
      *sizes_data++ = size;
    THTensorPtr tensor = THTensor_(newWithSize)(LIBRARY_STATE sizes_storage, NULL);

    int ndims = sizes.size();
    std::vector<size_t> indices(ndims);
    std::vector<THPObjectPtr> sequences(ndims);
    Py_INCREF(first_arg);
    item = first_arg;
    for (size_t i = 0; i < sequences.size(); i++) {
      PyObject *item_ptr = item.get();
      sequences[i] = std::move(item);
      if (i < sequences.size()-1) {
        item = PySequence_ITEM(item_ptr, 0);
        if (!item)
          return NULL;
      }
    }

    // half tensors don't have CPU counterparts so we have to buffer them as
    // floats while loading
#ifndef THC_REAL_IS_HALF
#define load_real real
#define UNPACK_REAL(item) THPUtils_(unpackReal)(item)
#else
#define load_real float
#define UNPACK_REAL(item) THPFloatUtils_unpackReal(item)
#endif
#ifndef THC_GENERIC_FILE
    real *data = tensor->storage->data;
#else
    size_t numel = THTensor_(numel)(LIBRARY_STATE tensor);
    std::unique_ptr<load_real> data_guard(new load_real[numel]);
    load_real *data = data_guard.get();
#endif
    THPObjectPtr final_sequence;
    while (true) {
      final_sequence = std::move(sequences[ndims-1]);
      try {
        // We're taking a fast-track over the last dimension
        for (size_t i = 0; i < sizes[ndims-1]; i++) {
          indices[ndims-1] = i;
          item = PySequence_ITEM(final_sequence, i);
          // We've checked the length earlier, so it must have been an error
          if (!item)
            return NULL;
          *data++ = UNPACK_REAL(item);
        }
      } catch(std::runtime_error &e) {
        std::string index = THPTensor_(indicesToString)(indices, ndims-1);
        THPUtils_setError("tried to construct a tensor from a %s%s sequence, "
            "but found an item of type %s at index %s",
            (ndims > 1 ? "nested " : ""),
            THPUtils_typeTraits<real>::python_type_str,
            THPUtils_typename(item.get()),
            index.c_str());
        return NULL;
      }
#ifdef THC_GENERIC_FILE
#ifdef THC_REAL_IS_HALF
      THFloatStorage *cpu_storage = THFloatStorage_newWithData(data_guard.get(), numel);
      cpu_storage->flag &= ~TH_STORAGE_FREEMEM;
      THCudaHalfStorage_copyFloat(LIBRARY_STATE tensor->storage, cpu_storage);
      THFloatStorage_free(cpu_storage);
#else
      THHostStorage *cpu_storage = THHostStorage_(newWithData)(data_guard.get(), numel);
      cpu_storage->flag &= ~TH_STORAGE_FREEMEM;
      THCStorage_(copyCPU)(LIBRARY_STATE tensor->storage, cpu_storage);
      THHostStorage_(free)(cpu_storage);
#endif
#endif
#undef UNPACK_REAL
#undef load_real

      // Update the counters
      int dim = ndims-2;
      size_t last_updated_dim = dim;
      while (dim >= 0) {
        last_updated_dim = dim;
        if (++indices[dim] == sizes[dim])
          indices[dim--] = 0;
        else
          break;
      }
      // Check if we've just made a full cycle
      if ((last_updated_dim == 0 && indices[0] == 0) || ndims == 1)
        break;
      // Update sequences
      for (int i = last_updated_dim+1; i < ndims; i++) {
        sequences[i] = PySequence_ITEM(sequences[i-1], indices[i-1]);
        if (!sequences[i]) {
          THPTensor_(setInconsistentDepthError)(sizes, indices, i, indices[i]);
          return NULL;
        }
        if (!PySequence_Check(sequences[i])) {
          std::string index_str = THPTensor_(indicesToString)(indices, i);
          THPUtils_setError("an item of time %s at index %s doesn't implement "
              "a sequence protocol");
          return NULL;
        }
        Py_ssize_t length = PySequence_Length(sequences[i]);
        if (length < 0) {
          std::string index_str = THPTensor_(indicesToString)(indices, i);
          THPUtils_setError("could not obtain a length of %s at index %s",
              THPUtils_typename(sequences[i].get()), index_str.c_str());
          return NULL;
        }
        if ((size_t)length != sizes[i]) {
          THPTensor_(setInconsistentDepthError)(sizes, indices, i, length);
          return NULL;
        }
      }
    }
    self->cdata = tensor.release();
    return (PyObject *)self.release();
  }

  // torch.Tensor(int ...)
  THLongStoragePtr sizes;
  if (THPUtils_tryUnpackLongVarArgs(args, 0, sizes)) {
    self->cdata = THTensor_(newWithSize)(LIBRARY_STATE sizes.get(), nullptr);
    return (PyObject *)self.release();
  }

  THPUtils_invalidArguments(args, THPTensorStr " constructor", 6,
          "no arguments",
          "(int ...)",
          "(" THPTensorStr " viewed_tensor)",
          "(torch.Size size)",
          "(" THPStorageStr " data)",
          "(Sequence data)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

#define INDEX_LONG(DIM, IDX_VARIABLE, TENSOR_VARIABLE, CASE_1D, CASE_MD)       \
  long idx = THPUtils_unpackLong(IDX_VARIABLE);                                \
  long dimsize = THTensor_(size)(LIBRARY_STATE TENSOR_VARIABLE, DIM);          \
  idx = (idx < 0) ? dimsize + idx : idx;                                       \
                                                                               \
  THPUtils_assertRet(false, dimsize > 0, "indexing an empty tensor");          \
  THPUtils_assertRet(false, idx >= 0 && idx < dimsize, "index %ld is out of range for " \
      "dimension %ld (of size %ld)", idx, DIM, dimsize);                       \
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
      return true;
    // Indexing with a slice
    } else if (PySlice_Check(index)) {
      tresult = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
      Py_ssize_t start, end, length;
      if (!THPUtils_parseSlice(index, THTensor_(size)(LIBRARY_STATE tresult, 0), &start, &end, &length))
        return false;
      THTensor_(narrow)(LIBRARY_STATE tresult, NULL, 0, start, length);
      return true;
    // Indexing multiple dimensions
    } else if(PyTuple_Check(index)) {
      long num_index_dim = (long)PyTuple_Size(index);
      long num_effective_index = num_index_dim;
      long num_tensor_dim = THTensor_(nDimension)(LIBRARY_STATE self->cdata);
      long ellipsis_idx = num_tensor_dim + 1;
      for (int i = 0; i < num_index_dim; i++) {
        if (PyTuple_GET_ITEM(index, i) == Py_Ellipsis) {
          ellipsis_idx = i;
          num_effective_index--;
          break;
        }
      }
      THPUtils_assertRet(false, num_effective_index <= num_tensor_dim,
          "trying to index %ld dimensions of a %ld dimensional tensor",
          num_effective_index, num_tensor_dim);

      tresult = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
      int t_dim = 0;
      for(int dim = 0; dim < num_index_dim; dim++) {
        if (dim == ellipsis_idx) {
          t_dim = tresult->nDimension - (num_index_dim - dim - 1);
          continue;
        }
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
        } else if (PySlice_Check(dimidx)) {
          Py_ssize_t start, end, length;
          if (!THPUtils_parseSlice(dimidx, THTensor_(size)(LIBRARY_STATE tresult, t_dim), &start, &end, &length))
            return false;
          THTensor_(narrow)(LIBRARY_STATE tresult, NULL, t_dim++, start, length);
        } else {
          THTensor_(free)(LIBRARY_STATE tresult);
          goto invalid_index_type;
        }
      }
      return true;
    }
  } catch(...) {
    if (tresult) {
      THTensor_(free)(LIBRARY_STATE tresult);
      tresult = NULL;
    }
    throw;
  }

invalid_index_type:
  THPUtils_setError("indexing a tensor with an object of type %s. The only "
      "supported types are integers, slices and "
#ifndef THC_GENERIC_FILE
      "torch.ByteTensor.",
#else
      "torch.cuda.ByteTensor.",
#endif
    THPUtils_typename(index));
  return false;
}
#undef INDEX_LONG
#undef GET_OFFSET

static PyObject * THPTensor_(getValue)(THPTensor *self, PyObject *index)
{
  HANDLE_TH_ERRORS
#ifndef THC_GENERIC_FILE
  if(THPByteTensor_Check(index)) {
    THTensor *t = THTensor_(new)(LIBRARY_STATE_NOARGS);
    THTensor_(maskedSelect)(LIBRARY_STATE t, self->cdata, ((THPByteTensor*)index)->cdata);
    return THPTensor_(New)(t);
  }
#else
  if(THCPByteTensor_Check(index)) {
    THTensor *t = THTensor_(new)(LIBRARY_STATE_NOARGS);
    THTensor_(maskedSelect)(LIBRARY_STATE t, self->cdata, ((THCPByteTensor*)index)->cdata);
    return THPTensor_(New)(t);
  }
#endif

  THTensor *tresult;
  THStorage *sresult;
  long storage_offset;
  if (!THPTensor_(_index)(self, index, tresult, sresult, storage_offset))
    return NULL;
  try {
    if (tresult)
      return THPTensor_(New)(tresult);
    if (sresult)
      return THPUtils_(newReal)(THStorage_(get)(LIBRARY_STATE sresult, storage_offset));
  } catch (...) {
    if (tresult) {
      THTensor_(free)(LIBRARY_STATE tresult);
      tresult = NULL;
    }
    throw;
  }
  THPUtils_setError("An unknown error has occured when indexing a tensor "
      "in THPTensor_(getValue). Please report this in a github issue at: "
      "https://github.com/pytorch/pytorch");
  return NULL;
  END_HANDLE_TH_ERRORS
}

int THPTensor_(setValue)(THPTensor *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
#ifndef THC_GENERIC_FILE
  if (THPByteTensor_Check(index)) {
    THPByteTensor *mask = (THPByteTensor*)index;
#else
  if (THCPByteTensor_Check(index)) {
    THCPByteTensor *mask = (THCPByteTensor*)index;
#endif
    if (THPUtils_(checkReal)(value)) {
      real v = THPUtils_(unpackReal)(value);
      THTensor_(maskedFill)(LIBRARY_STATE self->cdata, mask->cdata, v);
    } else if (THPTensor_(Check)(value)) {
      THTensor_(maskedCopy)(LIBRARY_STATE self->cdata, mask->cdata, ((THPTensor*)value)->cdata);
    } else {
      THPUtils_setError("can't assign %s to a " THPTensorStr " using a mask "
          "(only " THPTensorStr " or %s are supported)",
          THPUtils_typename(value), THPUtils_typeTraits<real>::python_type_str);
      // TODO
    }
    return 0;
  }

  THTensor *tresult;
  THStorage *sresult;
  long storage_offset;
  if (!THPTensor_(_index)(self, index, tresult, sresult, storage_offset))
    return -1;

  THTensorPtr tresult_ptr = tresult;
  if (sresult) {
    if (!THPUtils_(checkReal)(value)) {
      THPUtils_setError("can't assign a %s to a scalar value of type %s",
          THPUtils_typename(value), THPUtils_typeTraits<real>::python_type_str);
      return -1;
    }
    THStorage_(set)(LIBRARY_STATE sresult, storage_offset, THPUtils_(unpackReal)(value));
    return 0;
  } else if (tresult) {
    if (THPUtils_(checkReal)(value)) {
      THTensor_(fill)(LIBRARY_STATE tresult, THPUtils_(unpackReal)(value));
    } else {
      // TODO: try to do this without creating a temporary object
      THPTensorPtr tmp = (THPTensor*)THPTensor_(New)(tresult_ptr.get());
      if (!tmp)
        return -1;
      tresult_ptr.release();
      if (!THPModule_tensorCopy((PyObject*)tmp.get(), value))
        return -1;
    }
    return 0;
  }
  THPUtils_setError("An unknown error has occured when indexing a tensor "
      "in THPTensor_(setValue). Please report this in a github issue at: "
      "https://github.com/pytorch/pytorch");
  return -1;
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
#ifndef THC_GENERIC_FILE
  THVector_(vectorDispatchInit)();
#endif

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
