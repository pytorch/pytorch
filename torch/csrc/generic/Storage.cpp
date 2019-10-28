#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#else

PyObject *THPStorageClass = nullptr;

PyObject * THPStorage_(New)(THWStorage *ptr)
{
  AT_ASSERT(ptr);
  PyTypeObject *type = (PyTypeObject *)THPStorageClass;
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THPStorage *)obj)->cdata = ptr;
  } else {
    THWStorage_(free)(LIBRARY_STATE ptr);
  }
  return obj;
}

static void THPStorage_(dealloc)(THPStorage* self)
{
  THWStorage_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static THWStorage* THPStorage_(newWithAllocator)(int64_t size, at::Allocator* allocator)
{
#if defined(THC_GENERIC_FILE)
  THPUtils_setError(THPStorageStr " does not support custom allocators");
  return nullptr;
#else
  return THWStorage_(newWithAllocator)(LIBRARY_STATE size, allocator);
#endif
}

static PyObject * THPStorage_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

  THPStoragePtr self((THPStorage *)type->tp_alloc(type, 0));
  THPUtils_assert(self, "failed to allocate a " THPStorageStr " object");
  c10::Allocator* allocator = nullptr;

  // Internally we allow constructing with a keywoard only argument cdata
  if (kwargs != nullptr) {
    PyObject *allocator_ptr = PyDict_GetItemString(kwargs, "allocator");
    if (allocator_ptr) {
      THPUtils_assert(THPUtils_checkLong(allocator_ptr), "invalid allocator");
      allocator = static_cast<c10::Allocator*>(PyLong_AsVoidPtr(allocator_ptr));
      PyDict_DelItemString(kwargs, "allocator");
    }

    Py_ssize_t num_kwargs = PyDict_Size(kwargs);
    if (num_args == 0) {
      PyObject *cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
      if (num_kwargs == 1 && cdata_ptr && THPUtils_checkLong(cdata_ptr)) {
        THWStorage *ptr = (THWStorage*)PyLong_AsVoidPtr(cdata_ptr);
        self->cdata = ptr;
        return (PyObject*)self.release();
      }
    }
    THPUtils_assert(num_kwargs == 0, THPStorageStr "(): invalid keyword arguments");
  }

  // torch.Storage()
  if (num_args == 0) {
    if (allocator) {
      self->cdata = THPStorage_(newWithAllocator)(0, allocator);
    } else {
      self->cdata = THWStorage_(new)(LIBRARY_STATE_NOARGS);
    }
    return (PyObject*)self.release();
  }

  PyObject *first_arg = PyTuple_GET_ITEM(args, 0);

  // torch.Storage(size)
  if (num_args == 1 && THPUtils_checkLong(first_arg)) {
    int64_t size = THPUtils_unpackLong(first_arg);
    if (allocator) {
      self->cdata = THPStorage_(newWithAllocator)(size, allocator);
    } else {
      self->cdata = THWStorage_(newWithSize)(LIBRARY_STATE size);
    }
    return (PyObject*)self.release();
  }

  // torch.Storage(view_source, [offset, [size]])
  if (num_args < 4 && THPStorage_(Check)(first_arg)) {
    THPUtils_setError("storage views not supported");
    return nullptr;
  }

  // torch.Storage(sequence)
  if (num_args == 1 && PySequence_Check(first_arg)) {
    Py_ssize_t length = PySequence_Length(first_arg);
    THPUtils_assert(length >= 0, "couldn't obtain the length of %s",
        THPUtils_typename(first_arg));
    self->cdata = THWStorage_(newWithSize)(LIBRARY_STATE length);
    THPObjectPtr item;
    try {
      for (Py_ssize_t i = 0; i < length; i++) {
        item = PySequence_GetItem(first_arg, i);
        scalar_t value = THPUtils_(unpackReal)(item.get());
#if !defined(THC_GENERIC_FILE)
        self->cdata->unsafe_data<scalar_t>()[i] = value;
#else
        // TODO: this might be slow - consider batched updates?
        THCStorage_(set)(LIBRARY_STATE self->cdata, i, value);
#endif
      }
    } catch (const std::exception &e) {
      THPUtils_setError("tried to construct a storage from a sequence (%s), "
          "but one of the items was of type %s instead of %s",
          THPUtils_typename(first_arg),
          THPUtils_typename(item.get()),
          THPUtils_typeTraits<scalar_t>::python_type_str);
      return nullptr;
    }
    return (PyObject*)self.release();
  }

  THPUtils_invalidArguments(args, kwargs, THPStorageStr " constructor", 6,
          "no arguments",
          "(int size)",
          "(Sequence data)",
          "(" THPStorageStr " view_source)",
          "(" THPStorageStr " view_source, int offset)",
          "(" THPStorageStr " view_source, int offset, int size)");
  return nullptr;
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPStorage_(length)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return THWStorage_(size)(LIBRARY_STATE self->cdata);
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject * THPStorage_(get)(THPStorage *self, PyObject *index)
{
  HANDLE_TH_ERRORS
  /* Integer index */
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    if (nindex < 0)
      nindex += THWStorage_(size)(LIBRARY_STATE self->cdata);
    if (nindex < 0 || nindex >= self->cdata->numel()) {
      PyErr_Format(PyExc_IndexError, "index %" PRId64 " out of range for storage of "
              "size %" PRId64, (int64_t) nindex, (int64_t) self->cdata->numel());
      return nullptr;
    }
    scalar_t value = THWStorage_(get)(LIBRARY_STATE self->cdata, nindex);
    return THPUtils_(newReal)(value);
  /* Slice index */
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, slicelength, step;
    int64_t len = THWStorage_(size)(LIBRARY_STATE self->cdata);
    if (!THPUtils_parseSlice(index, len, &start, &stop, &step, &slicelength))
      return nullptr;
    if (step != 1) {
      THPUtils_setError("Trying to slice with a step of %" PRId64 ", but only a step of "
          "1 is supported", (int64_t)step);
      return nullptr;
    }

    scalar_t *data = THWStorage_(data)(LIBRARY_STATE self->cdata);

    at::StorageImpl* old_storage = self->cdata;
    c10::raw::intrusive_ptr::incref(old_storage);
    at::Storage new_storage(c10::make_intrusive<at::StorageImpl>(
      old_storage->dtype(),
      slicelength,
      at::DataPtr(static_cast<void*>(data + start),
                  old_storage,
                  [](void* s) { c10::raw::intrusive_ptr::decref(static_cast<at::StorageImpl*>(s)); },
                  old_storage->device()),
      old_storage->allocator(),
      /* resizable */ false));

    PyObject *_ret = THPStorage_(New)(new_storage.unsafeReleaseStorageImpl());
    return _ret;
  }
  PyErr_Format(PyExc_TypeError, "can't index a " THPStorageStr " with %s",
      THPUtils_typename(index));
  return nullptr;
  END_HANDLE_TH_ERRORS
}

static int THPStorage_(set)(THPStorage *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
  if (!THPUtils_(checkReal)(value)) {
    THPUtils_setError("can only set storage content with a %s, but got "
        "%s instead", THPUtils_typeTraits<scalar_t>::python_type_str,
        THPUtils_typename(value));
    return -1;
  }

  scalar_t rvalue = THPUtils_(unpackReal)(value);
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    THWStorage_(set)(LIBRARY_STATE self->cdata, nindex, rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, slicelength, step;
    int64_t len = THWStorage_(size)(LIBRARY_STATE self->cdata);
    if (!THPUtils_parseSlice(index, len, &start, &stop, &step, &slicelength))
      return -1;
    if (step != 1) {
      THPUtils_setError("Trying to slice with a step of %" PRId64 ", but only a step of "
          "1 is supported", (int64_t)step);
      return 0;
    }
    // TODO: check the bounds only once
    // TODO: fill?
    for (;start < stop; start++)
      THWStorage_(set)(LIBRARY_STATE self->cdata, start, rvalue);
    return 0;
  }
  THPUtils_setError("can't index a " THPStorageStr " with %s",
      THPUtils_typename(index));
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
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C." THPStorageBaseStr,         /* tp_name */
  sizeof(THPStorage),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPStorage_(dealloc),      /* tp_dealloc */
  nullptr,                                     /* tp_print */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  &THPStorage_(mappingmethods),          /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  nullptr,   /* will be assigned in init */    /* tp_methods */
  nullptr,   /* will be assigned in init */    /* tp_members */
  nullptr,                                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPStorage_(pynew),                    /* tp_new */
};

static struct PyMemberDef THPStorage_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPStorage, cdata), READONLY, nullptr},
  {nullptr}
};

static PyObject * THPStorage_(device)(THPStorage* self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata->device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(dtype)(THPStorage *self, void *unused)
{
  HANDLE_TH_ERRORS
  return torch::autograd::utils::wrap(
      torch::getDtype(at::typeMetaToScalarType(self->cdata->dtype())));
  END_HANDLE_TH_ERRORS
}

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPStorage_(properties)[] = {
  {"device", (getter)THPStorage_(device), nullptr, nullptr, nullptr},
  {"dtype",  (getter)THPStorage_(dtype), nullptr, nullptr, nullptr},
  {nullptr}
};

extern THPCopyList THWStorage_(copy_functions);
THPCopyList THWStorage_(copy_functions);

void THPStorage_(initCopyMethods)()
{
  auto& h = THWStorage_(copy_functions);
  // copy from CPU types
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPByteStorageType, h, &THWStorage_(copyByte));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPCharStorageType, h, &THWStorage_(copyChar));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPShortStorageType, h, &THWStorage_(copyShort));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPIntStorageType, h, &THWStorage_(copyInt));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPLongStorageType, h, &THWStorage_(copyLong));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPHalfStorageType, h, &THWStorage_(copyHalf));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPFloatStorageType, h, &THWStorage_(copyFloat));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPDoubleStorageType, h, &THWStorage_(copyDouble));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPBoolStorageType, h, &THWStorage_(copyBool));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THPBFloat16StorageType, h, &THWStorage_(copyBFloat16));
#ifdef THC_GENERIC_FILE
  // copy from GPU types
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPByteStorageType, h, &THWStorage_(copyCudaByte));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPCharStorageType, h, &THWStorage_(copyCudaChar));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPShortStorageType, h, &THWStorage_(copyCudaShort));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPIntStorageType, h, &THWStorage_(copyCudaInt));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPLongStorageType, h, &THWStorage_(copyCudaLong));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPFloatStorageType, h, &THWStorage_(copyCudaFloat));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPDoubleStorageType, h, &THWStorage_(copyCudaDouble));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPHalfStorageType, h, &THWStorage_(copyCudaHalf));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPBoolStorageType, h, &THWStorage_(copyCudaBool));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPBFloat16StorageType, h, &THWStorage_(copyCudaBFloat16));
  // add CPU <- GPU copies to base type
  /// #define THPCpuStorage TH_CONCAT_3(THP, Real, Storage)
  #define THCpuStorage_(name) TH_CONCAT_4(TH, Real, Storage_, name)
  extern THPCopyList THCpuStorage_(copy_functions);
  auto& b = THCpuStorage_(copy_functions);
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPByteStorageType, b, &THCpuStorage_(copyCudaByte));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPCharStorageType, b, &THCpuStorage_(copyCudaChar));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPShortStorageType, b, &THCpuStorage_(copyCudaShort));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPIntStorageType, b, &THCpuStorage_(copyCudaInt));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPLongStorageType, b, &THCpuStorage_(copyCudaLong));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPFloatStorageType, b, &THCpuStorage_(copyCudaFloat));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPDoubleStorageType, b, &THCpuStorage_(copyCudaDouble));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPHalfStorageType, b, &THCpuStorage_(copyCudaHalf));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPBoolStorageType, b, &THCpuStorage_(copyCudaBool));
  THPInsertStorageCopyFunction<THPStorage, THPStorage>(&THCPBFloat16StorageType, b, &THCpuStorage_(copyCudaBFloat16));
  #undef THCpuStorage
  #undef THCpuStorage_
#endif
}

#include <torch/csrc/generic/StorageMethods.cpp>
#include <torch/csrc/generic/StorageSharing.cpp>

bool THPStorage_(init)(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THPStorage_(methods));
  THPUtils_addPyMethodDefs(methods, THPStorage_(sharingMethods));

  THPStorageType.tp_methods = methods.data();
  THPStorageType.tp_members = THPStorage_(members);
  THPStorageType.tp_getset = THPStorage_(properties);
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject *)&THPStorageType);
  THPStorage_(initCopyMethods)();
  return true;
}

void THPStorage_(postInit)(PyObject *module)
{
  THPStorageClass = PyObject_GetAttrString(module,(char*)TH_CONCAT_STRING_2(Real,Storage));
  if (!THPStorageClass) throw python_error();

  at::Backend backend = at::Backend::CPU;
#ifdef THC_GENERIC_FILE
  backend = at::Backend::CUDA;
#endif

#ifdef THQUANTIZED
  backend = at::Backend::QuantizedCPU;
#endif

  torch::registerStoragePyTypeObject((PyTypeObject*)THPStorageClass, backend, TH_CONCAT_2(at::k, Real));
}

#endif
