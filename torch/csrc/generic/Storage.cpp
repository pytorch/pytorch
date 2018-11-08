#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.cpp"
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
#if defined(THC_GENERIC_FILE) || defined(THD_GENERIC_FILE)
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
  THAllocator* allocator = nullptr;

  // Internally we allow constructing with a keywoard only argument cdata
  if (kwargs != nullptr) {
    PyObject *allocator_ptr = PyDict_GetItemString(kwargs, "allocator");
    if (allocator_ptr) {
      THPUtils_assert(THPUtils_checkLong(allocator_ptr), "invalid allocator");
      allocator = (THAllocator*) PyLong_AsVoidPtr(allocator_ptr);
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
#ifdef THD_GENERIC_FILE
    THPUtils_setError("distributed storages don't support construction from a sequence");
#else
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
#endif
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
      return NULL;
    if (step != 1) {
      THPUtils_setError("Trying to slice with a step of %" PRId64 ", but only a step of "
          "1 is supported", (int64_t)step);
      return NULL;
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
  nullptr,                                  /* tp_doc */
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
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPStorage, cdata), READONLY, nullptr},
  {nullptr}
};

extern THPCopyList THWStorage_(copy_functions);
THPCopyList THWStorage_(copy_functions);

void THPStorage_(initCopyMethods)()
{
#ifndef THD_GENERIC_FILE
  auto& h = THWStorage_(copy_functions);
  // copy from CPU types
  THPInsertStorageCopyFunction<THPStorage, THPByteStorage>(&THPByteStorageType, h, &THWStorage_(copyByte));
  THPInsertStorageCopyFunction<THPStorage, THPCharStorage>(&THPCharStorageType, h, &THWStorage_(copyChar));
  THPInsertStorageCopyFunction<THPStorage, THPShortStorage>(&THPShortStorageType, h, &THWStorage_(copyShort));
  THPInsertStorageCopyFunction<THPStorage, THPIntStorage>(&THPIntStorageType, h, &THWStorage_(copyInt));
  THPInsertStorageCopyFunction<THPStorage, THPLongStorage>(&THPLongStorageType, h, &THWStorage_(copyLong));
  THPInsertStorageCopyFunction<THPStorage, THPHalfStorage>(&THPHalfStorageType, h, &THWStorage_(copyHalf));
  THPInsertStorageCopyFunction<THPStorage, THPFloatStorage>(&THPFloatStorageType, h, &THWStorage_(copyFloat));
  THPInsertStorageCopyFunction<THPStorage, THPDoubleStorage>(&THPDoubleStorageType, h, &THWStorage_(copyDouble));
#ifdef THC_GENERIC_FILE
  // copy from GPU types
  THPInsertStorageCopyFunction<THPStorage, THCPByteStorage>(&THCPByteStorageType, h, &THWStorage_(copyCudaByte));
  THPInsertStorageCopyFunction<THPStorage, THCPCharStorage>(&THCPCharStorageType, h, &THWStorage_(copyCudaChar));
  THPInsertStorageCopyFunction<THPStorage, THCPShortStorage>(&THCPShortStorageType, h, &THWStorage_(copyCudaShort));
  THPInsertStorageCopyFunction<THPStorage, THCPIntStorage>(&THCPIntStorageType, h, &THWStorage_(copyCudaInt));
  THPInsertStorageCopyFunction<THPStorage, THCPLongStorage>(&THCPLongStorageType, h, &THWStorage_(copyCudaLong));
  THPInsertStorageCopyFunction<THPStorage, THCPFloatStorage>(&THCPFloatStorageType, h, &THWStorage_(copyCudaFloat));
  THPInsertStorageCopyFunction<THPStorage, THCPDoubleStorage>(&THCPDoubleStorageType, h, &THWStorage_(copyCudaDouble));
  THPInsertStorageCopyFunction<THPStorage, THCPHalfStorage>(&THCPHalfStorageType, h, &THWStorage_(copyCudaHalf));
  // add CPU <- GPU copies to base type
  #define THPCpuStorage TH_CONCAT_3(THP, Real, Storage)
  #define THCpuStorage_(name) TH_CONCAT_4(TH, Real, Storage_, name)
  extern THPCopyList THCpuStorage_(copy_functions);
  auto& b = THCpuStorage_(copy_functions);
  THPInsertStorageCopyFunction<THPCpuStorage, THCPByteStorage>(&THCPByteStorageType, b, &THCpuStorage_(copyCudaByte));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPCharStorage>(&THCPCharStorageType, b, &THCpuStorage_(copyCudaChar));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPShortStorage>(&THCPShortStorageType, b, &THCpuStorage_(copyCudaShort));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPIntStorage>(&THCPIntStorageType, b, &THCpuStorage_(copyCudaInt));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPLongStorage>(&THCPLongStorageType, b, &THCpuStorage_(copyCudaLong));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPFloatStorage>(&THCPFloatStorageType, b, &THCpuStorage_(copyCudaFloat));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPDoubleStorage>(&THCPDoubleStorageType, b, &THCpuStorage_(copyCudaDouble));
  THPInsertStorageCopyFunction<THPCpuStorage, THCPHalfStorage>(&THCPHalfStorageType, b, &THCpuStorage_(copyCudaHalf));
  #undef THCpuStorage
  #undef THCpuStorage_
#endif
#endif // !defined(THD_GENERIC_FILE)
}

#include "StorageMethods.cpp"
#ifndef THD_GENERIC_FILE
#include "StorageSharing.cpp"
#endif

bool THPStorage_(init)(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THPStorage_(methods));
#ifndef THD_GENERIC_FILE
  THPUtils_addPyMethodDefs(methods, THPStorage_(sharingMethods));
#endif

  THPStorageType.tp_methods = methods.data();
  THPStorageType.tp_members = THPStorage_(members);
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

  bool is_cuda = false;
#ifdef THC_GENERIC_FILE
  is_cuda = true;
#endif
  const char *type_name = TH_CONCAT_STRING_2(Real,);
  torch::registerStoragePyTypeObject((PyTypeObject*)THPStorageClass, type_name, is_cuda, false);
}

#endif
