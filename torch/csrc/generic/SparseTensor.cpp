PyObject *THSPTensorClass = NULL;

static void THSPTensor_(initStorage)(THSTensor* tensor)
{
  // Ensure that PyTorch's "storage is not NULL" invariant is upheld.
  // See Note [Storage is not NULL]
  if (!tensor->indices->storage) {
#ifdef THC_GENERIC_FILE
    tensor->indices->storage = THCudaLongStorage_new(LIBRARY_STATE_NOARGS);
#else
    tensor->indices->storage = THLongStorage_new(LIBRARY_STATE_NOARGS);
#endif
  }
  if (!tensor->values->storage) {
    tensor->values->storage = THStorage_(new)(LIBRARY_STATE_NOARGS);
  }
}

PyObject * THSPTensor_(NewEmpty)()
{
  auto r = THSTensor_(new)(LIBRARY_STATE_NOARGS);
  THSPTensor_(initStorage)(r);
  return THSPTensor_(New)(r);
}

PyObject * THSPTensor_(New)(THSTensor *tensor)
{
  THSTensorPtr ptr(tensor);
  PyTypeObject *type = (PyTypeObject *)THSPTensorClass;
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THSPTensor *)obj)->cdata = ptr.release();
  }
  return obj;
}

static void THSPTensor_(dealloc)(THSPTensor* self)
{
  if (self->cdata)
    THSTensor_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THSPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
#ifdef THC_GENERIC_FILE
#define THPIndexTensor_Check THCPLongTensor_Check
#define THPIndexTensor THCPLongTensor
#define THIndexTensor THCudaLongTensor
#else
#define THPIndexTensor_Check THPLongTensor_Check
#define THPIndexTensor THPLongTensor
#define THIndexTensor THLongTensor
#endif
  HANDLE_TH_ERRORS

#ifdef THC_GENERIC_FILE
  THCPAutoGPU gpu_guard(args, NULL);
#endif

  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

  THSPTensorPtr self((THSPTensor *)type->tp_alloc(type, 0));
  THPUtils_assert(self, "failed to allocate a " THSPTensorStr " object");
  THLongStoragePtr sizes;

  // Internally we allow constructing with a keyword only argument cdata
  if (kwargs != NULL) {
    Py_ssize_t num_kwargs = PyDict_Size(kwargs);
    if (num_args == 0) {
      // NB: This is an internal option, so we don't want to advertise it.
      PyObject *cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
      if (num_kwargs == 1 && cdata_ptr && THPUtils_checkLong(cdata_ptr)) {
        THSTensor *ptr = (THSTensor*)PyLong_AsVoidPtr(cdata_ptr);
        self->cdata = ptr;
        return (PyObject*)self.release();
      }
    }
#ifdef THC_GENERIC_FILE
    PyObject *device_id = PyDict_GetItemString(kwargs, "device");
    if (device_id == Py_None) {
      num_kwargs--;
    } else if (device_id) {
      THPUtils_assert(THPUtils_checkLong(device_id), "device argument "
          " has to be an int, but got %s", THPUtils_typename(device_id));
      gpu_guard.setDevice(THPUtils_unpackLong(device_id));
      // simulate pop() and pretend this key was never there
      num_kwargs--;
    }
#endif
#ifdef THC_GENERIC_FILE
    THPUtils_assert(num_kwargs == 0, THSPTensorStr " constructor only "
        "accepts a 'device' keyword argument");
#else
    THPUtils_assert(num_kwargs == 0, THPTensorStr " constructor doesn't "
        "accept any keyword arguments");
#endif
  }

  // torch.SparseTensor()
  if (num_args == 0) {
    self->cdata = THSTensor_(new)(LIBRARY_STATE_NOARGS);
    return (PyObject*)self.release();
  }

  PyObject *first_arg = PyTuple_GET_ITEM(args, 0);

  // torch.SparseTensor(size)
  if (num_args == 1 && THPUtils_checkLong(first_arg)) {
    long size = THPUtils_unpackLong(first_arg);
    self->cdata = THSTensor_(newWithSize1d)(LIBRARY_STATE size);
  }
  // torch.SparseTensor(torch.Size sizes)
  else if (num_args == 1 && THPSize_Check(first_arg)) {
    THLongStoragePtr sizes(THPUtils_unpackSize(first_arg));
    self->cdata = THSTensor_(newWithSize)(LIBRARY_STATE sizes.get());
  }
  // torch.SparseTensor(torch.LongTensor indices, torch.LongTensor values)
  else if (num_args == 2 && THPIndexTensor_Check(first_arg)) {
    PyObject *second_arg = PyTuple_GET_ITEM(args, 1);
    if (!THPTensor_(Check)(second_arg)) goto invalid_arguments;

    THIndexTensor *indices = ((THPIndexTensor*)first_arg)->cdata;
    THTensor *values = ((THPTensor*)second_arg)->cdata;

#ifdef THC_GENERIC_FILE
    THCAssertSameGPU(THSTensor_(checkGPU)(LIBRARY_STATE 0, 2, indices, values));
#endif

    self->cdata = THSTensor_(newWithTensor)(LIBRARY_STATE indices, values);
  }
  // torch.SparseTensor(torch.LongTensor indices,
  //                    torch.Tensor values,
  //                    torch.Size sizes)
  else if (num_args > 2 && THPIndexTensor_Check(first_arg)) {
    PyObject *second_arg = PyTuple_GET_ITEM(args, 1);
    PyObject *third_arg = PyTuple_GET_ITEM(args, 2);
    if (!THPTensor_(Check)(second_arg)) goto invalid_arguments;
    if (!THPSize_Check(third_arg)) goto invalid_arguments;

    THIndexTensor *indices = ((THPIndexTensor*)first_arg)->cdata;
    THTensor *values = ((THPTensor*)second_arg)->cdata;
    THLongStoragePtr sizes(THPUtils_unpackSize(third_arg));

#ifdef THC_GENERIC_FILE
    THCAssertSameGPU(THSTensor_(checkGPU)(LIBRARY_STATE 0, 2, indices, values));
#endif

    self->cdata = THSTensor_(newWithTensorAndSize)(
        LIBRARY_STATE indices, values, sizes);
  }
  // torch.SparseTensor(int ...)
  else if (THPUtils_tryUnpackLongVarArgs(args, 0, sizes)) {
    self->cdata = THSTensor_(newWithSize)(LIBRARY_STATE sizes.get());
  }
  else goto invalid_arguments; // All other cases

  THSPTensor_(initStorage)(self->cdata);

  return (PyObject*)self.release();

invalid_arguments:
  THPUtils_invalidArguments(args, NULL, THSPTensorStr " constructor", 6,
      "no arguments",
      "(int size)",
      "(torch.Size sizes)",
#ifdef THC_GENERIC_FILE
      "(torch.cuda.LongTensor indices, " THPTensorStr " values)",
      "(torch.cuda.LongTensor indices, " THPTensorStr " values, torch.Size sizes)",
#else
      "(torch.LongTensor indices, " THPTensorStr " values)",
      "(torch.LongTensor indices, " THPTensorStr " values, torch.Size sizes)",
#endif
      "(int ...)");
  return NULL;
  END_HANDLE_TH_ERRORS
#undef THPIndexTensor_Check
#undef THPIndexTensor
#undef THIndexTensor
}

// TODO: implement equality
PyTypeObject THSPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C.Sparse" THPTensorBaseStr,    /* tp_name */
  sizeof(THSPTensor),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THSPTensor_(dealloc),      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,//&THSPTensor_(mappingmethods),          /* tp_as_mapping */
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
  THSPTensor_(pynew),                    /* tp_new */
};

static struct PyMemberDef THSPTensor_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THSPTensor, cdata), READONLY, NULL},
  {NULL} // Sentinel
};

typedef struct {
  PyObject_HEAD
} THSPTensorStateless;

PyTypeObject THSPTensorStatelessType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C.Sparse" THPTensorBaseStr ".stateless", /* tp_name */
  sizeof(THSPTensorStateless),            /* tp_basicsize */
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
  THSPTensor_stateless_(methods),        /* tp_methods */
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

bool THSPTensor_(init)(PyObject *module)
{
  THSPTensorType.tp_methods = THSPTensor_(methods);
  THSPTensorType.tp_members = THSPTensor_(members);
  if (PyType_Ready(&THSPTensorType) < 0)
    return false;
  THSPTensorStatelessType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&THSPTensorStatelessType) < 0)
    return false;

  PyModule_AddObject(module, THSPTensorBaseStr, (PyObject *)&THSPTensorType);
  return true;
}

bool THSPTensor_(postInit)(PyObject *module)
{
  THSPTensorClass = PyObject_GetAttrString(module, TH_CONCAT_STRING_2(Real,Tensor));
  if (!THSPTensorClass) return false;
  bool is_cuda = false;
#ifdef THC_GENERIC_FILE
  is_cuda = true;
#endif
  const char *type_name = TH_CONCAT_STRING_2(Real,);
  torch::registerPyTypeObject((PyTypeObject*)THSPTensorClass, type_name, is_cuda, true);
  return true;
}
