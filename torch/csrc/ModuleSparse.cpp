#include "THP.h"

PyObject* sparse_tensor_classes;

////////////////////////////////////////////////////////////////////////////////
// SPARSE MODULE INITIALIZATION
////////////////////////////////////////////////////////////////////////////////

static bool THSPModule_loadClasses(PyObject *module_dict)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  ASSERT_NOT_NULL(sparse_tensor_classes = PyMapping_GetItemString(module_dict, (char*)"_sparse_tensor_classes"));
  ASSERT_NOT_NULL(THSPDoubleTensorClass  = PyMapping_GetItemString(module_dict, (char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THSPFloatTensorClass   = PyMapping_GetItemString(module_dict, (char*)"FloatTensor"));
  ASSERT_NOT_NULL(THSPLongTensorClass    = PyMapping_GetItemString(module_dict, (char*)"LongTensor"));
  ASSERT_NOT_NULL(THSPIntTensorClass     = PyMapping_GetItemString(module_dict, (char*)"IntTensor"));
  ASSERT_NOT_NULL(THSPShortTensorClass   = PyMapping_GetItemString(module_dict, (char*)"ShortTensor"));
  ASSERT_NOT_NULL(THSPCharTensorClass    = PyMapping_GetItemString(module_dict, (char*)"CharTensor"));
  ASSERT_NOT_NULL(THSPByteTensorClass    = PyMapping_GetItemString(module_dict, (char*)"ByteTensor"));

  return true;
#undef ASSERT_NOT_NULL
}

static bool THSPModule_assignStateless()
{
#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_3(Sparse, type, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THSP,type,TensorClass), THP_STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS(Float);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS
}

// Callback for python part. Used for additional initialization of python classes
PyObject *THSPModule_initExtension(PyObject *self)
{
#define ASSERT_TRUE(cond) if (!(cond)) { Py_RETURN_FALSE; }
  PyObject *module = PyImport_ImportModule("torch.sparse");
  if (!module) {
    THPUtils_setError("class loader couldn't access torch.sparse module");
    return NULL;
  }

  PyObject* module_dict = PyModule_GetDict(module);
  ASSERT_TRUE(THSPModule_loadClasses(module_dict));
  ASSERT_TRUE(THSPModule_assignStateless());
  Py_RETURN_TRUE;
#undef ASSERT_TRUE
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Stateless Functions
////////////////////////////////////////////////////////////////////////////////

bool THPModule_isSparseTensor(PyObject *obj)
{
  int result = PySet_Contains(sparse_tensor_classes, (PyObject*)Py_TYPE(obj));
  if (result == -1)
    throw std::logic_error("FATAL: sparse_tensor_classes isn't a set!");
  return result;
}


#define IMPLEMENT_SPARSE_STATELESS(name)                                              \
static PyObject * TH_CONCAT_2(THSPModule_, name)(PyObject *_unused, PyObject *args, PyObject *kwargs) \
{                                                                              \
  PyObject *tensor = THSPFloatTensorClass;                                    \
  PyObject *key, *value;                                                       \
  Py_ssize_t pos = 0;                                                          \
  for (int i = 0; i < PyTuple_Size(args); i++) {                               \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item) || THPVariable_CheckType(item, THPModule_isSparseTensor)) { \
      tensor = item;                                                           \
      goto dispatch;                                                           \
    }                                                                          \
  }                                                                            \
  if (kwargs) {                                                                \
    while (PyDict_Next(kwargs, &pos, &key, &value)) {                          \
      if (THPModule_isTensor(value) || THPVariable_CheckType(value, THPModule_isSparseTensor)) {             \
        tensor = value;                                                        \
        goto dispatch;                                                         \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
dispatch:                                                                      \
  THPObjectPtr methods = PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME); \
  THPUtils_assert(methods, "Type %s doesn't implement stateless methods",      \
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor)); \
  THPObjectPtr method = PyObject_GetAttrString(methods, #name);                \
  THPUtils_assert(method, "Type %s doesn't implement stateless method " #name, \
      tensor == THPDefaultTensorClass ? THPUtils_classname(tensor) : THPUtils_typename(tensor)); \
  return PyObject_Call(method, args, kwargs);                                  \
}

IMPLEMENT_SPARSE_STATELESS(sspmm);
IMPLEMENT_SPARSE_STATELESS(sspaddmm);

#undef IMPLEMENT_SPARSE_STATELESS
