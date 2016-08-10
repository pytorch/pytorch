#include <Python.h>
#include <exception>

#define REQUIRES_CUDA $requres_cuda

// TODO: use THP instead of this hack
struct Tensor {
  PyObject_HEAD
  void *cdata;
};

PyObject *THPDoubleStorageClass = NULL;
PyObject *THPFloatStorageClass  = NULL;
PyObject *THPLongStorageClass   = NULL;
PyObject *THPIntStorageClass    = NULL;
PyObject *THPShortStorageClass  = NULL;
PyObject *THPCharStorageClass   = NULL;
PyObject *THPByteStorageClass   = NULL;

PyObject *THPDoubleTensorClass  = NULL;
PyObject *THPFloatTensorClass   = NULL;
PyObject *THPLongTensorClass    = NULL;
PyObject *THPIntTensorClass     = NULL;
PyObject *THPShortTensorClass   = NULL;
PyObject *THPCharTensorClass    = NULL;
PyObject *THPByteTensorClass    = NULL;

#if REQUIRES_CUDA
PyObject *THCPDoubleStorageClass = NULL;
PyObject *THCPFloatStorageClass  = NULL;
PyObject *THCPLongStorageClass   = NULL;
PyObject *THCPIntStorageClass    = NULL;
PyObject *THCPHalfStorageClass   = NULL;
PyObject *THCPShortStorageClass  = NULL;
PyObject *THCPCharStorageClass   = NULL;
PyObject *THCPByteStorageClass   = NULL;

PyObject *THCPDoubleTensorClass  = NULL;
PyObject *THCPFloatTensorClass   = NULL;
PyObject *THCPLongTensorClass    = NULL;
PyObject *THCPIntTensorClass     = NULL;
PyObject *THCPHalfTensorClass    = NULL;
PyObject *THCPShortTensorClass   = NULL;
PyObject *THCPCharTensorClass    = NULL;
PyObject *THCPByteTensorClass    = NULL;
#endif

static bool __loadClasses()
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { PyErr_SetString(PyExc_RuntimeError, "couldn't load classes"); return false; }
  PyObject *torch_module = PyImport_ImportModule("torch");
  if (!torch_module) {
    PyErr_SetString(PyExc_RuntimeError, "class loader couldn't access torch module");
    return false;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);

  ASSERT_NOT_NULL(THPDoubleStorageClass = PyMapping_GetItemString(module_dict,(char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THPFloatStorageClass  = PyMapping_GetItemString(module_dict,(char*)"FloatStorage"));
  ASSERT_NOT_NULL(THPLongStorageClass   = PyMapping_GetItemString(module_dict,(char*)"LongStorage"));
  ASSERT_NOT_NULL(THPIntStorageClass    = PyMapping_GetItemString(module_dict,(char*)"IntStorage"));
  ASSERT_NOT_NULL(THPShortStorageClass  = PyMapping_GetItemString(module_dict,(char*)"ShortStorage"));
  ASSERT_NOT_NULL(THPCharStorageClass   = PyMapping_GetItemString(module_dict,(char*)"CharStorage"));
  ASSERT_NOT_NULL(THPByteStorageClass   = PyMapping_GetItemString(module_dict,(char*)"ByteStorage"));

  ASSERT_NOT_NULL(THPDoubleTensorClass  = PyMapping_GetItemString(module_dict,(char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THPFloatTensorClass   = PyMapping_GetItemString(module_dict,(char*)"FloatTensor"));
  ASSERT_NOT_NULL(THPLongTensorClass    = PyMapping_GetItemString(module_dict,(char*)"LongTensor"));
  ASSERT_NOT_NULL(THPIntTensorClass     = PyMapping_GetItemString(module_dict,(char*)"IntTensor"));
  ASSERT_NOT_NULL(THPShortTensorClass   = PyMapping_GetItemString(module_dict,(char*)"ShortTensor"));
  ASSERT_NOT_NULL(THPCharTensorClass    = PyMapping_GetItemString(module_dict,(char*)"CharTensor"));
  ASSERT_NOT_NULL(THPByteTensorClass    = PyMapping_GetItemString(module_dict,(char*)"ByteTensor"));

#if REQUIRES_CUDA
  PyObject *cuda_module = PyImport_ImportModule("torch.cuda");
  if (!torch_module) {
    PyErr_SetString(PyExc_RuntimeError, "class loader couldn't access torch.cuda module");
    return false;
  }
  PyObject* cuda_module_dict = PyModule_GetDict(cuda_module);

  ASSERT_NOT_NULL(THCPDoubleStorageClass = PyMapping_GetItemString(cuda_module_dict, (char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THCPFloatStorageClass  = PyMapping_GetItemString(cuda_module_dict, (char*)"FloatStorage"));
  ASSERT_NOT_NULL(THCPHalfStorageClass   = PyMapping_GetItemString(cuda_module_dict, (char*)"HalfStorage"));
  ASSERT_NOT_NULL(THCPLongStorageClass   = PyMapping_GetItemString(cuda_module_dict, (char*)"LongStorage"));
  ASSERT_NOT_NULL(THCPIntStorageClass    = PyMapping_GetItemString(cuda_module_dict, (char*)"IntStorage"));
  ASSERT_NOT_NULL(THCPShortStorageClass  = PyMapping_GetItemString(cuda_module_dict, (char*)"ShortStorage"));
  ASSERT_NOT_NULL(THCPCharStorageClass   = PyMapping_GetItemString(cuda_module_dict, (char*)"CharStorage"));
  ASSERT_NOT_NULL(THCPByteStorageClass   = PyMapping_GetItemString(cuda_module_dict, (char*)"ByteStorage"));

  ASSERT_NOT_NULL(THCPDoubleTensorClass  = PyMapping_GetItemString(cuda_module_dict, (char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THCPHalfTensorClass    = PyMapping_GetItemString(cuda_module_dict, (char*)"HalfTensor"));
  ASSERT_NOT_NULL(THCPFloatTensorClass   = PyMapping_GetItemString(cuda_module_dict, (char*)"FloatTensor"));
  ASSERT_NOT_NULL(THCPLongTensorClass    = PyMapping_GetItemString(cuda_module_dict, (char*)"LongTensor"));
  ASSERT_NOT_NULL(THCPIntTensorClass     = PyMapping_GetItemString(cuda_module_dict, (char*)"IntTensor"));
  ASSERT_NOT_NULL(THCPShortTensorClass   = PyMapping_GetItemString(cuda_module_dict, (char*)"ShortTensor"));
  ASSERT_NOT_NULL(THCPCharTensorClass    = PyMapping_GetItemString(cuda_module_dict, (char*)"CharTensor"));
  ASSERT_NOT_NULL(THCPByteTensorClass    = PyMapping_GetItemString(cuda_module_dict, (char*)"ByteTensor"));
#endif

  return true;
#undef ASSERT_NOT_NULL
}

// TODO: duplicate code
#include <string>
void __invalidArgs(PyObject *given_args, const char *expected_args_desc) {
  static const std::string PREFIX = "Invalid arguments! Got ";
  std::string error_msg;
  error_msg.reserve(2000);
  error_msg += PREFIX;

  // TODO: assert that args is a tuple?
  Py_ssize_t num_args = PyTuple_Size(given_args);
  if (num_args == 0) {
    error_msg += "no arguments";
  } else {
    error_msg += "(";
    for (int i = 0; i < num_args; i++) {
      PyObject *arg = PyTuple_GET_ITEM(given_args, i);
      if (i > 0)
        error_msg += ", ";
      error_msg += Py_TYPE(arg)->tp_name;
    }
    error_msg += ")";
  }
  error_msg += ", but expected ";
  error_msg += expected_args_desc;
  PyErr_SetString(PyExc_ValueError, error_msg.c_str());
}

bool __checkFloat(PyObject *arg) {
  return PyFloat_Check(arg) || PyLong_Check(arg);
}

double __getFloat(PyObject *arg) {
  if (PyFloat_Check(arg)) {
    return PyFloat_AsDouble(arg);
  } else {
    return PyLong_AsDouble(arg);
  }
}
