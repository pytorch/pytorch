#define __STDC_FORMAT_MACROS

#include <torch/csrc/python_headers.h>
#include <structmember.h>
#include <fmt/format.h>

// See Note [TH abstraction violation]
//    - Used to get at allocator from storage
#include <TH/THTensor.hpp>
#include <THC/THCTensor.hpp>
#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/cuda/override_macros.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.cpp"
#include <THC/THCGenerateByteType.h>

bool THCPByteStorage_init(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THCPByteStorage_methods);
  THPUtils_addPyMethodDefs(methods, THCPByteStorage_sharingMethods);

  THPByteStorageType.tp_methods = methods.data();
  THPByteStorageType.tp_members = THCPByteStorage_members;
  THPByteStorageType.tp_getset = THCPByteStorage_properties;
  if (PyType_Ready(&THCPByteStorageType) < 0)
    return false;
  Py_INCREF(&THCPByteStorageType);
  PyModule_AddObject(module, "CudaByteStorageBase", (PyObject *)&THCPByteStorageType);
  THCPByteStorage_initCopyMethods();
  return true;
}

void THCPByteStorage_postInit(PyObject *module)
{
  THPByteStorageClass = PyObject_GetAttrString(module, "UntypedStorage");
  if (!THPByteStorageClass) throw python_error();

  at::Backend backend = at::Backend::CPU;
#ifdef THC_GENERIC_FILE
  backend = at::Backend::CUDA;
#endif

#ifdef THQUANTIZED
  backend = at::Backend::QuantizedCPU;
#endif

  torch::registerStoragePyTypeObject((PyTypeObject*)THCPByteStorageClass, backend, at::kByte);
}
