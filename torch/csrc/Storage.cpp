#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
#include <structmember.h>

#define THP_HOST_HALF

#include <TH/TH.h>
// See Note [TH abstraction violation]
//  - Used to get at the allocator associated with a storage
#include <TH/THStorageFunctions.hpp>
#include <libshm.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

#include <fmt/format.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <torch/csrc/generic/Storage.cpp>
#include <TH/THGenerateByteType.h>

template<>
void THPPointer<THStorage>::free() {
  if (ptr) {
    THStorage_free(ptr);
  }
}

bool THPByteStorage_init(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THPByteStorage_methods);
  THPUtils_addPyMethodDefs(methods, THPByteStorage_sharingMethods);

  THPByteStorageType.tp_methods = methods.data();
  THPByteStorageType.tp_members = THPByteStorage_members;
  THPByteStorageType.tp_getset = THPByteStorage_properties;
  if (PyType_Ready(&THPByteStorageType) < 0)
    return false;
  Py_INCREF(&THPByteStorageType);
  PyModule_AddObject(module, "ByteStorageBase", (PyObject *)&THPByteStorageType);
  THPByteStorage_initCopyMethods();
  return true;
}

void THPByteStorage_postInit(PyObject *module)
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

  torch::registerStoragePyTypeObject((PyTypeObject*)THPByteStorageClass, backend, at::kByte);
}
