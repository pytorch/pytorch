#include <torch/csrc/python_headers.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/PythonTypes.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>

#include <ATen/ATen.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include <THC/THC.h>
#endif

namespace torch {
namespace {
std::unordered_map<at::DeprecatedTypeProperties*, PyTypeObject*> attype_to_py_storage_type;
std::unordered_map<PyTypeObject*, at::DeprecatedTypeProperties*> py_storage_type_to_attype;

THPDtype* dtype_registry
  [static_cast<int>(at::ScalarType::NumOptions)] = {};

THPLayout* layout_registry
  [static_cast<int>(at::Backend::NumOptions)] = {};

at::Backend get_backend(bool is_cuda, bool is_sparse) {
  if (is_cuda) {
    if (is_sparse){
      return at::Backend::SparseCUDA;
    } else {
      return at::Backend::CUDA;
    }
  } else {
    if (is_sparse){
      return at::Backend::SparseCPU;
    } else {
      return at::Backend::CPU;
    }
  }
}

at::DeprecatedTypeProperties* get_type(at::Backend backend, at::ScalarType scalarType) {
  if (isSparse(backend) && scalarType == at::kHalf) {
    return nullptr;
  }
  return &at::getDeprecatedTypeProperties(backend, scalarType);
}

PyTypeObject* getPyTypeObject(const at::Storage& storage)
{
  at::ScalarType scalarType = at::typeMetaToScalarType(storage.dtype());
  at::TensorOptions options = at::TensorOptions(storage.device_type()).dtype(scalarType);
  auto attype = &at::getDeprecatedTypeProperties(
      at::dispatchKeyToBackend(at::computeDispatchKey(options)),
      scalarType);
  auto it = attype_to_py_storage_type.find(attype);
  if (it != attype_to_py_storage_type.end()) {
    return it->second;
  }
  throw std::invalid_argument("unsupported Storage type");
}
} // namespace

void registerStoragePyTypeObject(PyTypeObject *pytype, at::Backend backend, at::ScalarType scalarType) {
  auto attype = get_type(backend, scalarType);
  if (attype) {
    attype_to_py_storage_type[attype] = pytype;
    py_storage_type_to_attype[pytype] = attype;
  }
}

void registerDtypeObject(THPDtype *dtype, at::ScalarType scalarType) {
  dtype_registry[static_cast<int>(scalarType)] = dtype;
}

void registerLayoutObject(THPLayout *layout, at::Backend backend) {
  layout_registry[static_cast<int>(backend)] = layout;
}

THPDtype* getDtype(at::ScalarType scalarType) {
  auto dtype = dtype_registry[static_cast<int>(scalarType)];
  if (!dtype) {
    throw std::invalid_argument("unsupported scalarType");
  }
  return dtype;
}

THPLayout* getLayout(at::Backend backend) {
  auto layout = layout_registry[static_cast<int>(backend)];
  if (!layout) {
    throw std::invalid_argument("unsupported at::Backend");
  }
  return layout;
}

PyObject* createPyObject(const at::Storage& storage)
{
  auto type = getPyTypeObject(storage);
  auto obj = THPObjectPtr(type->tp_alloc(type, 0));
  if (!obj) throw python_error();
  ((THPVoidStorage*)obj.get())->cdata = (THVoidStorage *)at::Storage(/* copy */ storage).unsafeReleaseStorageImpl();
  return obj.release();
}

bool isStorage(PyObject* obj)
{
  return py_storage_type_to_attype.count(Py_TYPE(obj));
}
at::Storage createStorage(PyObject* obj)
{
  auto it = py_storage_type_to_attype.find(Py_TYPE(obj));
  if (it == py_storage_type_to_attype.end()) {
    throw TypeError("not a storage '%s'", Py_TYPE(obj)->tp_name);
  }
  auto& type = *it->second;
  return type.unsafeStorageFromTH(((THPVoidStorage*)obj)->cdata, true);
}

}  // namespace
