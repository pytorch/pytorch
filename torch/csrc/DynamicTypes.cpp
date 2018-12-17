#include "torch/csrc/python_headers.h"

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/Layout.h"
#include "torch/csrc/PythonTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/cuda_enabled.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

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
const std::unordered_map<std::string, at::ScalarType> attype_names = {
  {"Float", at::kFloat},
  {"Double", at::kDouble},
  {"Half", at::kHalf},
  {"Byte", at::kByte},
  {"Char", at::kChar},
  {"Short", at::kShort},
  {"Int", at::kInt},
  {"Long", at::kLong},
};

std::unordered_map<at::Type*, PyTypeObject*> attype_to_py_storage_type;
std::unordered_map<PyTypeObject*, at::Type*> py_storage_type_to_attype;

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

at::Type* get_type(const std::string& name, bool is_cuda, bool is_sparse) {
  if (is_sparse && name == "Half") {
    return nullptr;
  }
  at::Backend backend = get_backend(is_cuda, is_sparse);
  return &at::getNonVariableType(backend, attype_names.at(name));
}

PyTypeObject* getPyTypeObject(const at::Storage& storage)
{
  auto attype = at::globalContext().getNonVariableTypeOpt(
      at::deviceTypeToBackend(storage.device_type()),
      at::typeMetaToScalarType(storage.dtype()));
  auto it = attype_to_py_storage_type.find(attype);
  if (it != attype_to_py_storage_type.end()) {
    return it->second;
  }
  throw std::invalid_argument("unsupported Storage type");
}
} // namespace

void registerStoragePyTypeObject(PyTypeObject *pytype, const std::string& name, bool is_cuda, bool is_sparse)
{
  auto attype = get_type(name, is_cuda, is_sparse);
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

at::Type& getVariableType(at::ScalarType scalarType, const THPLayout& layout, const at::Device& device) {
  const at::Backend backend = get_backend(device.type() == at::Device::Type::CUDA, layout.layout == at::Layout::Sparse);
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  auto baseType = at::globalContext().getNonVariableTypeOpt(backend, scalarType);
  if (!baseType) {
    std::ostringstream oss;
    oss << "Error attempting to use dtype " << getDtype(scalarType)->name << " with layout " << layout.name
        << " and device type " << device.type() << ".";
    if (device.type() == at::Device::Type::CUDA && !torch::utils::cuda_enabled()) {
      oss << "  Torch not compiled with CUDA enabled." << std::endl;
    }
    throw std::runtime_error(oss.str());
  }
  return *torch::autograd::VariableType::getVariableTypeFromBaseType(*baseType);
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

at::Device::Type getDeviceType(const at::Type& type) {
  return type.is_cuda() ? at::Device::Type::CUDA : at::Device::Type::CPU;
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
