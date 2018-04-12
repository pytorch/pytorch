#include <Python.h>

#include "DynamicTypes.h"
#include "PythonTypes.h"
#include "Exceptions.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/cuda_enabled.h"

#include <vector>
#include <unordered_map>
#include <sstream>

#ifdef WITH_CUDA
#include <THC/THC.h>
#include <THCS/THCS.h>
#endif

namespace torch {

static std::unordered_map<std::string, at::ScalarType> attype_names = {
  {"Float", at::kFloat},
  {"Double", at::kDouble},
  {"Half", at::kHalf},
  {"Byte", at::kByte},
  {"Char", at::kChar},
  {"Short", at::kShort},
  {"Int", at::kInt},
  {"Long", at::kLong},
};

static std::unordered_map<at::Type*, PyTypeObject*> attype_to_py_storage_type;
static std::unordered_map<PyTypeObject*, at::Type*> py_storage_type_to_attype;

static const int NumBoolOptions = 2;
static THPDtype* dtype_registry
  [static_cast<int>(at::ScalarType::NumOptions)] = {};

static THPLayout* layout_registry
  [static_cast<int>(at::Backend::NumOptions)] = {};

static at::Backend get_backend(bool is_cuda, bool is_sparse) {
  if (is_cuda) {
    if (is_sparse){
      return at::kSparseCUDA;
    } else {
      return at::kCUDA;
    }
  } else {
    if (is_sparse){
      return at::kSparseCPU;
    } else {
      return at::kCPU;
    }
  }
}

static at::Type* get_type(const std::string& name, bool is_cuda, bool is_sparse) {
  if (is_sparse && name == "Half") {
    return nullptr;
  }
  at::Backend backend = get_backend(is_cuda, is_sparse);
  return &at::getType(backend, attype_names.at(name));
}

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

static PyTypeObject* getPyTypeObject(const at::Storage& storage)
{
  auto it = attype_to_py_storage_type.find(&storage.type());
  if (it != attype_to_py_storage_type.end()) {
    return it->second;
  }
  throw std::invalid_argument("unsupported Storage type");
}

at::Type& getType(at::ScalarType scalarType, const THPLayout& layout, const DeviceType& deviceType) {
  at::Backend backend = get_backend(deviceType == DeviceType::CUDA, !layout.is_strided);
  // use type_registry rather than context.getType() because getType throws exceptions.
  auto baseType = at::globalContext().type_registry[static_cast<int>(backend)]
                                                   [static_cast<int>(scalarType)].get();
  if (!baseType) {
    std::ostringstream oss;
    oss << "Error attempting to use dtype " << getDtype(scalarType)->name << " with layout " << layout.name
        << " and device type " << (deviceType == DeviceType::CPU ? "CPU" : "CUDA") << ".";
    if (deviceType == DeviceType::CUDA && !torch::utils::cuda_enabled()) {
      oss << "  Torch not compiled with CUDA enabled." << std::endl;
    }
    throw std::runtime_error(oss.str());
  }
  return *torch::autograd::VariableType::getType(*baseType);
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

DeviceType getDeviceType(const at::Type& type) {
  return type.is_cuda() ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
}

PyObject* createPyObject(const at::Storage& storage)
{
  auto type = getPyTypeObject(storage);
  auto obj = THPObjectPtr(type->tp_alloc(type, 0));
  if (!obj) throw python_error();
  ((THPVoidStorage*)obj.get())->cdata = (THVoidStorage *)storage.unsafeGetTH(true);
  return obj.release();
}

bool isStorage(PyObject* obj)
{
  auto it = py_storage_type_to_attype.find(Py_TYPE(obj));
  return it != py_storage_type_to_attype.end();
}
std::unique_ptr<at::Storage> createStorage(PyObject* obj)
{
  auto it = py_storage_type_to_attype.find(Py_TYPE(obj));
  if (it == py_storage_type_to_attype.end()) {
    throw TypeError("not a storage '%s'", Py_TYPE(obj)->tp_name);
  }
  auto& type = *it->second;
  return type.unsafeStorageFromTH(((THPVoidStorage*)obj)->cdata, true);
}

}  // namespace
