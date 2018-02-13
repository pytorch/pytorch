#include <Python.h>
#include "tensor_dtypes.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/generated/VariableType.h"

namespace torch { namespace utils {

static std::tuple<std::string, std::string> getDtypeNames(const at::Type &type) {
  switch(type.scalarType()) {
    case at::ScalarType::Byte:
      // no "byte" because byte is signed in numpy and we overload
      // byte to mean bool often
      return std::make_tuple("uint8", "");
    case at::ScalarType::Char:
      // no "char" because it is not consistently signed or unsigned; we want
      // to move to int8
      return std::make_tuple("int8", "");
    case at::ScalarType::Double:
      return std::make_tuple("float64", "double");
    case at::ScalarType::Float:
      return std::make_tuple("float32", "float");
    case at::ScalarType::Int:
      return std::make_tuple("int32", "int");
    case at::ScalarType::Long:
      return std::make_tuple("int64", "long");
    case at::ScalarType::Short:
      return std::make_tuple("int16", "short");
    case at::ScalarType::Half:
      return std::make_tuple("float16", "half");
    default:
      throw std::runtime_error("Unimplemented scalar type");
  }
}

void initializeDtypes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  auto cuda_module = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
  auto sparse_module = THPObjectPtr(PyImport_ImportModule("torch.sparse"));
  auto cuda_sparse_module = THPObjectPtr(PyImport_ImportModule("torch.cuda.sparse"));
  for (auto type : torch::autograd::VariableType::allTypes()) {
    std::string primary_name, legacy_name;
    std::tie(primary_name, legacy_name) = getDtypeNames(*type);
    PyObject *module = nullptr;
    switch (type->backend()) {
      case at::kCPU: {
        module = torch_module.get();
        break;
      }
      case at::kCUDA: {
        module = cuda_module.get();
        break;
      }
      case at::kSparseCPU: {
        module = sparse_module.get();
        break;
      }
      case at::kSparseCUDA: {
        module = cuda_sparse_module.get();
        break;
      }
      default: throw std::runtime_error("Unimplemented backend");
    }
    std::string name = std::string(PyModule_GetName(module)) + '.' + primary_name;
    THPDtype *dtype = (THPDtype*)THPDtype_New(type, name);
    torch::registerDtypeObject(dtype, *type);
    PyModule_AddObject(module, primary_name.c_str(), (PyObject*)dtype);
    if (legacy_name != "") {
      PyModule_AddObject(module, legacy_name.c_str(), (PyObject*)dtype);
    }
  }
}

}} // namespace torch::utils
