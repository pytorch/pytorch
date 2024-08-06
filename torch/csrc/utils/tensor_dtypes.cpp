#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_dtypes.h>

namespace torch::utils {

void initializeDtypes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,

  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

#undef DEFINE_SCALAR_TYPE

  for (at::ScalarType scalarType : all_scalar_types) {
    auto [primary_name, legacy_name] = c10::getDtypeNames(scalarType);
    PyObject* dtype = THPDtype_New(scalarType, primary_name);
    torch::registerDtypeObject((THPDtype*)dtype, scalarType);
    Py_INCREF(dtype);
    if (PyModule_AddObject(torch_module.get(), primary_name.c_str(), dtype) !=
        0) {
      throw python_error();
    }
    if (!legacy_name.empty()) {
      Py_INCREF(dtype);
      if (PyModule_AddObject(torch_module.get(), legacy_name.c_str(), dtype) !=
          0) {
        throw python_error();
      }
    }
  }
}

} // namespace torch::utils
