#include <torch/csrc/python_headers.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>

#include <ATen/ATen.h>
#include <ATen/FunctionalStorageImpl.h>

#include <array>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace {
std::array<THPDtype*, static_cast<int>(at::ScalarType::NumOptions)>
    dtype_registry = {};

std::array<THPLayout*, static_cast<int>(at::Layout::NumOptions)>
    layout_registry = {};

} // namespace

void registerDtypeObject(THPDtype* dtype, at::ScalarType scalarType) {
  dtype_registry[static_cast<int>(scalarType)] = dtype;
}

void registerLayoutObject(THPLayout* thp_layout, at::Layout layout) {
  layout_registry[static_cast<int>(layout)] = thp_layout;
}

THPDtype* getTHPDtype(at::ScalarType scalarType) {
  auto dtype = dtype_registry[static_cast<int>(scalarType)];
  if (!dtype) {
    throw std::invalid_argument("unsupported scalarType");
  }
  return dtype;
}

THPLayout* getTHPLayout(at::Layout layout) {
  auto thp_layout = layout_registry[static_cast<int>(layout)];
  if (!thp_layout) {
    throw std::invalid_argument("unsupported at::Layout");
  }
  return thp_layout;
}

PyObject* createPyObject(const at::Storage& storage) {
  if (storage.device_type() != at::DeviceType::Meta &&
      storage.data() == nullptr && storage.sym_nbytes() != 0 &&
      // Grabbing storage() from FunctionalTensorWrapper is allowed.
      // This is useful for checking aliasing info from python
      dynamic_cast<at::functionalization::FunctionalStorageImpl*>(
          storage.unsafeGetStorageImpl()) == nullptr) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "python bindings to nullptr storage (e.g., from torch.Tensor._make_wrapper_subclass) are currently unsafe and thus disabled.  See https://github.com/pytorch/pytorch/issues/61669 for more details");
  }
  PyObject* obj = THPStorage_Wrap(storage);
  if (!obj)
    throw python_error();
  return obj;
}

PyTypeObject* loadTypedStorageTypeObject() {
  PyObject* storage_module = PyImport_ImportModule("torch.storage");
  TORCH_INTERNAL_ASSERT(storage_module && PyModule_Check(storage_module));

  PyObject* typed_storage_obj =
      PyObject_GetAttrString(storage_module, "TypedStorage");
  TORCH_INTERNAL_ASSERT(typed_storage_obj && PyType_Check(typed_storage_obj));
  return reinterpret_cast<PyTypeObject*>(
      PyObject_GetAttrString(storage_module, "TypedStorage"));
}

PyTypeObject* getTypedStorageTypeObject() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static PyTypeObject* typed_storage_type_obj = loadTypedStorageTypeObject();
  return typed_storage_type_obj;
}

bool isStorage(PyObject* obj) {
  if (PyObject_TypeCheck(obj, getTypedStorageTypeObject())) {
    return true;
  }
  return THPStorage_Check(obj);
}

std::tuple<at::Storage, at::ScalarType, bool> createStorageGetType(
    PyObject* obj) {
  at::ScalarType scalar_type;
  bool is_typed_storage;

  is_typed_storage = PyObject_TypeCheck(obj, getTypedStorageTypeObject());
  PyObject* untyped_storage_obj;

  if (is_typed_storage) {
    // NOTE: `PyObject_GetAttrString` increments the refcounts to `dtype` and
    // `_untyped_storage`, so we must decrement them. The refcounts will still
    // stay nonzero since the `TypedStorage` maintains a reference.
    PyObject* dtype_obj = PyObject_GetAttrString(obj, "dtype");
    TORCH_INTERNAL_ASSERT(dtype_obj);
    TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_obj));
    scalar_type = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
    Py_DECREF(dtype_obj);

    untyped_storage_obj = PyObject_GetAttrString(obj, "_untyped_storage");
    TORCH_INTERNAL_ASSERT(untyped_storage_obj);
    Py_DECREF(untyped_storage_obj);

  } else {
    scalar_type = at::kByte;
    untyped_storage_obj = obj;
  }

  TORCH_CHECK(
      THPStorage_Check(untyped_storage_obj),
      "not a storage '",
      Py_TYPE(obj)->tp_name,
      "'");

  auto storage = THPStorage_Unpack(untyped_storage_obj);
  return std::make_tuple(storage, scalar_type, is_typed_storage);
}

at::Storage createStorage(PyObject* obj) {
  return std::get<0>(createStorageGetType(obj));
}

} // namespace torch
