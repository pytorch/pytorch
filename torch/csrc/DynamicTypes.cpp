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

at::DeprecatedTypeProperties* get_type_properties(
    at::DeviceType device_type,
    at::ScalarType scalarType) {
  at::Backend backend;
  if (device_type == at::kCPU) {
    backend = at::Backend::CPU;
  } else if (device_type == at::kCUDA) {
    backend = at::Backend::CUDA;
  } else if (device_type == at::kXPU) {
    backend = at::Backend::XPU;
  } else if (device_type == at::kHPU) {
    backend = at::Backend::HPU;
  } else if (device_type == at::kMPS) {
    backend = at::Backend::MPS;
  } else if (device_type == at::DeviceType::Meta) {
    backend = at::Backend::Undefined;
  } else if (device_type == at::DeviceType::PrivateUse1) {
    backend = at::Backend::PrivateUse1;
  } else {
    TORCH_CHECK(false, "Invalid device for storage: ", device_type);
  }
  return &at::getDeprecatedTypeProperties(backend, scalarType);
}
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
  PyTypeObject* type = reinterpret_cast<PyTypeObject*>(THPStorageClass);
  auto obj = THPObjectPtr(type->tp_alloc(type, 0));
  if (!obj)
    throw python_error();
  ((THPStorage*)obj.get())->cdata =
      c10::MaybeOwned<at::Storage>::owned(at::Storage(/* copy */ storage));
  return obj.release();
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
  auto obj_type = Py_TYPE(obj);

  return obj_type == reinterpret_cast<PyTypeObject*>(THPStorageClass);
}

at::Storage createStorageGetType(
    PyObject* obj,
    at::ScalarType& scalar_type,
    bool& is_typed_storage) {
  is_typed_storage = PyObject_TypeCheck(obj, getTypedStorageTypeObject());
  PyObject* untyped_storage_obj;

  if (is_typed_storage) {
    // NOTE: `PyObject_GetAttrString` increments the refcounts to `dtype` and
    // `_untyped_storage`, so we must decrement them. The refcounts will still
    // stay nonzero since the `TypedStorage` maintains a reference.
    PyObject* dtype_obj = PyObject_GetAttrString(obj, "dtype");
    TORCH_INTERNAL_ASSERT(dtype_obj);
    Py_DECREF(dtype_obj);

    TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_obj));
    scalar_type = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;

    untyped_storage_obj = PyObject_GetAttrString(obj, "_untyped_storage");
    TORCH_INTERNAL_ASSERT(untyped_storage_obj);
    Py_DECREF(untyped_storage_obj);

  } else {
    scalar_type = at::kByte;
    untyped_storage_obj = obj;
  }

  if (Py_TYPE(untyped_storage_obj) !=
      reinterpret_cast<PyTypeObject*>(THPStorageClass)) {
    throw TypeError("not a storage '%s'", Py_TYPE(obj)->tp_name);
  }

  const auto& storage = THPStorage_Unpack(untyped_storage_obj);
  c10::DeviceType device_type = storage.device().type();
  auto type_properties = get_type_properties(device_type, at::kByte);
  return type_properties->unsafeStorageFromTH(
      storage.unsafeGetStorageImpl(), true);
}

at::Storage createStorage(PyObject* obj) {
  at::ScalarType scalar_type;
  bool is_typed_storage = false;
  return createStorageGetType(obj, scalar_type, is_typed_storage);
}

} // namespace torch
