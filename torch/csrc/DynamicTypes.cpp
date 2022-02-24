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

#include <array>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace {
std::unordered_map<at::DeprecatedTypeProperties*, PyTypeObject*> attype_to_py_storage_type;
std::unordered_map<PyTypeObject*, at::DeprecatedTypeProperties*> py_storage_type_to_attype;

std::array<THPDtype*, static_cast<int>(at::ScalarType::NumOptions)> dtype_registry = {};

std::array<THPLayout*, static_cast<int>(at::Layout::NumOptions)> layout_registry = {};

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

PyTypeObject* getPyTypeObject(const at::Storage& storage) {
  // TODO: https://github.com/pytorch/pytorch/issues/47442
  if (storage.device_type() == at::DeviceType::Meta) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "python bindings for meta storage objects not supported");
  }
  if (storage.data() == nullptr && storage.nbytes() != 0) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "python bindings to nullptr storage (e.g., from torch.Tensor._make_wrapper_subclass) are currently unsafe and thus disabled.  See https://github.com/pytorch/pytorch/issues/61669 for more details");
  }
  at::ScalarType scalarType = at::ScalarType::Byte;
  auto attype = &at::getDeprecatedTypeProperties(
      at::dispatchKeyToBackend(c10::computeDispatchKey(scalarType, c10::nullopt, storage.device_type())),
      scalarType);
  auto it = attype_to_py_storage_type.find(attype);
  TORCH_INTERNAL_ASSERT(it != attype_to_py_storage_type.end(),
        "Failed to get the Python type of `_UntypedStorage`.");
  return it->second;
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

void registerLayoutObject(THPLayout *thp_layout, at::Layout layout) {
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
  auto type = getPyTypeObject(storage);
  auto obj = THPObjectPtr(type->tp_alloc(type, 0));
  if (!obj) throw python_error();
  ((THPVoidStorage*)obj.get())->cdata = at::Storage(/* copy */ storage).unsafeReleaseStorageImpl();
  return obj.release();
}

PyTypeObject* loadTypedStorageTypeObject() {
  PyObject* storage_module = PyImport_ImportModule("torch.storage");
  TORCH_INTERNAL_ASSERT(storage_module && PyModule_Check(storage_module));

  PyObject* typed_storage_obj = PyObject_GetAttrString(storage_module, "_TypedStorage");
  TORCH_INTERNAL_ASSERT(typed_storage_obj && PyType_Check(typed_storage_obj));
  return reinterpret_cast<PyTypeObject*>(
      PyObject_GetAttrString(storage_module, "_TypedStorage"));
}

PyTypeObject* getTypedStorageTypeObject() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static PyTypeObject* typed_storage_type_obj = loadTypedStorageTypeObject();
  return typed_storage_type_obj;
}

bool isStorage(PyObject* obj)
{
  if (PyObject_TypeCheck(obj, getTypedStorageTypeObject())) {
    return true;
  }
  auto obj_type = Py_TYPE(obj);
  for (auto const& item : py_storage_type_to_attype) {
    auto const& storage_type = item.first;
    if (obj_type == storage_type) {
      return true;
    }
  }
  return false;
}

at::Storage createStorageGetType(PyObject* obj, at::ScalarType& scalar_type, bool& is_typed_storage)
{
  is_typed_storage = PyObject_TypeCheck(obj, getTypedStorageTypeObject());
  THPObjectPtr maybe_untyped_storage;
  if (is_typed_storage) {
    PyObject* maybe_untyped_storage_obj = PyObject_GetAttrString(obj, "_storage");
    TORCH_INTERNAL_ASSERT(maybe_untyped_storage_obj);
    maybe_untyped_storage = maybe_untyped_storage_obj;
  }

  auto obj_type = Py_TYPE(obj);
  for (auto const& item : py_storage_type_to_attype) {
    auto const& storage_type = item.first;
    if (is_typed_storage) {
      if (Py_TYPE(maybe_untyped_storage.get()) == storage_type) {
        auto& type = *item.second;
        auto ret = type.unsafeStorageFromTH(
          ((THPVoidStorage*)maybe_untyped_storage.get())->cdata,
          true);
        PyObject* dtype_obj = PyObject_GetAttrString(obj, "dtype");
        TORCH_INTERNAL_ASSERT(dtype_obj && THPDtype_Check(dtype_obj));
        scalar_type = reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
        return ret;
      }
    }
    if (obj_type == storage_type) {
      auto& type = *item.second;
      // _UntypedStorage should always be interpreted with byte dtype
      scalar_type = at::kByte;
      return type.unsafeStorageFromTH(((THPVoidStorage*)obj)->cdata, true);
    }
  }
  throw TypeError("not a storage '%s'", Py_TYPE(obj)->tp_name);
}

at::Storage createStorage(PyObject* obj) {
  at::ScalarType scalar_type;
  bool is_typed_storage = false;
  return createStorageGetType(obj, scalar_type, is_typed_storage);
}

}  // namespace
