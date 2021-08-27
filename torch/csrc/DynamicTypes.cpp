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

#ifdef USE_CUDA
#include <THC/THC.h>
#endif

namespace torch {
namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<at::DeprecatedTypeProperties*, PyTypeObject*> attype_to_py_storage_type;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<PyTypeObject*, at::DeprecatedTypeProperties*> py_storage_type_to_attype;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::array<THPDtype*, static_cast<int>(at::ScalarType::NumOptions)> dtype_registry = {};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

PyTypeObject* getPyTypeObject(
    const at::Storage& storage,
    const caffe2::TypeMeta dtype) {
  // TODO: https://github.com/pytorch/pytorch/issues/47442
  if (storage.device_type() == at::DeviceType::Meta) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "python bindings for meta storage objects not supported");
  }
  at::ScalarType scalarType = at::ScalarType::Byte;
  auto attype = &at::getDeprecatedTypeProperties(
      at::dispatchKeyToBackend(c10::computeDispatchKey(scalarType, c10::nullopt, storage.device_type())),
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

PyObject* createPyObject(
    const at::Storage& storage,
    const caffe2::TypeMeta data_type) {
  auto type = getPyTypeObject(storage, data_type);
  auto obj = THPObjectPtr(type->tp_alloc(type, 0));
  if (!obj) throw python_error();
  ((THPVoidStorage*)obj.get())->cdata = (THVoidStorage *)at::Storage(/* copy */ storage).unsafeReleaseStorageImpl();
  return obj.release();
}

struct THPTypedStorage {
  PyObject_HEAD
};

static PyTypeObject THPTypedStorageType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "torch._C.TypedStorage",
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "THPTypedStorage",
    .tp_new = PyType_GenericNew,
};


bool initTHPTypedStorageType(PyObject* module) {
  if (PyType_Ready(&THPTypedStorageType) < 0)
    return false;
  Py_INCREF(&THPTypedStorageType);
  PyModule_AddObject(module, "TypedStorage",   (PyObject *)&THPTypedStorageType);
  return true;
}

bool isStorage(PyObject* obj)
{
  // TODO: Consider instead importing torch.storage with PyImport_ImportModule
  // and then getting TypedStorage directly from there, rather than using this
  // torch._C._TypedStorage subclass trick
  if (PyObject_TypeCheck(obj, &torch::THPTypedStorageType)) {
    // TODO: Maybe throw error if type is exactly THPTypedStorageType
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

at::Storage createStorage(PyObject* obj)
{
  auto obj_type = Py_TYPE(obj);
  for (auto const& item : py_storage_type_to_attype) {
    auto const& storage_type = item.first;
    if (obj_type == storage_type) {
      auto& type = *item.second;
      return type.unsafeStorageFromTH(((THPVoidStorage*)obj)->cdata, true);
    }
    // Check for TypedStorage, which has a `storage` attribute that matches
    // TODO: This check should only be performed once, outside of this loop
    // TODO: Consider instead importing torch.storage with PyImport_ImportModule
    // and then getting TypedStorage directly from there, rather than using this
    // torch._C._TypedStorage subclass trick
    if (PyObject_TypeCheck(obj, &torch::THPTypedStorageType)) {
      // TODO: Should probably throw error if type is exactly THPTypedStorageType
      PyObject* maybe_storage = PyObject_GetAttrString(obj, "_storage");
      if (maybe_storage && (Py_TYPE(maybe_storage) == storage_type)) {
        auto& type = *item.second;
        auto ret = type.unsafeStorageFromTH(((THPVoidStorage*)maybe_storage)->cdata, true);
        return ret;
      }
    }
  }
  throw TypeError("not a storage '%s'", Py_TYPE(obj)->tp_name);
}

}  // namespace
