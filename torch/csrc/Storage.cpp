#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
#include <structmember.h>

#include <ATen/mps/MPSDevice.h>
#include <c10/core/CPUAllocator.h>
#include <libshm.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/StorageMethods.h>
#include <torch/csrc/StorageSharing.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/util/intrusive_ptr.h>
#include <fmt/format.h>

template <>
void THPPointer<c10::StorageImpl>::free() {
  if (ptr) {
    c10::raw::intrusive_ptr::decref(ptr);
  }
}

PyObject* THPStorageClass = nullptr;

PyObject* THPStorage_New(c10::Storage storage) {
  PyTypeObject* type = (PyTypeObject*)THPStorageClass;
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THPStorage*)obj)->cdata =
        c10::MaybeOwned<c10::Storage>::owned(std::move(storage));
  }
  return obj;
}

static void THPStorage_subclass_dealloc(PyObject* self) {
  THPStorage* _self = (THPStorage*)self;
  // Some subclass of StorageBase are GC-tracked objects even
  // though the base class is not.
  auto* type = Py_TYPE(self);
  if (PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC) != 0) {
    PyObject_GC_UnTrack(self);
  }
  _self->cdata.~MaybeOwned<c10::Storage>();
  Py_TYPE(_self)->tp_free(self);
}

c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::Allocator* allocator,
    bool resizable,
    c10::optional<int64_t> allocator_opt,
    c10::optional<at::Device> device_opt) {
  at::OptionalDeviceGuard device_guard;
  // This will be non-nullptr only when there is a custom StorageImpl
  // constructor for the given device
  c10::StorageImplCreateHelper fptr = nullptr;
  // For directly passing allocator scenarios, only c10::StorageImpl objects can
  // be created. If you need to create a storageimpl object of a subclass, you
  // need to pass in the device information.
  if (allocator_opt.has_value()) {
    allocator = reinterpret_cast<c10::Allocator*>(allocator_opt.value());
  } else if (device_opt.has_value()) {
    at::Device device = device_opt.value();
    // We only need to check this here as this is the only case where we can
    // have a device that is not CPU (and thus for which the StorageImpl
    // constructor can be overwritten).
    fptr = c10::GetStorageImplCreate(device.type());
    if (device.type() == at::kCPU) {
      allocator = c10::GetDefaultCPUAllocator();
#ifdef USE_CUDA
    } else if (device.type() == at::kCUDA) {
      at::globalContext().lazyInitCUDA();
      allocator = c10::cuda::CUDACachingAllocator::get();
#endif
#ifdef USE_MPS
    } else if (device.type() == at::kMPS) {
      allocator = at::mps::GetMPSAllocator();
#endif
    } else if (device.type() == at::DeviceType::XPU) {
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::HPU) {
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::Meta) {
      allocator = c10::GetAllocator(device.type());
    } else if (device.type() == at::DeviceType::PrivateUse1) {
      allocator = c10::GetAllocator(device.type());

    } else {
      TORCH_CHECK(
          false,
          THPStorageStr,
          "(): Storage device not recognized: ",
          device.type());
    }
    device_guard.reset_device(device);
  } else {
    allocator = c10::GetDefaultCPUAllocator();
  }

  if (fptr != nullptr) {
    return fptr(use_byte_size, std::move(size_bytes), allocator, resizable);
  }

  // Create a c10::StorageImpl object.
  return c10::make_intrusive<c10::StorageImpl>(
      use_byte_size, std::move(size_bytes), allocator, resizable);
}

static PyObject* THPStorage_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      type != &THPStorageType,
      "Cannot directly construct StorageBase; subclass it and then construct that");
  static torch::PythonArgParser parser({
      THPStorageStr "(*, int64_t allocator=None, Device device=None)",
      THPStorageStr
      "(int64_t size, *, int64_t allocator=None, Device device=None)",
      THPStorageStr
      "(PyObject* sequence, *, int64_t allocator=None, Device device=None)",
  });
  torch::ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  int64_t allocator_arg_idx = 0;
  int64_t device_arg_idx = 1;

  if (r.idx > 0) {
    allocator_arg_idx = 1;
    device_arg_idx = 2;
  }

  c10::optional<int64_t> allocator_opt = r.toInt64Optional(allocator_arg_idx);
  c10::optional<at::Device> device_opt = r.deviceOptional(device_arg_idx);

  TORCH_CHECK(
      !allocator_opt.has_value() || !device_opt.has_value(),
      THPStorageStr,
      "(): only one or neither of 'allocator' or 'device' can ",
      "be given, but not both");

  THPStoragePtr self((THPStorage*)type->tp_alloc(type, 0));
  THPUtils_assert(self, "failed to allocate a " THPStorageStr " object");
  c10::Allocator* allocator = nullptr;

  // torch.Storage(*, ...)
  if (r.idx == 0) {
    self->cdata = c10::MaybeOwned<c10::Storage>::owned(make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        0,
        allocator,
        /*resizable=*/true,
        allocator_opt,
        device_opt));
    return (PyObject*)self.release();

    // torch.Storage(size, *, ...)
  } else if (r.idx == 1) {
    int64_t size = r.toInt64(0);
    self->cdata = c10::MaybeOwned<c10::Storage>::owned(make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        size,
        allocator,
        /*resizable=*/true,
        allocator_opt,
        device_opt));
    return (PyObject*)self.release();

    // torch.Storage(sequence, *, ...)
  } else if (r.idx == 2) {
    PyObject* sequence = r.pyobject(0);
    Py_ssize_t length = PySequence_Length(sequence);
    TORCH_CHECK(
        PySequence_Check(sequence),
        THPStorageStr,
        "(): Expected a sequence type, but got ",
        THPUtils_typename(sequence));
    TORCH_CHECK(
        length >= 0,
        THPStorageStr,
        "(): Could not obtain the length of sequence of type ",
        THPUtils_typename(sequence));
    self->cdata = c10::MaybeOwned<c10::Storage>::owned(make_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        length,
        allocator,
        /*resizable=*/true,
        allocator_opt,
        device_opt));
    THPObjectPtr item;
    try {
      for (Py_ssize_t i = 0; i < length; i++) {
        item = PySequence_GetItem(sequence, i);
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        uint8_t value = THPByteUtils_unpackReal(item.get());
        const auto& storage = THPStorage_Unpack(self);
        if (allocator == c10::GetDefaultCPUAllocator()) {
          static_cast<uint8_t*>(storage.mutable_data())[i] = value;
        } else {
          // TODO: this might be slow - consider batched updates?
          storage_set(storage, i, value);
        }
      }
    } catch (const std::exception& e) {
      THPUtils_setError(
          THPStorageStr
          "(): tried to construct a storage from a sequence (%s), "
          "but one of the items was of type %s instead of int",
          THPUtils_typename(sequence),
          THPUtils_typename(item.get()));
      return nullptr;
    }
    return (PyObject*)self.release();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPStorage_length(THPStorage* self) {
  HANDLE_TH_ERRORS
  return THPStorage_Unpack(self).nbytes();
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPStorage_get(THPStorage* self, PyObject* index) {
  HANDLE_TH_ERRORS
  const auto& storage = THPStorage_Unpack(self);
  /* Integer index */
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    if (nindex < 0)
      nindex += storage.nbytes();
    if (nindex < 0 || nindex >= static_cast<int64_t>(storage.nbytes())) {
      PyErr_SetString(
          PyExc_IndexError,
          fmt::format(
              "index {} out of range for storage of size {}",
              nindex,
              storage.nbytes()));
      return nullptr;
    }
    uint8_t value = storage_get(storage, nindex);
    return THPByteUtils_newReal(value);
    /* Slice index */
  } else if (PySlice_Check(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, slicelength, step;
    int64_t len = storage.nbytes();
    if (PySlice_GetIndicesEx(index, len, &start, &stop, &step, &slicelength) !=
        0)
      return nullptr;
    if (step != 1) {
      THPUtils_setError(
          "Trying to slice with a step of %lld, but only a step of "
          "1 is supported",
          (long long)step);
      return nullptr;
    }

    const auto& storage = THPStorage_Unpack(self);
    auto data = static_cast<uint8_t*>(storage.mutable_data());

    at::StorageImpl* old_storage_impl = storage.unsafeGetStorageImpl();
    c10::raw::intrusive_ptr::incref(old_storage_impl);
    auto new_storage_impl = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
#ifdef THQUANTIZED
        slicelength * sizeof(quantized_t),
#else
        slicelength,
#endif
        at::DataPtr(
            static_cast<void*>(data + start),
            old_storage_impl,
            [](void* s) {
              c10::raw::intrusive_ptr::decref(static_cast<at::StorageImpl*>(s));
            },
            old_storage_impl->device()),
        old_storage_impl->allocator(),
        /* resizable */ false);

    PyObject* _ret = THPStorage_New(std::move(new_storage_impl));
    return _ret;
  }
  PyErr_Format(
      PyExc_TypeError,
      "can't index a " THPStorageStr " with %s",
      THPUtils_typename(index));
  return nullptr;
  END_HANDLE_TH_ERRORS
}

static int THPStorage_set(THPStorage* self, PyObject* index, PyObject* value) {
  HANDLE_TH_ERRORS
  if (!THPByteUtils_checkReal(value)) {
    THPUtils_setError(
        "can only set storage content with a int types, but got "
        "%s instead",
        THPUtils_typename(value));
    return -1;
  }

  uint8_t rvalue = THPByteUtils_unpackReal(value);
  const auto& storage = THPStorage_Unpack(self);
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    storage_set(storage, nindex, rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start, stop, slicelength, step;
    int64_t len = storage.nbytes();
    if (PySlice_GetIndicesEx(index, len, &start, &stop, &step, &slicelength) !=
        0)
      return -1;
    if (step != 1) {
      THPUtils_setError(
          "Trying to slice with a step of %lld, but only a step of "
          "1 is supported",
          (long long)step);
      return 0;
    }
    // TODO: check the bounds only once
    // TODO: fill?
    for (; start < stop; start++)
      storage_set(storage, start, rvalue);
    return 0;
  }
  THPUtils_setError(
      "can't index a " THPStorageStr " with %s", THPUtils_typename(index));
  return -1;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyMappingMethods THPStorage_mappingmethods = {
    (lenfunc)THPStorage_length,
    (binaryfunc)THPStorage_get,
    (objobjargproc)THPStorage_set};

struct THPStorageMeta {
  PyHeapTypeObject base;
};

int THPStorageMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs);

PyTypeObject THPStorageMetaType = {
    PyVarObject_HEAD_INIT(
        DEFERRED_ADDRESS(&PyType_Type),
        0) "torch._C._StorageMeta", /* tp_name */
    sizeof(THPStorageMeta), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    DEFERRED_ADDRESS(&PyType_Type), /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    THPStorageMetaType_init, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

// TODO: implement equality
PyTypeObject THPStorageType = {
    PyVarObject_HEAD_INIT(
        &THPStorageMetaType,
        0) "torch._C.StorageBase", /* tp_name */
    sizeof(THPStorage), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THPStorage_mappingmethods, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr,
    /* will be assigned in init */ /* tp_methods */
    nullptr,
    /* will be assigned in init */ /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPStorage_pynew, /* tp_new */
};

int THPStorageMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs) {
  if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
    return -1;
  }
  ((PyTypeObject*)cls)->tp_dealloc = (destructor)THPStorage_subclass_dealloc;
  return 0;
}

static PyObject* THPStorage_device(THPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(THPStorage_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_get_cdata(THPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(THPStorage_Unpack(self).unsafeGetStorageImpl());
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPStorage_properties[] = {
    {"device", (getter)THPStorage_device, nullptr, nullptr, nullptr},
    {"_cdata", (getter)THPStorage_get_cdata, nullptr, nullptr, nullptr},
    {nullptr}};

bool THPStorage_init(PyObject* module) {
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THPStorage_getMethods());
  THPUtils_addPyMethodDefs(methods, THPStorage_getSharingMethods());

  THPStorageMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPStorageMetaType) < 0)
    return false;
  Py_INCREF(&THPStorageMetaType);
  PyModule_AddObject(module, "_StorageMeta", (PyObject*)&THPStorageMetaType);

  THPStorageType.tp_methods = methods.data();
  THPStorageType.tp_getset = THPStorage_properties;
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, "StorageBase", (PyObject*)&THPStorageType);
  return true;
}

void THPStorage_postInit(PyObject* module) {
  THPStorageClass = PyObject_GetAttrString(module, "UntypedStorage");
  if (!THPStorageClass)
    throw python_error();
}
