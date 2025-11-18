#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
#include <structmember.h>

#include <ATen/mps/MPSDevice.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/RefcountedDeleter.h>
#include <libshm.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/StorageMethods.h>
#include <torch/csrc/StorageSharing.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pyobject_preservation.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/util/intrusive_ptr.h>
#include <fmt/format.h>

using torch::utils::PyObjectPreservation;

template <>
void THPPointer<c10::StorageImpl>::free() {
  if (ptr) {
    c10::raw::intrusive_ptr::decref(ptr);
  }
}

PyTypeObject* THPStorageClass = nullptr;

// Create a new Python Storage object, but don't set the pyobj slot on the
// c10::Storage object.
static PyObject* THPStorage_New(PyTypeObject* type, c10::Storage _storage) {
  PyObject* obj = type->tp_alloc(type, 0);
  TORCH_CHECK(obj, "Failed to allocate a ", type->tp_name, " object");

  // Ensure that PyUnstable_TryIncref calls don't fail spuriously in
  // free-threaded Python.
  PyUnstable_EnableTryIncRef(obj);

  auto s = (THPStorage*)obj;
  new (&s->cdata) c10::Storage(std::move(_storage));
  return obj;
}

// Create a new Python Storage object for a new c10::Storage, and set the
// pyobj slot. The c10::Storage must not already have a pyobj set.
PyObject* THPStorage_NewWithStorage(PyTypeObject* type, c10::Storage _storage) {
  TORCH_CHECK(
      type == THPStorageClass || PyType_IsSubtype(type, &THPStorageType),
      "Creating a Storage subclass from a class that does not inherit from ",
      "Storage is not possible. Make sure your class inherits from Storage.");
  TORCH_INTERNAL_ASSERT(_storage.use_count() == 1);

  c10::StorageImpl* storage_impl = _storage.unsafeGetStorageImpl();
  PyObject* obj = THPStorage_New(type, std::move(_storage));
  PyObjectPreservation::init_fresh_nonatomic(
      storage_impl, storage_impl->pyobj_slot(), obj);
  return obj;
}

// Returns a PyObject wrapper for the c10::Storage object. The existing
// wrapper is returned if it already exists.
PyObject* THPStorage_Wrap(c10::Storage storage) {
  if (c10::impl::HermeticPyObjectTLS::get_state()) {
    return THPStorage_New(THPStorageClass, std::move(storage));
  }

  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
  c10::impl::PyObjectSlot* pyobj_slot = storage_impl->pyobj_slot();

  PyObject* obj = pyobj_slot->load_pyobj();
  if (obj) {
    return Py_NewRef(obj);
  }

  obj = THPStorage_New(THPStorageClass, std::move(storage));
  PyObject* wrapper =
      PyObjectPreservation::init_once(storage_impl, pyobj_slot, obj);
  if (wrapper != obj) {
    // Another thread beat us to it
    Py_DECREF(obj);
    return Py_NewRef(wrapper);
  }
  return obj;
}

static void THPStorage_dealloc(PyObject* self) {
  THPStorage* _self = reinterpret_cast<THPStorage*>(self);
  auto pyobj_slot = _self->cdata.unsafeGetStorageImpl()->pyobj_slot();
  if (pyobj_slot->load_pyobj() == self) {
    TORCH_INTERNAL_ASSERT(_self->cdata.use_count() == 1);
    pyobj_slot->clear();
  }
  _self->cdata.~Storage();
  Py_TYPE(_self)->tp_free(self);
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

  int allocator_arg_idx = 0;
  int device_arg_idx = 1;

  if (r.idx > 0) {
    allocator_arg_idx = 1;
    device_arg_idx = 2;
  }

  std::optional<int64_t> allocator_opt = r.toInt64Optional(allocator_arg_idx);
  std::optional<at::Device> device_opt = r.deviceOptional(device_arg_idx);

  TORCH_CHECK(
      !allocator_opt.has_value() || !device_opt.has_value(),
      THPStorageStr,
      "(): only one or neither of 'allocator' or 'device' can ",
      "be given, but not both");

  PyObject* self = nullptr;
  c10::Allocator* allocator = nullptr;
  at::OptionalDeviceGuard device_guard;

  if (allocator_opt.has_value()) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    allocator = reinterpret_cast<c10::Allocator*>(allocator_opt.value());
  } else if (device_opt.has_value()) {
    at::Device device = device_opt.value();
    torch::utils::maybe_initialize_device(device);

    switch (device.type()) {
      case at::kCPU:
        allocator = c10::GetDefaultCPUAllocator();
        break;
#ifdef USE_CUDA
      case at::kCUDA:
        allocator = c10::cuda::CUDACachingAllocator::get();
        break;
#endif
#ifdef USE_MPS
      case at::kMPS:
        allocator = at::mps::GetMPSAllocator();
        break;
#endif
      case at::DeviceType::XPU:
      case at::DeviceType::HPU:
      case at::DeviceType::Meta:
      case at::DeviceType::PrivateUse1:
      case at::DeviceType::MAIA:
      case at::DeviceType::MTIA:
        allocator = c10::GetAllocator(device.type());
        break;
      default:
        // NOLINTEND(bugprone-branch-clone)
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

  // torch.Storage(*, ...)
  if (r.idx == 0) {
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            0,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt));

    // torch.Storage(size, *, ...)
  } else if (r.idx == 1) {
    int64_t size = r.toInt64(0);
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            size,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt));

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
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            length,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt));
    THPObjectPtr item;
    try {
      const auto& storage = THPStorage_Unpack(self);
      for (Py_ssize_t i = 0; i < length; i++) {
        item = PySequence_GetItem(sequence, i);
        uint8_t value = THPByteUtils_unpackReal(item.get());
        if (allocator == c10::GetDefaultCPUAllocator()) {
          static_cast<uint8_t*>(storage.mutable_data())[i] = value;
        } else {
          // TODO: this might be slow - consider batched updates?
          storage_set(storage, i, value);
        }
      }
    } catch (const std::exception& e) {
      TORCH_CHECK(
          THPStorageStr "(): tried to construct a storage from a sequence (",
          THPUtils_typename(sequence),
          "), ",
          "but one of the items was of type ",
          THPUtils_typename(item.get()),
          " instead of int");
      return nullptr;
    }
  }
  return self;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPStorage_length(THPStorage* self) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  return static_cast<Py_ssize_t>(THPStorage_Unpack(self).nbytes());
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPStorage_get(THPStorage* self, PyObject* index) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  int64_t len = static_cast<int64_t>(storage.nbytes());
  /* Integer index */
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    if (nindex < 0)
      nindex += len;
    if (nindex < 0 || nindex >= len) {
      PyErr_SetString(
          PyExc_IndexError,
          fmt::format(
              "index {} out of range for storage of size {}", nindex, len));
      return nullptr;
    }
    uint8_t value = storage_get(storage, nindex);
    return THPUtils_packUInt32(value);
    /* Slice index */
  } else if (PySlice_Check(index)) {
    Py_ssize_t start = 0, stop = 0, slicelength = 0, step = 0;
    if (PySlice_Unpack(index, &start, &stop, &step) < 0) {
      return nullptr;
    }
    slicelength = PySlice_AdjustIndices(len, &start, &stop, step);
    if (step != 1) {
      TORCH_CHECK(
          "Trying to slice with a step of ",
          step,
          ", but only a step of "
          "1 is supported");
      return nullptr;
    }

    const auto& storage = THPStorage_Unpack(self);
    auto data = static_cast<uint8_t*>(storage.mutable_data());

    at::StorageImpl* old_storage_impl = storage.unsafeGetStorageImpl();
    c10::raw::intrusive_ptr::incref(old_storage_impl);
    std::optional<at::Device> device_opt = old_storage_impl->device();
    auto new_storage_impl = make_storage_impl(
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
        /* resizable */ false,
        device_opt);

    PyObject* _ret =
        THPStorage_NewWithStorage(Py_TYPE(self), std::move(new_storage_impl));

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
  THPStorage_assertNotNull(self);
  if (!THPByteUtils_checkReal(value)) {
    TORCH_CHECK(
        "can only set storage content with a int types, but got ",
        THPUtils_typename(value),
        " instead");
    return -1;
  }

  uint8_t rvalue = THPByteUtils_unpackReal(value);
  const auto& storage = THPStorage_Unpack(self);
  if (THPUtils_checkLong(index)) {
    int64_t nindex = THPUtils_unpackLong(index);
    storage_set(storage, nindex, rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start = 0, stop = 0, step = 0;
    Py_ssize_t len = static_cast<Py_ssize_t>(storage.nbytes());
    if (PySlice_Unpack(index, &start, &stop, &step) < 0) {
      return -1;
    }
    PySlice_AdjustIndices(len, &start, &stop, step);
    if (step != 1) {
      TORCH_CHECK(
          "Trying to slice with a step of ",
          step,
          ", but only a step of "
          "1 is supported");
      return 0;
    }
    // TODO: check the bounds only once
    // TODO: fill?
    for (; start < stop; start++)
      storage_set(storage, start, rvalue);
    return 0;
  }
  TORCH_CHECK(
      "can't index a " THPStorageStr " with ", THPUtils_typename(index));
  return -1;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyMappingMethods THPStorage_mappingmethods = {
    reinterpret_cast<lenfunc>(THPStorage_length),
    reinterpret_cast<binaryfunc>(THPStorage_get),
    reinterpret_cast<objobjargproc>(THPStorage_set)};

// TODO: implement equality
PyTypeObject THPStorageType = {
    PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
    "torch._C.StorageBase", /* tp_name */
    sizeof(THPStorage), /* tp_basicsize */
    0, /* tp_itemsize */
    THPStorage_dealloc, /* tp_dealloc */
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
    // NOLINTNEXTLINE(misc-redundant-expression)
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

static PyObject* THPStorage_device(THPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  return THPDevice_New(THPStorage_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_get_cdata(THPStorage* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(THPStorage_Unpack(self).unsafeGetStorageImpl());
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPStorage_properties[] = {
    {"device",
     reinterpret_cast<getter>(THPStorage_device),
     nullptr,
     nullptr,
     nullptr},
    {"_cdata",
     reinterpret_cast<getter>(THPStorage_get_cdata),
     nullptr,
     nullptr,
     nullptr},
    {nullptr}};

bool THPStorage_init(PyObject* module) {
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, THPStorage_getMethods());
  THPUtils_addPyMethodDefs(methods, THPStorage_getSharingMethods());

  THPStorageType.tp_methods = methods.data();
  THPStorageType.tp_getset = THPStorage_properties;
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(
      module, "StorageBase", reinterpret_cast<PyObject*>(&THPStorageType));
  return true;
}

void THPStorage_postInit(PyObject* module) {
  THPStorageClass = reinterpret_cast<PyTypeObject*>(
      PyObject_GetAttrString(module, "UntypedStorage"));
  if (!THPStorageClass)
    throw python_error();
}

void THPStorage_assertNotNull(THPStorage* storage) {
  TORCH_CHECK(
      THPStorage_Unpack(storage).unsafeGetStorageImpl(), "Got a null Storage");
}

void THPStorage_assertNotNull(PyObject* obj) {
  THPStorage_assertNotNull(reinterpret_cast<THPStorage*>(obj));
}
