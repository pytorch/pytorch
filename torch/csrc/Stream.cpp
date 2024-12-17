#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <structmember.h>
#include <cstdint>

PyTypeObject* THPStreamClass = nullptr;

static PyObject* THPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  int64_t stream_id = -1;
  c10::DeviceType device_type{};
  c10::DeviceIndex device_index{};
  int64_t priority = 0;

  static torch::PythonArgParser parser({
      "Stream(Device device=None, *, int64_t priority=0)",
      "Stream(int64_t stream_id, int64_t device_index, int64_t device_type, *, int64_t priority=0)",
  });

  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  std::unique_ptr<c10::DeviceGuard> device_guard_ptr;

  if (r.idx == 0) {
    auto default_accelerator = at::getAccelerator(false);
    auto device = r.deviceOptional(0);
    if (device.has_value()) {
      device_type = device->type();
      device_index = device->index();
      // Initialize device guard if device is not None.
      device_guard_ptr = std::make_unique<c10::DeviceGuard>(device.value());
    } else {
      // If device is None, we will use the current accelerator and index.
      // If the current accelerator is not set, we will use the CPU as device
      // type.
      device_type = default_accelerator.value_or(c10::DeviceType::CPU);
      c10::impl::VirtualGuardImpl impl{device_type};
      const auto current_device = impl.getDevice();
      device_index = current_device.index();
    }
    priority = r.toInt64WithDefault(1, 0);
  } else if (r.idx == 1) {
    stream_id = r.toInt64WithDefault(0, -1);
    device_index = static_cast<c10::DeviceIndex>(r.toInt64WithDefault(1, 0));
    device_type = static_cast<c10::DeviceType>(
        r.toInt64WithDefault(2, static_cast<int64_t>(c10::DeviceType::CPU)));
    priority = r.toInt64WithDefault(3, 0);
  } else {
    TORCH_CHECK(
        false,
        "parse stream arg fails please check the usage: ",
        parser.get_signatures());
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THPStream* self = (THPStream*)ptr.get();

  // If torch.Stream is not created from existing Stream, then create a new one.
  // It requires other device backends override getNewStream method. How the new
  // stream is created is backend specific. Backend should be able to correctly
  // manage the lifetime of streams.
  std::optional<c10::Stream> stream_opt;
  if (r.idx == 0) {
    c10::impl::VirtualGuardImpl impl{device_type};
    stream_opt = impl.getNewStream(
        c10::Device(device_type, device_index), static_cast<int>(priority));
  } else {
    stream_opt = c10::Stream::unpack3(stream_id, device_index, device_type);
  }

  TORCH_CHECK(stream_opt.has_value(), "Failed to create stream");
  self->stream_id = static_cast<int64_t>(stream_opt->id());
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(stream_opt->device_index());
  self->device_type = static_cast<int64_t>(stream_opt->device_type());
  self->context = nullptr;

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPStream_Wrap(const c10::Stream& stream) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)THPStreamClass;
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    throw python_error();
  }

  THPStream* self = (THPStream*)ptr.get();
  self->stream_id = stream.id();
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  self->context = nullptr;
  return ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPStream_dealloc(THPStream* self) {
  if (self->context) {
    Py_DECREF(self->context);
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THPStream_get_device(THPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(c10::Device(
      static_cast<c10::DeviceType>(self->device_type),
      static_cast<c10::DeviceIndex>(self->device_index)));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;

  return PyBool_FromLong(c10::Stream::unpack3(
                             self->stream_id,
                             static_cast<c10::DeviceIndex>(self->device_index),
                             static_cast<c10::DeviceType>(self->device_type))
                             .query());

  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto self = (THPStream*)_self;

    c10::Stream::unpack3(
        self->stream_id,
        static_cast<c10::DeviceIndex>(self->device_index),
        static_cast<c10::DeviceType>(self->device_type))
        .synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_wait_event(PyObject* _self, PyObject* _event) {
  HANDLE_TH_ERRORS {
    auto self = (THPStream*)_self;
    auto event = (THPEvent*)_event;
    c10::Stream::unpack3(
        self->stream_id,
        static_cast<c10::DeviceIndex>(self->device_index),
        static_cast<c10::DeviceType>(self->device_type))
        .wait(event->event);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_wait_stream(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS {
    auto self = (THPStream*)_self;
    auto other_stream = (THPStream*)_other;
    c10::Event new_event(
        static_cast<c10::DeviceType>(other_stream->device_type),
        c10::EventFlag::PYTORCH_DEFAULT);
    new_event.record(c10::Stream::unpack3(
        other_stream->stream_id,
        static_cast<c10::DeviceIndex>(other_stream->device_index),
        static_cast<c10::DeviceType>(other_stream->device_type)));
    c10::Stream::unpack3(
        self->stream_id,
        static_cast<c10::DeviceIndex>(self->device_index),
        static_cast<c10::DeviceType>(self->device_type))
        .wait(new_event);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_record_event(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;
  PyObject* _new_event = nullptr;
  PyObject* _event = Py_None;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* accepted_args[] = {"event", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|O",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(accepted_args),
          &_event)) {
    TORCH_CHECK(false, "parse record_event arg fails");
  }
  if (_event != Py_None) {
    // Increase the refcount of the event to avoid it being destroyed.
    Py_INCREF(_event);
    _new_event = _event;
  } else {
    _new_event = THPEvent_new(
        static_cast<c10::DeviceType>(self->device_type),
        c10::EventFlag::PYTORCH_DEFAULT);
  }
  auto new_event = (THPEvent*)_new_event;
  TORCH_CHECK(new_event, "event must not be null");
  new_event->event.record(c10::Stream::unpack3(
      self->stream_id,
      static_cast<c10::DeviceIndex>(self->device_index),
      static_cast<c10::DeviceType>(self->device_type)));
  return (PyObject*)new_event;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_repr(THPStream* self) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(
      "torch.Stream device_type=" +
      c10::DeviceTypeName(
          static_cast<c10::DeviceType>(self->device_type), true) +
      ", device_index=" + std::to_string(self->device_index) +
      ", stream_id=" + std::to_string(self->stream_id));
  END_HANDLE_TH_ERRORS
}

static Py_hash_t THPStream_hash(THPStream* self) {
  return static_cast<long>(at::hash_combine(
      self->device_type,
      (at::hash_combine(self->stream_id, self->device_index))));
}

static PyObject* THPStream_eq(THPStream* self, THPStream* other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(
      (self->stream_id == other->stream_id) &&
      (self->device_index == other->device_index) &&
      (self->device_type == other->device_type));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_enter(PyObject* _self, PyObject* unused) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;
  c10::DeviceType stream_device_type =
      static_cast<c10::DeviceType>(self->device_type);
  // No operation is performed if the stream does not belong to an accelerator.
  if (C10_UNLIKELY(!at::accelerator::isAccelerator(stream_device_type))) {
    // Support with torch.Stream() as stream: works
    Py_INCREF(_self);
    return _self;
  }
  c10::DeviceIndex cur_device_idx = at::accelerator::getDeviceIndex();
  c10::DeviceIndex stream_device_idx =
      static_cast<c10::DeviceIndex>(self->device_index);
  // If the stream is not on the current device, switch the current device to
  // the device of the stream.
  if (stream_device_idx != cur_device_idx) {
    at::accelerator::setDeviceIndex(stream_device_idx);
  }
  c10::Stream cur_stream = at::accelerator::getCurrentStream(stream_device_idx);
  at::accelerator::setCurrentStream(c10::Stream::unpack3(
      self->stream_id, stream_device_idx, stream_device_type));
  // Save the current device index and previous stream to the context.
  auto ctx_device_index = py::reinterpret_steal<py::object>(
      THPUtils_packDeviceIndex(cur_device_idx));
  auto ctx_stream =
      py::reinterpret_steal<py::object>(THPStream_Wrap(cur_stream));
  PyObject_SetAttrString(_self, "_ctx_device_index", ctx_device_index.ptr());
  PyObject_SetAttrString(_self, "_ctx_stream", ctx_stream.ptr());
  // Support with torch.Stream() as stream: works
  Py_INCREF(_self);
  return _self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_exit(PyObject* _self, PyObject* unused) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;
  // No operation is performed if the stream does not belong to an accelerator.
  if (C10_UNLIKELY(!at::accelerator::isAccelerator(
          static_cast<c10::DeviceType>(self->device_type)))) {
    Py_RETURN_NONE;
  }
  auto ctx_stream = THPObjectPtr(PyObject_GetAttrString(_self, "_ctx_stream"));
  auto ctx_device_index =
      THPObjectPtr(PyObject_GetAttrString(_self, "_ctx_device_index"));
  if ((!ctx_stream) || (!ctx_device_index)) {
    throw python_error();
  }
  Py_DECREF(ctx_stream.get());
  Py_DECREF(ctx_device_index.get());
  THPStream* prev_stream = (THPStream*)ctx_stream.release();
  auto prev_device_index =
      THPUtils_unpackDeviceIndex(ctx_device_index.release());
  at::accelerator::setCurrentStream(c10::Stream::unpack3(
      prev_stream->stream_id,
      static_cast<c10::DeviceIndex>(prev_stream->device_index),
      static_cast<c10::DeviceType>(prev_stream->device_type)));
  // Reset the current device to the previous device if they differ.
  if (static_cast<c10::DeviceIndex>(self->device_index) != prev_device_index) {
    at::accelerator::setDeviceIndex(prev_device_index);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static void THPStream_maybe_init_context(THPStream* self) {
  if (!(self->context)) {
    auto dict = THPObjectPtr(PyDict_New());
    if (!dict) {
      throw python_error();
    }
    self->context = dict.release();
  }
}

static PyObject* THPStream_getattro(PyObject* _self, PyObject* attr_name) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;
  auto attr_name_str = THPUtils_unpackString(attr_name);
  if (attr_name_str == "_ctx_device_index" || attr_name_str == "_ctx_stream") {
    THPStream_maybe_init_context(self);
    // borrowed reference
    auto value = THPObjectPtr(PyDict_GetItem(self->context, attr_name));
    Py_INCREF(value.get());
    return value.release();
  }
  return PyObject_GenericGetAttr(_self, attr_name);
  END_HANDLE_TH_ERRORS
}

static int THPStream_setattro(
    PyObject* _self,
    PyObject* attr_name,
    PyObject* value) {
  HANDLE_TH_ERRORS
  auto self = (THPStream*)_self;
  auto attr_name_str = THPUtils_unpackString(attr_name);
  if (attr_name_str == "_ctx_device_index" || attr_name_str == "_ctx_stream") {
    THPStream_maybe_init_context(self);
    return PyDict_SetItem(self->context, attr_name, value);
  }
  return PyObject_GenericSetAttr(_self, attr_name, value);
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPStream_ne(THPStream* self, THPStream* other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(
      (self->stream_id != other->stream_id) ||
      (self->device_index != other->device_index) ||
      (self->device_type != other->device_type));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_richcompare(
    PyObject* self,
    PyObject* other,
    int op) {
  PyObject* result = nullptr;
  if (other == Py_None) {
    result = Py_False;
  } else {
    switch (op) {
      case Py_EQ:
        result = THPStream_eq((THPStream*)self, (THPStream*)other);
        break;
      case Py_NE:
        result = THPStream_ne((THPStream*)self, (THPStream*)other);
        break;
      default:
        result = Py_False;
        break;
    }
  }
  Py_XINCREF(result);
  return result;
}

static const std::initializer_list<PyMemberDef> THPStream_members = {
    {"stream_id",
     T_LONGLONG,
     offsetof(THPStream, stream_id),
     READONLY,
     nullptr},
    {"device_index",
     T_LONGLONG,
     offsetof(THPStream, device_index),
     READONLY,
     nullptr},
    {"device_type",
     T_LONGLONG,
     offsetof(THPStream, device_type),
     READONLY,
     nullptr},
    {nullptr}};

static const std::initializer_list<PyGetSetDef> THPStream_properties = {
    {"device", (getter)THPStream_get_device, nullptr, nullptr, nullptr},
    {nullptr}};

static const std::initializer_list<PyMethodDef> THPStream_methods = {
    {"query", THPStream_query, METH_NOARGS, nullptr},
    {"synchronize", THPStream_synchronize, METH_NOARGS, nullptr},
    {"wait_event", THPStream_wait_event, METH_O, nullptr},
    {"wait_stream", THPStream_wait_stream, METH_O, nullptr},
    {"record_event",
     castPyCFunctionWithKeywords(THPStream_record_event),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"__eq__", (PyCFunction)THPStream_eq, METH_O, nullptr},
    {"__enter__", THPStream_enter, METH_NOARGS, nullptr},
    {"__exit__", THPStream_exit, METH_VARARGS, nullptr},
    {nullptr}};

static PyTypeObject THPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.Stream", /* tp_name */
    sizeof(THPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPStream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPStream_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    (hashfunc)THPStream_hash, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    THPStream_getattro, /* tp_getattro */
    THPStream_setattro, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    THPStream_richcompare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyMethodDef*>(std::data(THPStream_methods)), /* tp_methods */
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyMemberDef*>(std::data(THPStream_members)), /* tp_members */
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyGetSetDef*>(std::data(THPStream_properties)), /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPStream_pynew, /* tp_new */
};

void THPStream_init(PyObject* module) {
  THPStreamClass = &THPStreamType;
  Py_SET_TYPE(&THPStreamType, &PyType_Type);
  if (PyType_Ready(&THPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPStreamType);
  if (PyModule_AddObject(module, "Stream", (PyObject*)&THPStreamType) < 0) {
    throw python_error();
  }
}
