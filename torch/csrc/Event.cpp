#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/core/Event.h>
#include <c10/core/Stream.h>

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <structmember.h>
#include <string>

PyTypeObject* THPEventClass = nullptr;

static PyObject* THPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  static torch::PythonArgParser parser({
      "Event(Device device=None, *, bool enable_timing=False, bool blocking=False, bool interprocess=False)",
  });

  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  auto device = r.deviceOptional(0);

  if (!device.has_value()) {
    device = at::Device(at::getAccelerator(false).value_or(at::kCPU));
  }
  enable_timing = r.toBoolWithDefault(1, false);
  blocking = r.toBoolWithDefault(2, false);
  interprocess = r.toBoolWithDefault(3, false);

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    TORCH_CHECK(ptr, "Failed to allocate memory for Event");
  }

  THPEvent* self = (THPEvent*)ptr.get();

  // TODO: blocking and interprocess are not supported yet. To support them, the
  // flag system of c10::Event needs to be refactored. C10::Event should also
  // provide a generic constructor to support blocking and interprocess events.
  (void)blocking;
  (void)interprocess;

  new (&self->event) c10::Event(
      device->type(),
      // See note [Flags defining the behavior of events]
      // BACKEND_DEFAULT is a enable-timing flag, and
      // PYTORCH_DEFAULT is a disable-timing flag.
      (enable_timing ? c10::EventFlag::BACKEND_DEFAULT
                     : c10::EventFlag::PYTORCH_DEFAULT));

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEvent_new(c10::DeviceType device_type, c10::EventFlag flag) {
  auto type = (PyTypeObject*)&THPEventType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  TORCH_CHECK(self, "Failed to allocate memory for Event");
  auto self_ = reinterpret_cast<THPEvent*>(self.get());
  new (&self_->event) c10::Event(device_type, flag);
  return self.release();
}

static void THPEvent_dealloc(THPEvent* self) {
  {
    pybind11::gil_scoped_release no_gil{};
    self->event.~Event();
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THPEvent_get_device(THPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->event.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_record(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto self = (THPEvent*)_self;
  PyObject* _stream = Py_None;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* accepted_args[] = {"stream", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|O",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(accepted_args),
          &_stream)) {
    TORCH_WARN("Parsing THPEvent_record arg fails");
    return nullptr;
  }
  if (_stream != Py_None) {
    auto stream = (THPStream*)_stream;
    self->event.record(c10::Stream::unpack3(
        stream->stream_id,
        static_cast<c10::DeviceIndex>(stream->device_index),
        static_cast<c10::DeviceType>(stream->device_type)));
  } else {
    c10::impl::VirtualGuardImpl impl{
        static_cast<c10::DeviceType>(self->event.device_type())};
    self->event.record(impl.getStream(impl.getDevice()));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_from_ipc_handle(
    PyObject* _type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "torch.Event ipc is not supported yet, please open an issue if you need this!");
  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THPEvent* self = (THPEvent*)ptr.get();

  // TODO: for constructing event from ipc handle, the c10::Event needs to have
  // more general constructor to achieve that.
  new (&self->event) c10::Event(device.type(), c10::EventFlag::PYTORCH_DEFAULT);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_ipc_handle(
    PyObject* _self [[maybe_unused]],
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "torch.Event ipc is not supported yet, please open an issue if you need this!");
  constexpr const char* handle = "0";
  return PyBytes_FromStringAndSize(
      handle, std::char_traits<char>::length(handle));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_wait(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS {
    auto self = (THPEvent*)_self;
    PyObject* _stream = Py_None;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    constexpr const char* accepted_args[] = {"stream", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|O",
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<char**>(accepted_args),
            &_stream)) {
      TORCH_WARN("Parsing THPEvent_wait arg fails");
      return nullptr;
    }
    if (_stream != Py_None) {
      auto stream = (THPStream*)_stream;
      self->event.block(c10::Stream::unpack3(
          stream->stream_id,
          static_cast<c10::DeviceIndex>(stream->device_index),
          static_cast<c10::DeviceType>(stream->device_type)));
    } else {
      c10::impl::VirtualGuardImpl impl{
          static_cast<c10::DeviceType>(self->event.device_type())};
      self->event.block(impl.getStream(impl.getDevice()));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPEvent*)_self;
  return PyBool_FromLong(self->event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THPEvent*)_self;
  auto other = (THPEvent*)_other;
  return PyFloat_FromDouble(self->event.elapsedTime(other->event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil{};
    auto self = (THPEvent*)_self;
    self->event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_evend_id(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPEvent*)_self;
  return PyLong_FromVoidPtr(self->event.eventId());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEvent_repr(THPEvent* self) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(
      "torch.Event device_type=" +
      c10::DeviceTypeName(
          static_cast<c10::DeviceType>(self->event.device_type()), true) +
      ", device_index=" + std::to_string(self->event.device_index()) +
      ", event_flag=" +
      std::to_string(static_cast<int64_t>(self->event.flag())) + ", event_id=" +
      std::to_string(reinterpret_cast<int64_t>(self->event.eventId())));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static struct PyGetSetDef THPEvent_properties[] = {
    {"device", (getter)THPEvent_get_device, nullptr, nullptr, nullptr},
    {"event_id", (getter)THPEvent_evend_id, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static PyMethodDef THPEvent_methods[] = {
    {"from_ipc_handle",
     castPyCFunctionWithKeywords(THPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"record",
     castPyCFunctionWithKeywords(THPEvent_record),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"wait",
     castPyCFunctionWithKeywords(THPEvent_wait),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"query", THPEvent_query, METH_NOARGS, nullptr},
    {"elapsed_time", THPEvent_elapsed_time, METH_O, nullptr},
    {"synchronize", THPEvent_synchronize, METH_NOARGS, nullptr},
    {"ipc_handle", THPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.Event", /* tp_name */
    sizeof(THPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPEvent_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPEvent_repr, /* tp_repr */
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
    THPEvent_methods, /* tp_methods */
    nullptr, /* tp_members */
    THPEvent_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPEvent_pynew, /* tp_new */
};

void THPEvent_init(PyObject* module) {
  THPEventClass = &THPEventType;
  if (PyType_Ready(&THPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPEventType);
  if (PyModule_AddObject(module, "Event", (PyObject*)&THPEventType) < 0) {
    throw python_error();
  }
}
