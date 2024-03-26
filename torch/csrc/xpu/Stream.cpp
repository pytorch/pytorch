#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/xpu/Module.h>
#include <torch/csrc/xpu/Stream.h>

#include <structmember.h>

PyObject* THXPStreamClass = nullptr;

static PyObject* THXPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  const auto current_device = c10::xpu::current_device();

  int32_t priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority", "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  at::xpu::XPUStream stream = (stream_id || device_index || device_type)
      ? at::xpu::XPUStream::unpack3(
            stream_id,
            static_cast<c10::DeviceIndex>(device_index),
            static_cast<c10::DeviceType>(device_type))
      : at::xpu::getStreamFromPool(priority, current_device);

  THXPStream* self = (THXPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->xpu_stream) at::xpu::XPUStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THXPStream_dealloc(THXPStream* self) {
  self->xpu_stream.~XPUStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THXPStream_get_device(THXPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->xpu_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPStream_get_sycl_queue(THXPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(&self->xpu_stream.queue());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPStream_get_priority(THXPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->xpu_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPStream_priority_range(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto [least_priority, greatest_priority] =
      at::xpu::XPUStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THXPStream*)_self;
  return PyBool_FromLong(self->xpu_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto* self = (THXPStream*)_self;
    self->xpu_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPStream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto* self = (THXPStream*)_self;
  auto* other = (THXPStream*)_other;
  return PyBool_FromLong(self->xpu_stream == other->xpu_stream);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMemberDef THXPStream_members[] = {{nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyGetSetDef THXPStream_properties[] = {
    {"sycl_queue",
     (getter)THXPStream_get_sycl_queue,
     nullptr,
     nullptr,
     nullptr},
    {"priority", (getter)THXPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static PyMethodDef THXPStream_methods[] = {
    {"query", THXPStream_query, METH_NOARGS, nullptr},
    {"synchronize", THXPStream_synchronize, METH_NOARGS, nullptr},
    {"priority_range",
     THXPStream_priority_range,
     METH_STATIC | METH_NOARGS,
     nullptr},
    {"__eq__", THXPStream_eq, METH_O, nullptr},
    {nullptr}};

PyTypeObject THXPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._XpuStreamBase", /* tp_name */
    sizeof(THXPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THXPStream_dealloc, /* tp_dealloc */
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
    THXPStream_methods, /* tp_methods */
    THXPStream_members, /* tp_members */
    THXPStream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THXPStream_pynew, /* tp_new */
};

void THXPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THXPStreamType.tp_base = THPStreamClass;
  THXPStreamClass = (PyObject*)&THXPStreamType;
  if (PyType_Ready(&THXPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THXPStreamType);
  if (PyModule_AddObject(module, "_XpuStreamBase", (PyObject*)&THXPStreamType) <
      0) {
    throw python_error();
  }
}
