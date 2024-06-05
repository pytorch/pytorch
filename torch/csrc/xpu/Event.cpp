#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/xpu/Event.h>
#include <torch/csrc/xpu/Module.h>
#include <torch/csrc/xpu/Stream.h>

#include <structmember.h>

PyObject* THXPEventClass = nullptr;

static PyObject* THXPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"enable_timing", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|b",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &enable_timing)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THXPEvent* self = (THXPEvent*)ptr.get();

  new (&self->xpu_event) at::xpu::XPUEvent(enable_timing);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THXPEvent_dealloc(THXPEvent* self) {
  {
    pybind11::gil_scoped_release no_gil{};
    self->xpu_event.~XPUEvent();
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THXPEvent_get_sycl_event(THXPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(&self->xpu_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPEvent_get_device(THXPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->xpu_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  auto* self = (THXPEvent*)_self;
  auto* stream = (THXPStream*)_stream;
  self->xpu_event.record(stream->xpu_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  auto* self = (THXPEvent*)_self;
  auto* stream = (THXPStream*)_stream;
  self->xpu_event.block(stream->xpu_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THXPEvent*)_self;
  return PyBool_FromLong(self->xpu_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto* self = (THXPEvent*)_self;
  auto* other = (THXPEvent*)_other;
  return PyFloat_FromDouble(self->xpu_event.elapsed_time(other->xpu_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto* self = (THXPEvent*)_self;
    self->xpu_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static struct PyGetSetDef THXPEvent_properties[] = {
    {"device", (getter)THXPEvent_get_device, nullptr, nullptr, nullptr},
    {"sycl_event", (getter)THXPEvent_get_sycl_event, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static PyMethodDef THXPEvent_methods[] = {
    {(char*)"record", THXPEvent_record, METH_O, nullptr},
    {(char*)"wait", THXPEvent_wait, METH_O, nullptr},
    {(char*)"query", THXPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THXPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", THXPEvent_synchronize, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THXPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._XpuEventBase", /* tp_name */
    sizeof(THXPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THXPEvent_dealloc, /* tp_dealloc */
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
    THXPEvent_methods, /* tp_methods */
    nullptr, /* tp_members */
    THXPEvent_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THXPEvent_pynew, /* tp_new */
};

void THXPEvent_init(PyObject* module) {
  THXPEventClass = (PyObject*)&THXPEventType;
  if (PyType_Ready(&THXPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THXPEventType);
  if (PyModule_AddObject(module, "_XpuEventBase", (PyObject*)&THXPEventType) <
      0) {
    throw python_error();
  }
}
