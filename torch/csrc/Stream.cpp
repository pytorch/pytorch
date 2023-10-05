#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <structmember.h>

PyTypeObject* THPStreamClass = nullptr;

static PyObject* THPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THPStream* self = (THPStream*)ptr.get();
  self->stream_id = stream_id;
  self->device_index = device_index;
  self->device_type = device_type;
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
  return ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPStream_dealloc(THPStream* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THPStream_get_device(THPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(c10::Device(
      static_cast<c10::DeviceType>(self->device_type),
      static_cast<c10::DeviceIndex>(self->device_index)));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStream_eq(THPStream* self, THPStream* other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(
      self->stream_id == other->stream_id &&
      self->device_index == other->device_index &&
      self->device_type == other->device_type);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMemberDef THPStream_members[] = {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPStream_properties[] = {
    {"device", (getter)THPStream_get_device, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPStream_methods[] = {
    {"__eq__", (PyCFunction)THPStream_eq, METH_O, nullptr},
    {nullptr}};

PyTypeObject THPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.Stream", /* tp_name */
    sizeof(THPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THPStream_dealloc, /* tp_dealloc */
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
    THPStream_methods, /* tp_methods */
    THPStream_members, /* tp_members */
    THPStream_properties, /* tp_getset */
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
