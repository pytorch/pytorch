#include <torch/csrc/THP.h>
#include <torch/csrc/mps/Stream.h>

#ifdef USE_MPS
// #include <torch/csrc/Device.h>
#include <torch/csrc/utils/pybind.h>

#include <structmember.h>

PyObject* THMPStreamClass = nullptr;

static PyObject* THMPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist))) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  at::mps::MPSStream* stream = at::mps::getStreamFromPool();
  c10::Stream unwrapped = stream->unwrap();

  THMPStream* self = (THMPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(unwrapped.id());
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(unwrapped.device_index());
  self->device_type = static_cast<int64_t>(unwrapped.device_type());
  self->mps_stream = stream;

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THMPStream_dealloc(THMPStream* self) {
  // Only the Python object needs to be deleted, not `self->mps_stream`, the
  // underlying MPSStream, since that is kept alive in the pool.
  THPStream_dealloc_common(reinterpret_cast<THPStream*>(self));
}

static PyObject* THMPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto self = (THMPStream*)_self;
    self->mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMemberDef THMPStream_members[] = {{nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyGetSetDef THMPStream_properties[] = {{nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static PyMethodDef THMPStream_methods[] = {
    {"synchronize", THMPStream_synchronize, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THMPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._MPSStreamBase", /* tp_name */
    sizeof(THMPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THMPStream_dealloc, /* tp_dealloc */
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
    0, /* tp_weaklistoffset (inherited from THPStreamType via tp_base) */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THMPStream_methods, /* tp_methods */
    THMPStream_members, /* tp_members */
    THMPStream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THMPStream_pynew, /* tp_new */
};

void THMPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THMPStreamType.tp_base = THPStreamClass;
  THMPStreamClass = (PyObject*)&THMPStreamType;
  if (PyType_Ready(&THMPStreamType) < 0) {
    throw python_error(); // @allow-raw-throw
  }
  Py_INCREF(&THMPStreamType);
  if (PyModule_AddObject(module, "_MPSStreamBase", (PyObject*)&THMPStreamType) <
      0) {
    throw python_error(); // @allow-raw-throw
  }
}

#endif // USE_MPS
