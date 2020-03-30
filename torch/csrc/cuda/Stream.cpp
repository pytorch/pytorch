#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>

#include <c10/cuda/CUDAGuard.h>

#include <structmember.h>
#include <cuda_runtime_api.h>

PyObject *THCPStreamClass = nullptr;

static PyObject * THCPStream_pynew(
  PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS

  int current_device;
  THCudaCheck(cudaGetDevice(&current_device));

  int priority = 0;
  uint64_t cdata = 0;

  static char *kwlist[] = {"priority", "_cdata", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "|iK", kwlist, &priority, &cdata)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  at::cuda::CUDAStream stream =
    cdata ?
    at::cuda::CUDAStream::unpack(cdata) :
    at::cuda::getStreamFromPool(
      /* isHighPriority */ priority < 0 ? true : false);

  THCPStream* self = (THCPStream *)ptr.get();
  self->cdata = stream.pack();
  new (&self->cuda_stream) at::cuda::CUDAStream(stream);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPStream_dealloc(THCPStream *self) {
  self->cuda_stream.~CUDAStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THCPStream_get_device(THCPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cuda_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_get_cuda_stream(THCPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->cuda_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_get_priority(THCPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(self->cuda_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_priority_range() {
  HANDLE_TH_ERRORS
  int least_priority, greatest_priority;
  std::tie(least_priority, greatest_priority) =
    at::cuda::CUDAStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_query(THCPStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->cuda_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_synchronize(THCPStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    self->cuda_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPStream_eq(THCPStream *self, THCPStream *other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->cuda_stream == other->cuda_stream);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THCPStream_members[] = {
  {(char*)"_cdata",
    T_ULONGLONG, offsetof(THCPStream, cdata), READONLY, nullptr},
  {nullptr}
};

static struct PyGetSetDef THCPStream_properties[] = {
  {"device", (getter)THCPStream_get_device, nullptr, nullptr, nullptr},
  {"cuda_stream",
    (getter)THCPStream_get_cuda_stream, nullptr, nullptr, nullptr},
  {"priority", (getter)THCPStream_get_priority, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THCPStream_methods[] = {
  {(char*)"query", (PyCFunction)THCPStream_query, METH_NOARGS, nullptr},
  {(char*)"synchronize",
    (PyCFunction)THCPStream_synchronize, METH_NOARGS, nullptr},
  {(char*)"priority_range",
    (PyCFunction)(void(*)(void))THCPStream_priority_range, METH_STATIC | METH_NOARGS, nullptr},
  {(char*)"__eq__", (PyCFunction)THCPStream_eq, METH_O, nullptr},
  {nullptr}
};

PyTypeObject THCPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._CudaStreamBase",            /* tp_name */
  sizeof(THCPStream),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THCPStream_dealloc,        /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THCPStream_methods,                    /* tp_methods */
  THCPStream_members,                    /* tp_members */
  THCPStream_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THCPStream_pynew,                      /* tp_new */
};


void THCPStream_init(PyObject *module)
{
  THCPStreamClass = (PyObject*)&THCPStreamType;
  if (PyType_Ready(&THCPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPStreamType);
  if (PyModule_AddObject(
      module, "_CudaStreamBase", (PyObject *)&THCPStreamType) < 0) {
    throw python_error();
  }
}
