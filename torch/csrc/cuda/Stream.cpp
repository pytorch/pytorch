#include "Stream.h"

#include "THP.h"
#include "Module.h"

#include "THC/THCStream.h"
#include "ATen/cuda/CUDAStream.h"

#include <structmember.h>
#include <cuda_runtime_api.h>

PyObject *THCPStreamClass = nullptr;

static PyObject * THCPStream_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS

  int current_device;
  THCudaCheck(cudaGetDevice(&current_device));

  int priority = 0;
  unsigned long long cdata = 0;

  static char *kwlist[] = {"priority", "_cdata", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iK", kwlist, &priority, &cdata)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THCStream* stream;
  if (cdata) {
    stream = (THCStream*) cdata;
  } else {
    const bool isHighPriority = priority < 0 ? true : false;
    stream = at::cuda::detail::CUDAStream_getStreamFromPool(isHighPriority);
  }

  THCPStream* self = (THCPStream *)ptr.get();
  self->cdata = stream;
  self->device = stream ? THCStream_device(stream) : current_device;
  self->cuda_stream = stream ? THCStream_stream(stream) : nullptr;
  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPStream_dealloc(THCPStream* self)
{
  THCStream_free(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static struct PyMemberDef THCPStream_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THCPStream, cdata), READONLY, nullptr},
  {(char*)"device", T_INT, offsetof(THCPStream, device), READONLY, nullptr},
  {(char*)"cuda_stream", T_ULONGLONG, offsetof(THCPStream, cuda_stream), READONLY, nullptr},
  {nullptr}
};

static PyMethodDef THCPStream_methods[] = {
  {nullptr}
};

PyTypeObject THCPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._CudaStreamBase",             /* tp_name */
  sizeof(THCPStream),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THCPStream_dealloc,        /* tp_dealloc */
  0,                                     /* tp_print */
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
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THCPStream_pynew,                      /* tp_new */
};


bool THCPStream_init(PyObject *module)
{
  THCPStreamClass = (PyObject*)&THCPStreamType;
  if (PyType_Ready(&THCPStreamType) < 0)
    return false;
  Py_INCREF(&THCPStreamType);
  PyModule_AddObject(module, "_CudaStreamBase", (PyObject *)&THCPStreamType);
  return true;
}
