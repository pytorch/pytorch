#include "Stream.h"

#include "THP.h"
#include "Module.h"

#include <structmember.h>
#include <cuda_runtime_api.h>

PyObject *THCPStreamClass = NULL;

static PyObject * THCPStream_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS

  int current_device;
  THCudaCheck(cudaGetDevice(&current_device));

  THPObjectPtr ptr = (PyObject *)type->tp_alloc(type, 0);
  THCPStream* self = (THCPStream *)ptr.get();
  THCStream* stream = NULL;
  if (kwargs && PyDict_Size(kwargs) > 0) {
    PyObject *cdata_ptr = PyDict_GetItemString(kwargs, "_cdata");
    if (cdata_ptr && PyDict_Size(kwargs) == 1 && THPUtils_checkLong(cdata_ptr)) {
      stream = (THCStream*) PyLong_AsVoidPtr(cdata_ptr);
      if (stream) {
        THCStream_retain(stream);
      }
    } else {
      THPUtils_setError("torch.cuda.Stream(): invalid keyword arguments");
      return NULL;
    }
  } else {
    stream = THCStream_new(cudaStreamNonBlocking);
  }

  self->cdata = stream;
  self->device = stream ? stream->device : current_device;
  self->cuda_stream = stream ? stream->stream : NULL;
  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPStream_dealloc(THCPStream* self)
{
  THCStream_free(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static struct PyMemberDef THCPStream_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THCPStream, cdata), READONLY, NULL},
  {(char*)"device", T_INT, offsetof(THCPStream, device), READONLY, NULL},
  {(char*)"cuda_stream", T_ULONGLONG, offsetof(THCPStream, cuda_stream), READONLY, NULL},
  {NULL}
};

static PyMethodDef THCPStream_methods[] = {
  {NULL}
};

PyTypeObject THCPStreamType = {
  PyVarObject_HEAD_INIT(NULL, 0)
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
  NULL,                                  /* tp_doc */
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
