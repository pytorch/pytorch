#ifndef THCP_GDSFILE_INC
#define THCP_GDSFILE_INC

#include <ATen/cuda/CUDAGdsFile.h>
#include <torch/csrc/python_headers.h>

struct THCPGdsFile {
  PyObject_HEAD at::cuda::GDSFile gds_file;
};
extern PyObject* THCPGdsFileClass;

void THCPGdsFile_init(PyObject* module);

inline bool THCPGdsFile_Check(PyObject* obj) {
  return THCPGdsFileClass && PyObject_IsInstance(obj, THCPGdsFileClass);
}

PyObject* THCPModule_gds_register_buffer(PyObject* self, PyObject* args);
PyObject* THCPModule_gds_deregister_buffer(PyObject* self, PyObject* args);

#endif // THCP_GDSFILE_INC
