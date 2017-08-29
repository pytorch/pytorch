#ifndef THCP_AUTOGPU_INC
#define THCP_AUTOGPU_INC

#include <Python.h>
#include "THP.h"
#include "torch/csrc/utils/auto_gpu.h"

#if defined(WITH_CUDA) && defined(_MSC_VER)
class THP_CLASS THCPAutoGPU : public AutoGPU {
#else
class THCPAutoGPU : public AutoGPU {
#endif
public:
  explicit THCPAutoGPU(int device_id=-1);
  THCPAutoGPU(PyObject *args, PyObject *self=NULL);
  void setObjDevice(PyObject *obj);
};

#endif
