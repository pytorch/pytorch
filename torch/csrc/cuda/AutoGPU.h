#ifndef THCP_AUTOGPU_INC
#define THCP_AUTOGPU_INC

#include <Python.h>
#include "THP.h"
#include "torch/csrc/utils/auto_gpu.h"

#ifdef WITH_CUDA
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
