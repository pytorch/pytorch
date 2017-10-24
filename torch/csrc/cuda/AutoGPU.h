#ifndef THCP_AUTOGPU_INC
#define THCP_AUTOGPU_INC

#include <Python.h>
#include "THP.h"
#include "torch/csrc/utils/auto_gpu.h"

class THP_CLASS THCPAutoGPU : public AutoGPU {
public:
  explicit THCPAutoGPU(int device_id=-1);
#ifdef WITH_CUDA
  THCPAutoGPU(PyObject *args, PyObject *self=NULL);
  void setObjDevice(PyObject *obj);
#endif
};

#endif
