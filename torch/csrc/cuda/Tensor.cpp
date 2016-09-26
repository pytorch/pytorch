#include <Python.h>
#include <structmember.h>

#include <TH/THMath.h>
#include <stdbool.h>
#include <vector>
#include <stack>
#include <tuple>
#include "THCP.h"

#include "override_macros.h"

class THCPAutoGPU {
public:
  THCPAutoGPU(PyObject *args, PyObject *self=NULL) {
    if (self && setDevice(self))
      return;

    if (!args)
      return;
    for (int i = 0; i < PyTuple_Size(args); i++) {
      PyObject *arg = PyTuple_GET_ITEM(args, i);
      if (setDevice(arg)) return;
    }
  }

  bool setDevice(PyObject *obj) {
    int new_device = -1;
    PyObject *obj_type = (PyObject*)Py_TYPE(obj);
    if (obj_type == THCPDoubleTensorClass) {
      new_device = THCudaDoubleTensor_getDevice(LIBRARY_STATE ((THCPDoubleTensor*)obj)->cdata);
    } else if (obj_type == THCPFloatTensorClass) {
      new_device = THCudaTensor_getDevice(LIBRARY_STATE ((THCPFloatTensor*)obj)->cdata);
    } else if (obj_type == THCPLongTensorClass) {
      new_device = THCudaLongTensor_getDevice(LIBRARY_STATE ((THCPLongTensor*)obj)->cdata);
    } else if (obj_type == THCPIntTensorClass) {
      new_device = THCudaIntTensor_getDevice(LIBRARY_STATE ((THCPIntTensor*)obj)->cdata);
    } else if (obj_type == THCPShortTensorClass) {
      new_device = THCudaShortTensor_getDevice(LIBRARY_STATE ((THCPShortTensor*)obj)->cdata);
    } else if (obj_type == THCPCharTensorClass) {
      new_device = THCudaCharTensor_getDevice(LIBRARY_STATE ((THCPCharTensor*)obj)->cdata);
    } else if (obj_type == THCPByteTensorClass) {
      new_device = THCudaByteTensor_getDevice(LIBRARY_STATE ((THCPByteTensor*)obj)->cdata);
    }
    if (new_device != -1) {
      THCudaCheck(cudaGetDevice(&device));
      THCPModule_setDevice(new_device);
      return true;
    }
    return false;
  }

  // This can throw... But if it does I have no idea how to recover.
  ~THCPAutoGPU() {
    if (device != -1)
      THCPModule_setDevice(device);
  }

  int device = -1;
};

#define THC_GENERIC_FILE "torch/csrc/generic/Tensor.cpp"
#include <THC/THCGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/TensorCopy.cpp"
#include <THC/THCGenerateAllTypes.h>

#include "undef_macros.h"
#include "restore_macros.h"

#include "generic/TensorCopyAsync.cpp"
#include <THC/THCGenerateAllTypes.h>

