#include "AutoGPU.h"

#include "THCP.h"
#include <THC/THC.h>

THCPAutoGPU::THCPAutoGPU(int device_id) {
  setDevice(device_id);
}

THCPAutoGPU::THCPAutoGPU(PyObject *args, PyObject *self) {
  if (self && setObjDevice(self))
    return;

  if (!args)
    return;
  for (int i = 0; i < PyTuple_Size(args); i++) {
    PyObject *arg = PyTuple_GET_ITEM(args, i);
    if (setObjDevice(arg)) return;
  }
}

bool THCPAutoGPU::setObjDevice(PyObject *obj) {
  int new_device = -1;
  PyObject *obj_type = (PyObject*)Py_TYPE(obj);
  if (obj_type == THCPDoubleTensorClass) {
    new_device = THCudaDoubleTensor_getDevice(LIBRARY_STATE ((THCPDoubleTensor*)obj)->cdata);
  } else if (obj_type == THCPFloatTensorClass) {
    new_device = THCudaTensor_getDevice(LIBRARY_STATE ((THCPFloatTensor*)obj)->cdata);
  } else if (obj_type == THCPHalfTensorClass) {
    new_device = THCudaHalfTensor_getDevice(LIBRARY_STATE ((THCPHalfTensor*)obj)->cdata);
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
  return setDevice(new_device);
}

bool THCPAutoGPU::setDevice(int new_device) {
  if (new_device == -1)
    return false;

  if (device == -1)
    THCudaCheck(cudaGetDevice(&device));
  if (new_device != device)
    THCPModule_setDevice(new_device);
  return true;
}

// This can throw... But if it does I have no idea how to recover.
THCPAutoGPU::~THCPAutoGPU() {
  if (device != -1)
    THCPModule_setDevice(device);
}

