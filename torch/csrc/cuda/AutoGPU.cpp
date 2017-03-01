#include "AutoGPU.h"

#include "THCP.h"
#include <THC/THC.h>

static int getObjDevice(PyObject *obj) {
  PyObject *obj_type = (PyObject*)Py_TYPE(obj);
  if (obj_type == THCPDoubleTensorClass) {
    return THCudaDoubleTensor_getDevice(LIBRARY_STATE ((THCPDoubleTensor*)obj)->cdata);
  } else if (obj_type == THCPFloatTensorClass) {
    return THCudaTensor_getDevice(LIBRARY_STATE ((THCPFloatTensor*)obj)->cdata);
  } else if (obj_type == THCPHalfTensorClass) {
    return THCudaHalfTensor_getDevice(LIBRARY_STATE ((THCPHalfTensor*)obj)->cdata);
  } else if (obj_type == THCPLongTensorClass) {
    return THCudaLongTensor_getDevice(LIBRARY_STATE ((THCPLongTensor*)obj)->cdata);
  } else if (obj_type == THCPIntTensorClass) {
    return THCudaIntTensor_getDevice(LIBRARY_STATE ((THCPIntTensor*)obj)->cdata);
  } else if (obj_type == THCPShortTensorClass) {
    return THCudaShortTensor_getDevice(LIBRARY_STATE ((THCPShortTensor*)obj)->cdata);
  } else if (obj_type == THCPCharTensorClass) {
    return THCudaCharTensor_getDevice(LIBRARY_STATE ((THCPCharTensor*)obj)->cdata);
  } else if (obj_type == THCPByteTensorClass) {
    return THCudaByteTensor_getDevice(LIBRARY_STATE ((THCPByteTensor*)obj)->cdata);
  }
  return -1;
}

static int getObjDevice(PyObject *args, PyObject *self) {
  if (self) {
    int device = getObjDevice(self);
    if (device != -1) {
      return device;
    }
  }
  if (args) {
    for (int i = 0; i < PyTuple_Size(args); i++) {
      int device = getObjDevice(PyTuple_GET_ITEM(args, i));
      if (device != -1) {
        return device;
      }
    }
  }
  return -1;
}

THCPAutoGPU::THCPAutoGPU(int device_id) : AutoGPU(device_id) {}

THCPAutoGPU::THCPAutoGPU(PyObject *args, PyObject *self)
  : AutoGPU(getObjDevice(args, self)) {
}

void THCPAutoGPU::setObjDevice(PyObject *obj) {
  setDevice(getObjDevice(obj));
}
