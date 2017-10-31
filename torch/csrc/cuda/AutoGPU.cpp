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
  } else if (obj_type == THCSPDoubleTensorClass) {
    return THCSDoubleTensor_getDevice(LIBRARY_STATE ((THCSPDoubleTensor*)obj)->cdata);
  } else if (obj_type == THCSPFloatTensorClass) {
    return THCSFloatTensor_getDevice(LIBRARY_STATE ((THCSPFloatTensor*)obj)->cdata);
  } else if (obj_type == THCSPHalfTensorClass) {
    return THCSHalfTensor_getDevice(LIBRARY_STATE ((THCSPHalfTensor*)obj)->cdata);
  } else if (obj_type == THCSPLongTensorClass) {
    return THCSLongTensor_getDevice(LIBRARY_STATE ((THCSPLongTensor*)obj)->cdata);
  } else if (obj_type == THCSPIntTensorClass) {
    return THCSIntTensor_getDevice(LIBRARY_STATE ((THCSPIntTensor*)obj)->cdata);
  } else if (obj_type == THCSPShortTensorClass) {
    return THCSShortTensor_getDevice(LIBRARY_STATE ((THCSPShortTensor*)obj)->cdata);
  } else if (obj_type == THCSPCharTensorClass) {
    return THCSCharTensor_getDevice(LIBRARY_STATE ((THCSPCharTensor*)obj)->cdata);
  } else if (obj_type == THCSPByteTensorClass) {
    return THCSByteTensor_getDevice(LIBRARY_STATE ((THCSPByteTensor*)obj)->cdata);
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

THCPAutoGPU::THCPAutoGPU(PyObject *args, PyObject *self)
  : AutoGPU(getObjDevice(args, self)) {
}

void THCPAutoGPU::setObjDevice(PyObject *obj) {
  setDevice(getObjDevice(obj));
}
