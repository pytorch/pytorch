#ifndef THCP_CUDA_MODULE_INC
#define THCP_CUDA_MODULE_INC

extern PyObject *THCPDoubleStorageClass;
extern PyObject *THCPFloatStorageClass;
extern PyObject *THCPLongStorageClass;
extern PyObject *THCPIntStorageClass;
extern PyObject *THCPHalfStorageClass;
extern PyObject *THCPShortStorageClass;
extern PyObject *THCPCharStorageClass;
extern PyObject *THCPByteStorageClass;

extern PyObject *THCPDoubleTensorClass;
extern PyObject *THCPFloatTensorClass;
extern PyObject *THCPLongTensorClass;
extern PyObject *THCPIntTensorClass;
extern PyObject *THCPHalfTensorClass;
extern PyObject *THCPShortTensorClass;
extern PyObject *THCPCharTensorClass;
extern PyObject *THCPByteTensorClass;

extern PyObject * THCPModule_getDevice_wrap(PyObject *self);
extern PyObject * THCPModule_setDevice_wrap(PyObject *self, PyObject *arg);
void THCPModule_setDevice(int idx);

#endif
