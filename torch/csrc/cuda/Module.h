#ifndef THCP_CUDA_MODULE_INC
#define THCP_CUDA_MODULE_INC

void THCPModule_setDevice(int idx);
PyObject* THCPModule_getDevice_wrap(PyObject* self);
PyObject* THCPModule_setDevice_wrap(PyObject* self, PyObject* arg);
PyObject* THCPModule_getDeviceName_wrap(PyObject* self, PyObject* arg);
PyObject* THCPModule_getDriverVersion(PyObject* self);
PyObject* THCPModule_isDriverSufficient(PyObject* self);
PyObject* THCPModule_getCurrentBlasHandle_wrap(PyObject* self);

#endif
