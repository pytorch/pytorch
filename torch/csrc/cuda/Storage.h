#ifndef THCP_STORAGE_INC
#define THCP_STORAGE_INC

#define THCPStorageStr TH_CONCAT_STRING_3(torch.cuda.,Real,Storage)
#define THCPStorageClass TH_CONCAT_3(THCP,Real,StorageClass)
#define THCPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)

#define THCPDoubleStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPDoubleStorageClass)
#define THCPFloatStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPFloatStorageClass)
#define THCPHalfStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPHalfStorageClass)
#define THCPLongStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPLongStorageClass)
#define THCPIntStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPIntStorageClass)
#define THCPShortStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPShortStorageClass)
#define THCPCharStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPCharStorageClass)
#define THCPByteStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPByteStorageClass)
#define THCPBoolStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPBoolStorageClass)
#define THCPBFloat16Storage_Check(obj) \
    PyObject_IsInstance(obj, THCPBFloat16StorageClass)
#define THCPComplexDoubleStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPDoubleStorageClass)
#define THCPComplexFloatStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPFloatStorageClass)

#define THCPDoubleStorage_CData(obj)             (obj)->cdata
#define THCPFloatStorage_CData(obj)              (obj)->cdata
#define THCPLongStorage_CData(obj)               (obj)->cdata
#define THCPIntStorage_CData(obj)                (obj)->cdata
#define THCPShortStorage_CData(obj)              (obj)->cdata
#define THCPCharStorage_CData(obj)               (obj)->cdata
#define THCPByteStorage_CData(obj)               (obj)->cdata
#define THCPBoolStorage_CData(obj)               (obj)->cdata
#define THCPBFloat16Storage_CData(obj)           (obj)->cdata
#define THCPComplexDoubleStorage_CData(obj)      (obj)->cdata
#define THCPComplexFloatStorage_CData(obj)       (obj)->cdata

#define THCPStorageType TH_CONCAT_3(THCP,Real,StorageType)
#define THCPStorageBaseStr TH_CONCAT_STRING_3(Cuda,Real,StorageBase)

#include <torch/csrc/cuda/override_macros.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <THH/THHGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <THH/THHGenerateComplexTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <THH/THHGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <THH/THHGenerateBFloat16Type.h>

#endif
