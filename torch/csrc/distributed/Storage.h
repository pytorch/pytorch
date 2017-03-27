#ifndef THDP_STORAGE_INC
#define THDP_STORAGE_INC

#define THDPStorage TH_CONCAT_3(THDP,Real,Storage)
#define THDPStorageStr TH_CONCAT_STRING_3(torch.cuda.,Real,Storage)
#define THDPStorageClass TH_CONCAT_3(THDP,Real,StorageClass)
#define THDPStorage_(NAME) TH_CONCAT_4(THDP,Real,Storage_,NAME)

#define THDPDoubleStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPDoubleStorageClass)
#define THDPFloatStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPFloatStorageClass)
#define THDPHalfStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPHalfStorageClass)
#define THDPLongStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPLongStorageClass)
#define THDPIntStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPIntStorageClass)
#define THDPShortStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPShortStorageClass)
#define THDPCharStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPCharStorageClass)
#define THDPByteStorage_Check(obj) \
    PyObject_IsInstance(obj, THDPByteStorageClass)

#define THDPDoubleStorage_CData(obj)  (obj)->cdata
#define THDPFloatStorage_CData(obj)   (obj)->cdata
#define THDPLongStorage_CData(obj)    (obj)->cdata
#define THDPIntStorage_CData(obj)     (obj)->cdata
#define THDPShortStorage_CData(obj)   (obj)->cdata
#define THDPCharStorage_CData(obj)    (obj)->cdata
#define THDPByteStorage_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THDPStorageType TH_CONCAT_3(THDP,Real,StorageType)
#define THDPStorageBaseStr TH_CONCAT_STRING_3(Distributed,Real,StorageBase)
#endif

#include "override_macros.h"

#define THD_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <THD/base/THDGenerateAllTypes.h>

#endif

