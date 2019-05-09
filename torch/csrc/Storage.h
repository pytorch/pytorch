#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC

#define THPStorageStr TH_CONCAT_STRING_3(torch.,Real,Storage)
#define THPStorageClass TH_CONCAT_3(THP,Real,StorageClass)
#define THPStorage_(NAME) TH_CONCAT_4(THP,Real,Storage_,NAME)

#define THPDoubleStorage_Check(obj) \
    PyObject_IsInstance(obj, THPDoubleStorageClass)
#define THPFloatStorage_Check(obj) \
    PyObject_IsInstance(obj, THPFloatStorageClass)
#define THPHalfStorage_Check(obj) \
    PyObject_IsInstance(obj, THPFloatStorageClass)
#define THPLongStorage_Check(obj) \
    PyObject_IsInstance(obj, THPLongStorageClass)
#define THPIntStorage_Check(obj) \
    PyObject_IsInstance(obj, THPIntStorageClass)
#define THPShortStorage_Check(obj) \
    PyObject_IsInstance(obj, THPShortStorageClass)
#define THPCharStorage_Check(obj) \
    PyObject_IsInstance(obj, THPCharStorageClass)
#define THPByteStorage_Check(obj) \
    PyObject_IsInstance(obj, THPByteStorageClass)
#define THPBoolStorage_Check(obj) \
    PyObject_IsInstance(obj, THPBoolStorageClass)

#define THPDoubleStorage_CData(obj)  (obj)->cdata
#define THPFloatStorage_CData(obj)   (obj)->cdata
#define THPHalfStorage_CData(obj)    (obj)->cdata
#define THPLongStorage_CData(obj)    (obj)->cdata
#define THPIntStorage_CData(obj)     (obj)->cdata
#define THPShortStorage_CData(obj)   (obj)->cdata
#define THPCharStorage_CData(obj)    (obj)->cdata
#define THPByteStorage_CData(obj)    (obj)->cdata
#define THPBoolStorage_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THPStorageType TH_CONCAT_3(THP,Real,StorageType)
#define THPStorageBaseStr TH_CONCAT_STRING_2(Real,StorageBase)
#endif

#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateAllTypes.h>

#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateHalfType.h>

#include <torch/csrc/generic/Storage.h>
#include <TH/THGenerateBoolType.h>

#endif
