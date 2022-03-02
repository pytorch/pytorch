#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC
#include <torch/csrc/THConcat.h>

#define THPStorageStr TH_CONCAT_STRING_3(torch.,Real,Storage)
#define THPStorageClass TH_CONCAT_3(THP,Real,StorageClass)
#define THPStorage_(NAME) TH_CONCAT_4(THP,Real,Storage_,NAME)

#define THPByteStorage_Check(obj) \
    PyObject_IsInstance(obj, THPByteStorageClass)

#define THPByteStorage_CData(obj)           (obj)->cdata

#define THPStorageType TH_CONCAT_3(THP,Real,StorageType)
#define THPStorageBaseStr TH_CONCAT_STRING_2(Real,StorageBase)

#include <torch/csrc/generic/Storage.h>
#include <torch/csrc/THGenerateByteType.h>

#endif
