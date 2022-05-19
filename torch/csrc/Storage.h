#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC
#include <torch/csrc/THConcat.h>

#define THPStorageStr "torch._UntypedStorage"
#define THPStorage_(NAME) TH_CONCAT_2(THPStorage_,NAME)

#define THPStorage_Check(obj) \
    PyObject_IsInstance(obj, THPStorageClass)

#define THPStorage_CData(obj)           (obj)->cdata

#define THPStorageBaseStr "StorageBase"

#include <torch/csrc/generic/Storage.h>
#include <torch/csrc/THGenerateByteType.h>

#endif
