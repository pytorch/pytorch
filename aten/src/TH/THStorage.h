#ifndef TH_STORAGE_INC
#define TH_STORAGE_INC

#include "THGeneral.h"
#include "THAllocator.h"

#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)

#include "generic/THStorage.h"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.h"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.h"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.h"
#include "THGenerateHalfType.h"

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
TH_API void THStorage_free(THStorage *storage);

TH_API THDescBuff THLongStorage_sizeDesc(const THLongStorage *size);
TH_API THLongStorage *THLongStorage_newInferSize(THLongStorage *size, ptrdiff_t nElement);

// Given the sizes of {2,N} tensors, write out the size when the tensors are expanded together.
TH_API int THLongStorage_inferSize2(THLongStorage *output, int64_t *sizesA, int64_t dimsA,
                                    int64_t *sizesB, int64_t dimsB, char *error_buffer, int buffer_len);
TH_API int THLongStorage_inferSizeN(THLongStorage *output, int n, int64_t **sizes, int64_t *dims,
                                    char *error_buffer, int buffer_len);

TH_API int THLongStorage_inferExpandGeometry(int64_t *tensorSizes, int64_t *tensorStrides, int64_t tensorDim,
                                             THLongStorage *sizes, int64_t **expandedSizes, int64_t **expandedStrides,
                                             char *error_buffer, int buffer_len);

#endif
