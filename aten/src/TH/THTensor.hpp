#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THTensor.h"
#include "THStorage.hpp"

#include <atomic>

typedef struct _THTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage;
    ptrdiff_t storageOffset;
    std::atomic<int> refcount;

    char flag;

} _THTensor;

#include "generic/THTensor.hpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.hpp"
#include "THGenerateHalfType.h"

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"
