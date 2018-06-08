#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <atomic>
#include "THCStream.h"

struct THCStream
{
    cudaStream_t stream;
    int device;
    std::atomic<int> refcount;
};
