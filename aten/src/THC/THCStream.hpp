#pragma once
#include <atomic>
#include "THCStream.h"

struct THCStream
{
    cudaStream_t stream;
    int device;
    std::atomic<int> refcount;
};
