#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.hpp"
#else

#include "ATen/CUDAStorageImpl.h"

typedef struct THCStorage
{
    real *data;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct at::CUDAStorageImpl *view;
    int device;
} THCStorage;

#endif
