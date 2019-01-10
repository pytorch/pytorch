#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.hpp"
#else

typedef struct THTensor
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

} THTensor;

#endif
