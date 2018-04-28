#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.hpp"
#else

typedef struct THCTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    THCStorage *storage;
    ptrdiff_t storageOffset;
    std::atomic<int> refcount;

    char flag;

} THCTensor;

#endif
