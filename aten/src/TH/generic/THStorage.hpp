#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.hpp"
#else

typedef struct THStorage
{
    real *data;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;
} THStorage;

#endif
