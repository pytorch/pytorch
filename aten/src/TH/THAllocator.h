#ifndef TH_ALLOCATOR_INC
#define TH_ALLOCATOR_INC

#include "THGeneral.h"

#define TH_ALLOCATOR_MAPPED_SHARED 1
#define TH_ALLOCATOR_MAPPED_SHAREDMEM 2
#define TH_ALLOCATOR_MAPPED_EXCLUSIVE 4
#define TH_ALLOCATOR_MAPPED_NOCREATE 8
#define TH_ALLOCATOR_MAPPED_KEEPFD 16
#define TH_ALLOCATOR_MAPPED_FROMFD 32
#define TH_ALLOCATOR_MAPPED_UNLINK 64

#ifdef __cplusplus
#include <ATen/Allocator.h>
using THAllocator = at::Allocator;
#else
// struct at_THAllocator doesn't and will never exist, but we cannot name
// the actual struct because it's a namespaced C++ thing
typedef struct at_THAllocator THAllocator;
#endif

/* default malloc/free allocator. malloc and realloc raise an error (using
 * THError) on allocation failure.
 */
TH_API THAllocator* getTHDefaultAllocator();

/* file map allocator
 */
typedef struct THMapAllocatorContext_  THMapAllocatorContext;
TH_API THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int flags);
TH_API THMapAllocatorContext *THMapAllocatorContext_newWithFd(const char *filename,
    int fd, int flags);
TH_API char * THMapAllocatorContext_filename(THMapAllocatorContext *ctx);
TH_API int THMapAllocatorContext_fd(THMapAllocatorContext *ctx);
TH_API ptrdiff_t THMapAllocatorContext_size(THMapAllocatorContext *ctx);
TH_API void THMapAllocatorContext_free(THMapAllocatorContext *ctx);
TH_API void THRefcountedMapAllocator_incref(THMapAllocatorContext *ctx, void *data);
TH_API int THRefcountedMapAllocator_decref(THMapAllocatorContext *ctx, void *data);

TH_API THAllocator* getTHMapAllocator();
TH_API THAllocator* getTHRefcountedMapAllocator();

#endif
