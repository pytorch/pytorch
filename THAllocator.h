#ifndef TH_ALLOCATOR_INC
#define TH_ALLOCATOR_INC

#include "THGeneral.h"

#define TH_ALLOCATOR_MAPPED_SHARED 1
#define TH_ALLOCATOR_MAPPED_SHAREDMEM 2

/* Custom allocator
 */
typedef struct THAllocator {
  void* (*malloc)(void*, long);
  void* (*realloc)(void*, void*, long);
  void (*free)(void*, void*);
} THAllocator;

/* default malloc/free allocator. malloc and realloc raise an error (using
 * THError) on allocation failure.
 */
extern THAllocator THDefaultAllocator;

/* file map allocator
 */
typedef struct THMapAllocatorContext_  THMapAllocatorContext;
THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int shared);
long THMapAllocatorContext_size(THMapAllocatorContext *ctx);
void THMapAllocatorContext_free(THMapAllocatorContext *ctx);

extern THAllocator THMapAllocator;

#endif
