#ifndef TH_ALLOCATOR_INC
#define TH_ALLOCATOR_INC

#include "THGeneral.h"

#ifdef __cplusplus
#include <ATen/Allocator.h>
#endif

#define TH_ALLOCATOR_MAPPED_SHARED 1
#define TH_ALLOCATOR_MAPPED_SHAREDMEM 2
#define TH_ALLOCATOR_MAPPED_EXCLUSIVE 4
#define TH_ALLOCATOR_MAPPED_NOCREATE 8
#define TH_ALLOCATOR_MAPPED_KEEPFD 16
#define TH_ALLOCATOR_MAPPED_FROMFD 32
#define TH_ALLOCATOR_MAPPED_UNLINK 64

#ifdef __cplusplus
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
#ifdef __cplusplus
struct TH_API THDefaultDeleter final : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    THFree(ptr);
  }
  static at::BoundDeleter make() {
    return {&singleton_, nullptr};
  }
private:
  static THDefaultDeleter singleton_;
};
#endif

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

// NB: These functions steal the passed context
#ifdef __cplusplus
AT_API std::unique_ptr<void, at::BoundDeleter> THMapAllocatorContext_alloc(THMapAllocatorContext *ctx, ptrdiff_t size);
AT_API std::unique_ptr<void, at::BoundDeleter> THRefcountedMapAllocator_alloc(void *_ctx, ptrdiff_t size);

struct TH_API THMapDeleter final : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override;
  static at::BoundDeleter make(THMapAllocatorContext* ctx) {
    return {&singleton_, ctx};
  }
  static THMapAllocatorContext* getContext(at::BoundDeleter bd) {
    if (bd.deleter_ != &singleton_) return nullptr;
    return static_cast<THMapAllocatorContext*>(bd.ctx_);
  }
private:
  static THMapDeleter singleton_;
};

struct TH_API THRefcountedMapDeleter final : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override;
  static at::BoundDeleter make(THMapAllocatorContext* ctx) {
    return {&singleton_, ctx};
  }
  static THMapAllocatorContext* getContext(at::BoundDeleter bd) {
    if (bd.deleter_ != &singleton_) return nullptr;
    return static_cast<THMapAllocatorContext*>(bd.ctx_);
  }
private:
  static THRefcountedMapDeleter singleton_;
};
#endif

TH_API void THRefcountedMapAllocator_incref(THMapAllocatorContext *ctx, void *data);
TH_API int THRefcountedMapAllocator_decref(THMapAllocatorContext *ctx, void *data);

#endif
