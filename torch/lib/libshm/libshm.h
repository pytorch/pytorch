#ifndef LIBSHM_H
#define LIBSHM_H

#include <TH/TH.h>

#ifdef __cplusplus
#define EXPORT_API extern "C"
#else
#define EXPORT_API
#endif

typedef struct {
  char *manager_handle;
  // NB: th_context is a temporary field
  THMapAllocatorContext *th_context;
  at::BoundDeleter th_deleter;
} libshm_context;

EXPORT_API void libshm_init(const char *manager_exec_path);
EXPORT_API libshm_context * libshm_context_new(const char *manager_handle, const char *filename, int flags);
EXPORT_API at::SupervisedPtr libshm_alloc(void *_ctx, ptrdiff_t size);
EXPORT_API void libshm_context_free(libshm_context *context);

struct EXPORT_API THManagedSharedDeleter : public at::Deleter {
  void deallocate(void* ctx, void* data) const override;
  static at::BoundDeleter make(libshm_context* ctx, at::BoundDeleter deleter) {
    ctx->th_deleter = deleter;
    return {&singleton_, ctx};
  }
  static libshm_context * getContext(at::BoundDeleter bd) {
    if (bd.deleter_ != &singleton_) return nullptr;
    return static_cast<libshm_context*>(bd.ctx_);
  }
private:
  static THManagedSharedDeleter singleton_;
};

#endif
