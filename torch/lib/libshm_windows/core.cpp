#include <cstring>
#include <string>
#include <unordered_map>

#include <TH/TH.h>
#include "libshm.h"


void libshm_init(const char *manager_exec_path) {
}

libshm_context * libshm_context_new(const char *manager_handle, const char *filename, int flags) {
  libshm_context *ctx = new libshm_context();
  ctx->manager_handle = "no_manager";
  ctx->th_context = THMapAllocatorContext_new(filename, flags);
  return ctx;
}

void libshm_context_free(libshm_context *ctx) {
  delete ctx;
}

void * libshm_alloc(void *_ctx, ptrdiff_t size) {
  auto *ctx = (libshm_context*)_ctx;
  return THRefcountedMapAllocator.malloc(ctx->th_context, size);
}

void * libshm_realloc(void *_ctx, void *data, ptrdiff_t size) {
  THError("cannot realloc shared memory");
  return NULL;
}

void libshm_free(void *_ctx, void *data) {
  auto *ctx = (libshm_context*)_ctx;
  THRefcountedMapAllocator.free(ctx->th_context, data);
  libshm_context_free(ctx);
}

THAllocator THManagedSharedAllocator = {
  libshm_alloc,
  libshm_realloc,
  libshm_free,
};
