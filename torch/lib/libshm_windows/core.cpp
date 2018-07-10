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

THManagedSharedDeleter THManagedSharedDeleter::singleton_;

void THManagedSharedDeleter::deallocate(void* ctx, void* data) const {
  return libshm_free(ctx, data);
}
