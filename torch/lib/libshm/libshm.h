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
  THMapAllocatorContext *th_context;
} libshm_context;

EXPORT_API void libshm_init(const char *manager_exec_path);
EXPORT_API libshm_context * libshm_context_new(const char *manager_handle, const char *filename, int flags);
EXPORT_API void libshm_context_free(libshm_context *context);

extern THAllocator THManagedSharedAllocator;

#endif
