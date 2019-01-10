#ifndef LIBSHM_H
#define LIBSHM_H

#include <TH/TH.h>

#ifdef __cplusplus
#define SHM_EXTERNC extern "C"
#else
#define SHM_EXTERNC
#endif

#ifdef SHM_EXPORTS
# define SHM_API SHM_EXTERNC __declspec(dllexport)
#else
# define SHM_API SHM_EXTERNC __declspec(dllimport)
#endif

typedef struct {
  char *manager_handle;
  THMapAllocatorContext *th_context;
} libshm_context;

SHM_API void libshm_init(const char *manager_exec_path);
SHM_API libshm_context * libshm_context_new(const char *manager_handle, const char *filename, int flags);
SHM_API void libshm_context_free(libshm_context *context);

SHM_API THAllocator THManagedSharedAllocator;

#endif
