#pragma once

#include <TH/TH.h>

#ifdef __cplusplus

#ifdef SHM_EXPORTS
# define SHM_API __declspec(dllexport)
#else
# define SHM_API __declspec(dllimport)
#endif

SHM_API void libshm_init(const char *manager_exec_path);

class THManagedMapAllocator : public THRefcountedMapAllocator {
public:
  THManagedMapAllocator(const char* manager_handle, const char* filename, int flags, ptrdiff_t size)
    : THRefcountedMapAllocator(filename, flags, size);

  static at::SupervisedPtr makeSupervisedPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size);
  static THManagedMapAllocator* fromSupervisedPtr(const at::SupervisedPtr&);

  const char* manager_handle() const { return "no_manager"; }
};

#endif
