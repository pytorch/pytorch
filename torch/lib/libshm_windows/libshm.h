#pragma once

#include <c10/core/MapAllocator.h>

#ifdef __cplusplus

#ifdef SHM_EXPORTS
#define SHM_API __declspec(dllexport)
#else
#define SHM_API __declspec(dllimport)
#endif

SHM_API void libshm_init(const char* manager_exec_path);

class SHM_API THManagedMapAllocator : public c10::RefcountedMapAllocator {
 public:
  THManagedMapAllocator(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size)
      : c10::RefcountedMapAllocator(filename, flags, size) {}

  static c10::DataPtr makeDataPtr(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size);
  static THManagedMapAllocator* fromDataPtr(const c10::DataPtr&);

  const char* manager_handle() const {
    return "no_manager";
  }
};

#endif
