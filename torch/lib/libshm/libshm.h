#pragma once

#include <TH/TH.h>

#ifdef __cplusplus

void libshm_init(const char *manager_exec_path);

// Superclass to run a constructor before THRefcountedMapAllocator
class THManagedMapAllocatorInit {
protected:
  THManagedMapAllocatorInit(const char* manager_handle, const char* filename);
  std::string manager_handle_;
};

// Like a THRefcountedMapAllocator, but it also makes use of an external
// shared memory manager process to ensure that shared memory regions actually
// get freed in the end (even if processes lose the memory).
class THManagedMapAllocator : private THManagedMapAllocatorInit, public THRefcountedMapAllocator {
public:
  THManagedMapAllocator(const char* manager_handle, const char* filename, int flags, ptrdiff_t size);

  void close() override;

  ~THManagedMapAllocator() { close(); }

  static at::DataPtr makeDataPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size);
  static THManagedMapAllocator* fromDataPtr(const at::DataPtr&);

  const char* manager_handle() const { return manager_handle_.c_str(); }
};

#endif
