#pragma once

#include <ATen/MapAllocator.h>

#ifdef __cplusplus

void libshm_init(const char* manager_exec_path);

// Superclass to run a constructor before at::RefcountedMapAllocator
class THManagedMapAllocatorInit {
 protected:
  THManagedMapAllocatorInit(const char* manager_handle, const char* filename);
  std::string manager_handle_;
};

// Like a at::RefcountedMapAllocator, but it also makes use of an external
// shared memory manager process to ensure that shared memory regions actually
// get freed in the end (even if processes lose the memory).
class THManagedMapAllocator : private THManagedMapAllocatorInit,
                              public at::RefcountedMapAllocator {
 public:
  THManagedMapAllocator(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size);

  void close() override;

  ~THManagedMapAllocator() override {
    close();
  }

  static at::DataPtr makeDataPtr(
      const char* manager_handle,
      const char* filename,
      int flags,
      size_t size);
  static THManagedMapAllocator* fromDataPtr(const at::DataPtr&);

  const char* manager_handle() const {
    return manager_handle_.c_str();
  }
};

#endif
