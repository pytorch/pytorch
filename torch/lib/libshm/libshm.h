#pragma once

#include <TH/TH.h>

#ifdef __cplusplus

void libshm_init(const char *manager_exec_path);

// Like a THRefcountedMapAllocator, but it also makes use of an external
// shared memory manager process to ensure that shared memory regions actually
// get freed in the end (even if processes lose the memory).
class THManagedMapAllocator : public THRefcountedMapAllocator {
public:
  THManagedMapAllocator(const char* manager_handle, const char* filename, int flags, ptrdiff_t size);

  void close() override;

  static at::SupervisedPtr makeSupervisedPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size);
  static THManagedMapAllocator* fromSupervisedPtr(const at::SupervisedPtr&);

  const char* manager_handle() const { return manager_handle_.c_str(); }

protected:
  void initializeManager();
  std::string manager_handle_;
};

#endif
