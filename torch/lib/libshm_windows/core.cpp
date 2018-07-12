#include <cstring>
#include <string>
#include <unordered_map>

#include <TH/TH.h>
#include "libshm.h"


void libshm_init(const char *manager_exec_path) {
}

static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

at::SupervisedPtr THManagedMapAllocator::makeSupervisedPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size) {
  auto* supervisor = new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {supervisor->data(), {supervisor, &deleteTHManagedMapAllocator}, at::kCPU};
}

THManagedMapAllocator* THManagedMapAllocator::fromSupervisedPtr(const at::SupervisedPtr& sptr) {
  if (sptr.supervisor_.get_deleter() != &deleteTHManagedMapAllocator) return nullptr;
  return static_cast<THManagedMapAllocator*>(sptr.supervisor_.get());
}
