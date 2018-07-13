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

at::DevicePtr THManagedMapAllocator::makeDevicePtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size) {
  auto* supervisor = new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {supervisor->data(), {supervisor, &deleteTHManagedMapAllocator}, at::kCPU};
}

THManagedMapAllocator* THManagedMapAllocator::fromDevicePtr(const at::DevicePtr& dptr) {
  return dptr.cast_supervisor<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}
