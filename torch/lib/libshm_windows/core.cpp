#include <cstring>
#include <string>
#include <unordered_map>

#include <TH/TH.h>
#include <libshm_windows/libshm.h>


void libshm_init(const char *manager_exec_path) {
}

static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

at::DataPtr THManagedMapAllocator::makeDataPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size) {
  auto* context = new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {context->data(), context, &deleteTHManagedMapAllocator, at::kCPU};
}

THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}
