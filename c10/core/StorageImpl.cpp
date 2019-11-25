#include <c10/core/StorageImpl.h>
#include <c10/util/SmallVector.h>

constexpr int ALLOC_CACHE_SIZE = 8;

static thread_local at::SmallVector<void*, ALLOC_CACHE_SIZE> alloc_cache;

void* c10::StorageImpl::operator new(std::size_t sz) {
  AT_ASSERT(sz == sizeof(StorageImpl));
  if (alloc_cache.size()) {
    return alloc_cache.pop_back_val();
  }

  return ::operator new (sz);
}

void c10::StorageImpl::operator delete(void* ptr) {
  if (alloc_cache.size() < ALLOC_CACHE_SIZE) {
    alloc_cache.push_back(ptr);
  } else {
    return ::operator delete (ptr);
  }
}
