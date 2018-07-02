#include <ATen/Allocator.h>

namespace at {

static StorageDeleterAllocator storage_deleter_allocator;
Allocator* getStorageDeleterAllocator() {
  return &storage_deleter_allocator;
}

}
