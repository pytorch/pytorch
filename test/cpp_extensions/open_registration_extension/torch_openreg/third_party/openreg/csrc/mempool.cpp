#include <include/openreg.h>

namespace openreg {
namespace internal {
    struct orMemPool {
        void* basePtr;
        size_t size;
        size_t used;
        orMemoryType type;
    };
} // namespace internal
} // namespace openreg

orError_t orMemPoolCreate(openreg::internal::orMemPool** memPool, size_t size, orMemoryType type) {
    if (!memPool || size == 0) {
        return orErrorInvalidValue;
    }

    *memPool = new openreg::internal::orMemPool();
    if (!*memPool) {
        return orErrorOutOfMemory;
    }

    (*memPool)->size = size;
    (*memPool)->used = 0;
    (*memPool)->type = type;

    if (type == orMemoryType::orMemoryTypeDevice) {
        void* devPtr = nullptr;
        orError_t err = openreg::internal::MemoryManager::getInstance().allocate(
            &devPtr, size, orMemoryType::orMemoryTypeDevice);
        if (err != orSuccess) {
            delete *memPool;
            *memPool = nullptr;
            return err;
        }
        (*memPool)->basePtr = devPtr;
    } else {
        void* hostPtr = nullptr;
        orError_t err = openreg::internal::MemoryManager::getInstance().allocate(
            &hostPtr, size, orMemoryType::orMemoryTypeHost);
        if (err != orSuccess) {
            delete *memPool;
            *memPool = nullptr;
            return err;
        }
        (*memPool)->basePtr = hostPtr;
    }

    return orSuccess;
}