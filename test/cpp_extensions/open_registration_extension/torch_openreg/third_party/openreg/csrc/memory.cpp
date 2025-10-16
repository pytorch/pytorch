#include "memory.h"

#include <include/openreg.h>

#include <map>
#include <mutex>

namespace {

struct Block {
  orMemoryType type = orMemoryType::orMemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
  int refcount{0};
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  orError_t allocate(void** ptr, size_t size, orMemoryType type) {
    if (!ptr || size == 0)
      return orErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    long page_size = openreg::get_pagesize();
    size_t aligned_size = ((size - 1) / page_size + 1) * page_size;
    void* mem = nullptr;
    int current_device = -1;

    if (type == orMemoryType::orMemoryTypeDevice) {
      orGetDevice(&current_device);

      mem = openreg::mmap(aligned_size);
      if (mem == nullptr)
        return orErrorUnknown;
      if (openreg::mprotect(mem, aligned_size, F_PROT_NONE) != 0) {
        openreg::munmap(mem, aligned_size);
        return orErrorUnknown;
      }
    } else {
      if (openreg::alloc(&mem, page_size, aligned_size) != 0) {
        return orErrorUnknown;
      }
    }

    m_registry[mem] = {type, current_device, mem, aligned_size, 0};
    *ptr = mem;
    return orSuccess;
  }

  orError_t free(void* ptr) {
    if (!ptr)
      return orSuccess;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end())
      return orErrorUnknown;

    const auto& info = it->second;
    if (info.type == orMemoryType::orMemoryTypeDevice) {
      openreg::mprotect(info.pointer, info.size, F_PROT_READ | F_PROT_WRITE);
      openreg::munmap(info.pointer, info.size);
    } else {
      openreg::free(info.pointer);
    }

    m_registry.erase(it);
    return orSuccess;
  }

  orError_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      orMemcpyKind kind) {
    if (!dst || !src || count == 0)
      return orErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* dst_info = getBlockInfoNoLock(dst);
    Block* src_info = getBlockInfoNoLock(src);

    switch (kind) {
      case orMemcpyHostToDevice:
        if ((!dst_info || dst_info->type != orMemoryType::orMemoryTypeDevice) ||
            (src_info && src_info->type == orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToHost:
        if ((dst_info && dst_info->type == orMemoryType::orMemoryTypeDevice) ||
            (!src_info || src_info->type != orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToDevice:
        if ((!dst_info || dst_info->type != orMemoryType::orMemoryTypeDevice) ||
            (!src_info || src_info->type != orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
      case orMemcpyHostToHost:
        if ((dst_info && dst_info->type == orMemoryType::orMemoryTypeDevice) ||
            (src_info && src_info->type == orMemoryType::orMemoryTypeDevice))
          return orErrorUnknown;
        break;
    }

    unprotectNoLock(dst_info);
    unprotectNoLock(src_info);
    ::memcpy(dst, src, count);
    protectNoLock(dst_info);
    protectNoLock(src_info);

    return orSuccess;
  }

  orError_t getPointerAttributes(
      orPointerAttributes* attributes,
      const void* ptr) {
    if (!attributes || !ptr)
      return orErrorUnknown;

    std ::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);

    if (!info) {
      attributes->type = orMemoryType::orMemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->pointer;
    }

    return orSuccess;
  }

  orError_t unprotect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return unprotectNoLock(getBlockInfoNoLock(ptr));
  }

  orError_t protect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return protectNoLock(getBlockInfoNoLock(ptr));
  }

 private:
  MemoryManager() = default;

  orError_t unprotectNoLock(Block* info) {
    if (info && info->type == orMemoryType::orMemoryTypeDevice) {
      if (info->refcount == 0) {
        if (openreg::mprotect(
                info->pointer, info->size, F_PROT_READ | F_PROT_WRITE) != 0) {
          return orErrorUnknown;
        }
      }

      info->refcount++;
    }

    return orSuccess;
  }

  orError_t protectNoLock(Block* info) {
    if (info && info->type == orMemoryType::orMemoryTypeDevice) {
      if (info->refcount == 1) {
        if (openreg::mprotect(info->pointer, info->size, F_PROT_NONE) != 0) {
          return orErrorUnknown;
        }
      }

      info->refcount--;
    }

    return orSuccess;
  }

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p_char = static_cast<const char*>(ptr);
      const char* base_char = static_cast<const char*>(it->first);
      if (p_char >= base_char && p_char < (base_char + it->second.size)) {
        return &it->second;
      }
    }

    return nullptr;
  }

  std::map<void*, Block> m_registry;
  std::mutex m_mutex;
};

} // namespace

orError_t orMalloc(void** devPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      devPtr, size, orMemoryType::orMemoryTypeDevice);
}

orError_t orFree(void* devPtr) {
  return MemoryManager::getInstance().free(devPtr);
}

orError_t orMallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, orMemoryType::orMemoryTypeHost);
}

orError_t orFreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
}

orError_t orMemcpy(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind) {
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream) {
  if (!stream) {
    return orErrorUnknown;
  }

  auto& mm = MemoryManager::getInstance();

  return orLaunchKernel(
      stream, &MemoryManager::memcpy, &mm, dst, src, count, kind);
}

orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

orError_t orMemoryUnprotect(void* devPtr) {
  return MemoryManager::getInstance().unprotect(devPtr);
}

orError_t orMemoryProtect(void* devPtr) {
  return MemoryManager::getInstance().protect(devPtr);
}
