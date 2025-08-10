#include <include/openreg.h>

#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace openreg {
namespace internal {

class ScopedMemoryProtector {
 public:
  ScopedMemoryProtector(const orPointerAttributes& info)
      : m_info(info), m_protected(false) {
    if (m_info.type == orMemoryType::orMemoryTypeDevice) {
      if (mprotect(m_info.pointer, m_info.size, PROT_READ | PROT_WRITE) ==
          0) {
        m_protected = true;
      }
    }
  }
  ~ScopedMemoryProtector() {
    if (m_protected) {
      mprotect(m_info.pointer, m_info.size, PROT_NONE);
    }
  }
  ScopedMemoryProtector(const ScopedMemoryProtector&) = delete;
  ScopedMemoryProtector& operator=(const ScopedMemoryProtector&) = delete;

 private:
  orPointerAttributes m_info;
  bool m_protected;
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
    long page_size = sysconf(_SC_PAGESIZE);
    size_t aligned_size = ((size - 1) / page_size + 1) * page_size;
    void* mem = nullptr;
    int current_device = -1;

    if (type == orMemoryType::orMemoryTypeDevice) {
      orGetDevice(&current_device);

      mem = mmap(
          nullptr,
          aligned_size,
          PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS,
          -1,
          0);
      if (mem == MAP_FAILED)
        return orErrorUnknown;
      if (mprotect(mem, aligned_size, PROT_NONE) != 0) {
        munmap(mem, aligned_size);
        return orErrorUnknown;
      }
    } else {
      if (posix_memalign(&mem, page_size, aligned_size) != 0) {
        return orErrorUnknown;
      }
    }

    m_registry[mem] = {type, current_device, mem, aligned_size};
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
      mprotect(info.pointer, info.size, PROT_READ | PROT_WRITE);
      munmap(info.pointer, info.size);
    } else {
      ::free(info.pointer);
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
    orPointerAttributes dst_info = getPointerInfo(dst);
    orPointerAttributes src_info = getPointerInfo(src);
    switch (kind) {
      case orMemcpyHostToDevice:
        if (dst_info.type != orMemoryType::orMemoryTypeDevice ||
            src_info.type == orMemoryType::orMemoryTypeDevice)
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToHost:
        if (dst_info.type == orMemoryType::orMemoryTypeDevice ||
            src_info.type != orMemoryType::orMemoryTypeDevice)
          return orErrorUnknown;
        break;
      case orMemcpyDeviceToDevice:
        if (dst_info.type != orMemoryType::orMemoryTypeDevice ||
            src_info.type != orMemoryType::orMemoryTypeDevice)
          return orErrorUnknown;
        break;
      case orMemcpyHostToHost:
        if (dst_info.type == orMemoryType::orMemoryTypeDevice ||
            src_info.type == orMemoryType::orMemoryTypeDevice)
          return orErrorUnknown;
        break;
    }
    {
      ScopedMemoryProtector dst_protector(dst_info);
      ScopedMemoryProtector src_protector(src_info);
      ::memcpy(dst, src, count);
    }

    return orSuccess;
  }

  orError_t getPointerAttributes(
      orPointerAttributes* attributes,
      const void* ptr) {
    if (!attributes || !ptr)
      return orErrorUnknown;

    std ::lock_guard<std::mutex> lock(m_mutex);
    orPointerAttributes info = getPointerInfo(ptr);

    attributes->type = info.type;
    if (info.type == orMemoryType::orMemoryTypeUnmanaged) {
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
      attributes->size = 0;
    } else {
      attributes->device = info.device;
      attributes->pointer = info.pointer;
      attributes->size = info.size;
    }

    return orSuccess;
  }

  orError_t unprotect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    orPointerAttributes info = getPointerInfo(ptr);
    if (info.type != orMemoryType::orMemoryTypeDevice) {
      return orErrorUnknown;
    }
    if (mprotect(info.pointer, info.size, PROT_READ | PROT_WRITE) != 0) {
      return orErrorUnknown;
    }
    return orSuccess;
  }

  orError_t protect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    orPointerAttributes info = getPointerInfo(ptr);
    if (info.type != orMemoryType::orMemoryTypeDevice) {
      return orErrorUnknown;
    }
    if (mprotect(info.pointer, info.size, PROT_NONE) != 0) {
      return orErrorUnknown;
    }
    return orSuccess;
  }

 private:
  MemoryManager() = default;
  orPointerAttributes getPointerInfo(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it == m_registry.begin())
      return {};
    --it;
    const char* p_char = static_cast<const char*>(ptr);
    const char* base_char = static_cast<const char*>(it->first);
    if (p_char >= base_char && p_char < (base_char + it->second.size)) {
      return it->second;
    }
    return {};
  }
  std::map<void*, orPointerAttributes> m_registry;
  std::mutex m_mutex;
};

} // namespace internal
} // namespace openreg

orError_t orMalloc(void** devPtr, size_t size) {
  return openreg::internal::MemoryManager::getInstance().allocate(
      devPtr, size, orMemoryType::orMemoryTypeDevice);
}

orError_t orFree(void* devPtr) {
  return openreg::internal::MemoryManager::getInstance().free(devPtr);
}

orError_t orMallocHost(void** hostPtr, size_t size) {
  return openreg::internal::MemoryManager::getInstance().allocate(
      hostPtr, size, orMemoryType::orMemoryTypeHost);
}

orError_t orFreeHost(void* hostPtr) {
  return openreg::internal::MemoryManager::getInstance().free(hostPtr);
}

orError_t orMemcpy(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind) {
  return openreg::internal::MemoryManager::getInstance().memcpy(
      dst, src, count, kind);
}

orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr) {
  return openreg::internal::MemoryManager::getInstance().getPointerAttributes(
      attributes, ptr);
}

orError_t orMemoryUnprotect(void* devPtr) {
  return openreg::internal::MemoryManager::getInstance().unprotect(devPtr);
}

orError_t orMemoryProtect(void* devPtr) {
  return openreg::internal::MemoryManager::getInstance().protect(devPtr);
}
