#include <include/openreg.h>

#ifdef _WIN32
  #include <windows.h>
  #include <memoryapi.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

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
#ifdef _WIN32
      DWORD oldProtect;
      if (VirtualProtect(m_info.pointer, m_info.size, PAGE_READWRITE, &oldProtect)) {
        m_protected = true;
        m_oldProtect = oldProtect;
      }
#else
      if (mprotect(m_info.pointer, m_info.size, PROT_READ | PROT_WRITE) == 0) {
        m_protected = true;
      }
#endif
    }
  }
  ~ScopedMemoryProtector() {
    if (m_protected) {
#ifdef _WIN32
      DWORD oldProtect;
      VirtualProtect(m_info.pointer, m_info.size, PAGE_NOACCESS, &oldProtect);
#else
      mprotect(m_info.pointer, m_info.size, PROT_NONE);
#endif
    }
  }
  ScopedMemoryProtector(const ScopedMemoryProtector&) = delete;
  ScopedMemoryProtector& operator=(const ScopedMemoryProtector&) = delete;

 private:
  orPointerAttributes m_info;
  bool m_protected;
#ifdef _WIN32
  DWORD m_oldProtect;
#endif
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
    
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    size_t page_size = sysInfo.dwPageSize;
#else
    long page_size = sysconf(_SC_PAGESIZE);
#endif
    
    size_t aligned_size = ((size - 1) / page_size + 1) * page_size;
    void* mem = nullptr;
    int current_device = -1;

    if (type == orMemoryType::orMemoryTypeDevice) {
      orGetDevice(&current_device);

#ifdef _WIN32
      // Windows implementation using VirtualAlloc
      mem = VirtualAlloc(
          nullptr,
          aligned_size,
          MEM_COMMIT | MEM_RESERVE,
          PAGE_READWRITE);
      if (mem == nullptr)
        return orErrorUnknown;
      
      // Protect memory (make it inaccessible)
      DWORD oldProtect;
      if (!VirtualProtect(mem, aligned_size, PAGE_NOACCESS, &oldProtect)) {
        VirtualFree(mem, 0, MEM_RELEASE);
        return orErrorUnknown;
      }
#else
      // Unix implementation using mmap
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
#endif
    } else {
#ifdef _WIN32
      // Windows aligned allocation
      mem = _aligned_malloc(aligned_size, page_size);
      if (mem == nullptr) {
        return orErrorUnknown;
      }
#else
      // Unix aligned allocation
      if (posix_memalign(&mem, page_size, aligned_size) != 0) {
        return orErrorUnknown;
      }
#endif
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
#ifdef _WIN32
      // Windows: Unprotect then free
      DWORD oldProtect;
      VirtualProtect(info.pointer, info.size, PAGE_READWRITE, &oldProtect);
      VirtualFree(info.pointer, 0, MEM_RELEASE);
#else
      // Unix: Unprotect then unmap
      mprotect(info.pointer, info.size, PROT_READ | PROT_WRITE);
      munmap(info.pointer, info.size);
#endif
    } else {
#ifdef _WIN32
      _aligned_free(info.pointer);
#else
      ::free(info.pointer);
#endif
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
#ifdef _WIN32
    DWORD oldProtect;
    if (!VirtualProtect(info.pointer, info.size, PAGE_READWRITE, &oldProtect)) {
      return orErrorUnknown;
    }
#else
    if (mprotect(info.pointer, info.size, PROT_READ | PROT_WRITE) != 0) {
      return orErrorUnknown;
    }
#endif
    return orSuccess;
  }

  orError_t protect(void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    orPointerAttributes info = getPointerInfo(ptr);
    if (info.type != orMemoryType::orMemoryTypeDevice) {
      return orErrorUnknown;
    }
#ifdef _WIN32
    DWORD oldProtect;
    if (!VirtualProtect(info.pointer, info.size, PAGE_NOACCESS, &oldProtect)) {
      return orErrorUnknown;
    }
#else
    if (mprotect(info.pointer, info.size, PROT_NONE) != 0) {
      return orErrorUnknown;
    }
#endif
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
