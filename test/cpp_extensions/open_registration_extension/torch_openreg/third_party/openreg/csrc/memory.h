#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <include/openreg.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#define F_PROT_NONE 0x0
#define F_PROT_READ 0x1
#define F_PROT_WRITE 0x2

namespace openreg {

void* mmap(size_t size) {
#if defined(_WIN32)
  return VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
  void* addr = ::mmap(
      nullptr,
      size,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  return (addr == MAP_FAILED) ? nullptr : addr;
#endif
}

void munmap(void* addr, size_t size) {
#if defined(_WIN32)
  VirtualFree(addr, 0, MEM_RELEASE);
#else
  ::munmap(addr, size);
#endif
}

int mprotect(void* addr, size_t size, int prot) {
#if defined(_WIN32)
  DWORD win_prot = 0;
  DWORD old;
  if (prot == F_PROT_NONE) {
    win_prot = PAGE_NOACCESS;
  } else {
    win_prot = PAGE_READWRITE;
  }

  return VirtualProtect(addr, size, win_prot, &old) ? 0 : -1;
#else
  int native_prot = 0;
  if (prot == F_PROT_NONE)
    native_prot = PROT_NONE;
  else {
    if (prot & F_PROT_READ)
      native_prot |= PROT_READ;
    if (prot & F_PROT_WRITE)
      native_prot |= PROT_WRITE;
  }

  return ::mprotect(addr, size, native_prot);
#endif
}

int alloc(void** mem, size_t alignment, size_t size) {
#ifdef _WIN32
  *mem = _aligned_malloc(size, alignment);
  return *mem ? 0 : -1;
#else
  return posix_memalign(mem, alignment, size);
#endif
}

void free(void* mem) {
#ifdef _WIN32
  _aligned_free(mem);
#else
  ::free(mem);
#endif
}

long get_pagesize() {
#ifdef _WIN32
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return static_cast<long>(si.dwPageSize);
#else
  return sysconf(_SC_PAGESIZE);
#endif
}

} // namespace openreg
