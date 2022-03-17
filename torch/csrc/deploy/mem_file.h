#pragma once

#include <c10/util/Exception.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>

namespace torch {
namespace deploy {

// Memory maps a file into the address space read-only, and manages the lifetime
// of the mapping. Here are a few use cases:
// 1. Used in the loader to read in initial image, and to inspect
// ELF files for dependencies before callling dlopen.
//
// 2. Used in unity to load the elf file.
struct MemFile {
  explicit MemFile(const char* filename_) : fd_(0), mem_(nullptr), n_bytes_(0) {
    fd_ = open(filename_, O_RDONLY);
    TORCH_CHECK(fd_ != -1, "failed to open {}: {}", filename_, strerror(errno));
    // NOLINTNEXTLINE
    struct stat s;
    if (-1 == fstat(fd_, &s)) {
      close(fd_); // destructors don't run during exceptions
      TORCH_CHECK(false, "failed to stat {}: {}", filename_, strerror(errno));
    }
    n_bytes_ = s.st_size;
    mem_ = mmap(nullptr, n_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (MAP_FAILED == mem_) {
      close(fd_);
      TORCH_CHECK(false, "failed to mmap {}: {}", filename_, strerror(errno));
    }
  }
  MemFile(const MemFile&) = delete;
  MemFile& operator=(const MemFile&) = delete;
  [[nodiscard]] const char* data() const {
    return (const char*)mem_;
  }
  ~MemFile() {
    if (mem_) {
      munmap((void*)mem_, n_bytes_);
    }
    if (fd_) {
      close(fd_);
    }
  }
  size_t size() {
    return n_bytes_;
  }
  [[nodiscard]] int fd() const {
    return fd_;
  }

 private:
  int fd_;
  void* mem_;
  size_t n_bytes_;
};

} // namespace deploy
} // namespace torch
