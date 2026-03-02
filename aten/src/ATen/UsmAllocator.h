#pragma once

#include <string_view>

#include <c10/core/Allocator.h>
#include <ATen/MapAllocator.h>

namespace at {

// UsmAllocator - Unified Shared Memory allocator using mmap + THP + DIO
class TORCH_API UsmAllocator {
 public:
  UsmAllocator(std::string_view filename, size_t size);
  UsmAllocator(
      WithFd,
      std::string_view filename,
      int fd,
      size_t size);
  UsmAllocator(const UsmAllocator&) = delete;
  UsmAllocator& operator=(const UsmAllocator&) = delete;
  UsmAllocator(UsmAllocator&&) = delete;
  UsmAllocator& operator=(UsmAllocator&&) = delete;

  const char* filename() const {
    return filename_.c_str();
  }
  int fd() const {
#ifdef _WIN32
    TORCH_CHECK(false, "UsmAllocator::fd() is unsupported on Windows");
#else
    return fd_;
#endif
  }
  size_t size() const {
    return size_;
  }
  void* data() const {
    return base_ptr_;
  }

  static UsmAllocator* fromDataPtr(const at::DataPtr&);
  static at::DataPtr makeDataPtr(
      std::string_view filename,
      size_t size,
      size_t* actual_size_out);
  static at::DataPtr makeDataPtr(
      WithFd,
      const char* filename,
      int fd,
      size_t size,
      size_t* actual_size_out);

  void close();
  ~UsmAllocator();

 private:
  bool closed_ = false;
  std::string filename_;
  size_t size_; /* allocated size */
#ifdef _WIN32
  // Windows not supported for USM
#else
  int fd_ = -1;
#endif
  void* base_ptr_ = nullptr;
};
    
} // namespace at
