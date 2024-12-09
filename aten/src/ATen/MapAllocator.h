#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/string_view.h>

namespace at {

enum MappedAllocatorModes {
  ALLOCATOR_MAPPED_SHARED = 1,
  ALLOCATOR_MAPPED_SHAREDMEM = 2,
  ALLOCATOR_MAPPED_EXCLUSIVE = 4,
  ALLOCATOR_MAPPED_NOCREATE = 8,
  ALLOCATOR_MAPPED_KEEPFD = 16,
  ALLOCATOR_MAPPED_FROMFD = 32,
  ALLOCATOR_MAPPED_UNLINK = 64
};

// Sentinel value/type to help distinguish the file descriptor constructor from
// the non-file descriptor constructor
enum WithFd { WITH_FD };

TORCH_API std::string NewProcessWideShmHandle();

class TORCH_API MapAllocator {
 public:
  MapAllocator(std::string_view filename, int flags, size_t size);
  MapAllocator(
      WithFd,
      std::string_view filename,
      int fd,
      int flags,
      size_t size);
  MapAllocator(const MapAllocator&) = delete;
  MapAllocator& operator=(const MapAllocator&) = delete;
  MapAllocator(MapAllocator&&) = delete;
  MapAllocator& operator=(MapAllocator&&) = delete;

  const char* filename() const {
    return filename_.c_str();
  }
  int fd() const {
#ifdef _WIN32
    TORCH_CHECK(false, "MapAllocator::fd() is unsupported on Windows");
#else
    return fd_;
#endif
  }
  ptrdiff_t size() const {
    return size_;
  }
  // Return a pointer to the actual data for this allocator
  // (in the case of the refcounted allocator, this is offset
  // from the base pointer.)
  virtual void* data() const {
    return base_ptr_;
  }

  int flags() const {
    return flags_;
  }

  static MapAllocator* fromDataPtr(const at::DataPtr&);
  static at::DataPtr makeDataPtr(
      std::string_view filename,
      int flags,
      size_t size,
      size_t* actual_size_out);
  static at::DataPtr makeDataPtr(
      WithFd,
      const char* filename,
      int fd,
      int flags,
      size_t size,
      size_t* actual_size_out);

  // Closes the data.  Helps us avoid destructor shenanigans
  virtual void close();

  // This is very dangerous.  You have to redefine this destructor for each
  // subclass
  virtual ~MapAllocator();

 protected:
  bool closed_ = false;
  std::string filename_;
  int flags_ = 0;
  ptrdiff_t size_; /* mapped size */
#ifdef _WIN32
  void* handle_;
  void* event_;
  std::string eventname_;
#else
  int fd_ = -1;
#endif
  void* base_ptr_ = nullptr;
};

// Base-from-member idiom
struct TORCH_API RefcountedMapAllocatorArgCheck {
  RefcountedMapAllocatorArgCheck(int flags);
};

class TORCH_API RefcountedMapAllocator : private RefcountedMapAllocatorArgCheck,
                                         public MapAllocator {
 public:
  RefcountedMapAllocator(const char* filename, int flags, size_t size);
  RefcountedMapAllocator(
      WithFd,
      const char* filename,
      int fd,
      int flags,
      size_t size);

  static RefcountedMapAllocator* fromDataPtr(const at::DataPtr&);
  RefcountedMapAllocator(const RefcountedMapAllocator&) = delete;
  RefcountedMapAllocator(RefcountedMapAllocator&&) = delete;
  RefcountedMapAllocator& operator=(const RefcountedMapAllocator&) = delete;
  RefcountedMapAllocator& operator=(RefcountedMapAllocator&&) = delete;
  static at::DataPtr makeDataPtr(
      const char* filename,
      int flags,
      size_t size,
      size_t* actual_size_out);
  static at::DataPtr makeDataPtr(
      WithFd,
      const char* filename,
      int fd,
      int flags,
      size_t size,
      size_t* actual_size_out);

  void* data() const override;

  void incref();
  int decref();
  void close() override;

  ~RefcountedMapAllocator() override {
    RefcountedMapAllocator::close();
  }

 protected:
  void checkFlags();
  void initializeAlloc();
};

} // namespace at
