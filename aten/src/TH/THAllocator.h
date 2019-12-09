#pragma once

#include <TH/THGeneral.h>

#include <c10/core/Allocator.h>

#define TH_ALLOCATOR_MAPPED_SHARED 1
#define TH_ALLOCATOR_MAPPED_SHAREDMEM 2
#define TH_ALLOCATOR_MAPPED_EXCLUSIVE 4
#define TH_ALLOCATOR_MAPPED_NOCREATE 8
#define TH_ALLOCATOR_MAPPED_KEEPFD 16
#define TH_ALLOCATOR_MAPPED_FROMFD 32
#define TH_ALLOCATOR_MAPPED_UNLINK 64

/* default malloc/free allocator. malloc and realloc raise an error (using
 * THError) on allocation failure.
 */
TH_API c10::Allocator* getTHDefaultAllocator(void);

// Sentinel value/type to help distinguish the file descriptor constructor from
// the non-file descriptor constructor
enum WithFd { WITH_FD };

class CAFFE2_API THMapAllocator {
 public:
  THMapAllocator(const char *filename, int flags, size_t size);
  THMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size);
  THMapAllocator(const THMapAllocator&) = delete;
  THMapAllocator& operator=(const THMapAllocator&) = delete;
  THMapAllocator(THMapAllocator&&) = delete;
  THMapAllocator& operator=(THMapAllocator&&) = delete;

  const char* filename() const { return filename_.c_str(); }
  int fd() const {
#ifdef _WIN32
    AT_ERROR("THMapAllocator::fd() is unsupported on Windows");
#else
    return fd_;
#endif
  }
  ptrdiff_t size() const { return size_; }
  // Return a pointer to the actual data for this allocator
  // (in the case of the refcounted allocator, this is offset
  // from the base pointer.)
  virtual void* data() const { return base_ptr_; }

  static THMapAllocator* fromDataPtr(const at::DataPtr&);
  static at::DataPtr makeDataPtr(const char *filename, int flags, size_t size, size_t* actual_size_out);
  static at::DataPtr makeDataPtr(WithFd, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out);

  // Closes the data.  Helps us avoid destructor shenanigans
  virtual void close();

  // This is very dangerous.  You have to redefine this destructor for each
  // subclass
  virtual ~THMapAllocator() { close(); }

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
  void *base_ptr_ = nullptr;
};

// Base-from-member idiom
struct CAFFE2_API THRefcountedMapAllocatorArgCheck {
  THRefcountedMapAllocatorArgCheck(int flags);
};

class CAFFE2_API THRefcountedMapAllocator
    : private THRefcountedMapAllocatorArgCheck,
      public THMapAllocator {
 public:
  THRefcountedMapAllocator(const char *filename, int flags, size_t size);
  THRefcountedMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size);

  static THRefcountedMapAllocator* fromDataPtr(const at::DataPtr&);
  static at::DataPtr makeDataPtr(const char *filename, int flags, size_t size, size_t* actual_size_out);
  static at::DataPtr makeDataPtr(WithFd, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out);

  void* data() const override;

  void incref();
  int decref();
  void close() override;

  virtual ~THRefcountedMapAllocator() { close(); }

protected:
  void checkFlags();
  void initializeAlloc();
};
