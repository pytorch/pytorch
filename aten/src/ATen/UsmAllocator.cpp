#include <ATen/UsmAllocator.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <mach/mach_vm.h>
#endif

#include <c10/util/error.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace at {

// ==========================================================
// Helper Functions (Internal)
// ==========================================================

namespace {

// Wrapper for read() that handles EINTR (interrupted system call).
// It attempts to read exactly `count` bytes.
static ssize_t read_full(int fd, void* buf, size_t count) {
  size_t bytes_read = 0;
  char* ptr = static_cast<char*>(buf);

  // Limit chunk size to avoid potential issues with very large reads
  constexpr size_t kMaxChunkSize = 1024UL * 1024UL * 1024UL; 

  while (bytes_read < count) {
    size_t remaining = count - bytes_read;
    size_t chunk = std::min(remaining, kMaxChunkSize);
    
    ssize_t r = ::read(fd, ptr + bytes_read, chunk);
    
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (r == 0) break; // EOF
    bytes_read += r;
  }
  return static_cast<ssize_t>(bytes_read);
}

// ----------------------------------------------------------
// Memory Allocation Strategy
// ----------------------------------------------------------

struct AllocResult {
  void* ptr;
  size_t size;
};

#ifdef __APPLE__
static AllocResult allocate_macos(size_t size) {
  mach_vm_address_t addr = 0;
  kern_return_t kr = mach_vm_allocate(mach_task_self(), &addr, size, VM_FLAGS_ANYWHERE);
  
  // TODO: Consider using superpages on macOS if available
  TORCH_WARN_ONCE("USM: macOS does not support allocating huge pages. "
                  "Performance may be suboptimal compared to the default GPU allocator.");

  if (kr != KERN_SUCCESS) {
    TORCH_CHECK(false, "USM: mach_vm_allocate failed for size ", size, " (mach error: ", kr, ")");
  }
  
  return {reinterpret_cast<void*>(addr), size};
}

static void deallocate_macos(void* ptr, size_t size) {
  if (ptr) {
    mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)ptr, size);
  }
}

#else

static AllocResult allocate_linux(size_t size) {
  constexpr size_t kHugePageSize = 2 * 1024 * 1024;
  size_t aligned_size = (size + kHugePageSize - 1) & ~(kHugePageSize - 1);

  // Allocate anonymous memory
  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (ptr == MAP_FAILED) {
    TORCH_CHECK(false, "USM: mmap failed: ", c10::utils::str_error(errno));
  }

  // Enable Transparent Huge Pages (THP) explicitly
  if (madvise(ptr, aligned_size, MADV_HUGEPAGE) != 0) {
    // This is a warning, not an error. The kernel might still back it with huge pages if configured globally.
    TORCH_WARN_ONCE("USM: madvise(MADV_HUGEPAGE) failed: ", strerror(errno));
  }

  return {ptr, aligned_size};
}

static void deallocate_linux(void* ptr, size_t size) {
  if (ptr) {
    munmap(ptr, size);
  }
}
#endif

} // namespace

// ==========================================================
// UsmAllocator Implementation
// ==========================================================

static void deleteUsmAllocator(void* ptr) {
  delete static_cast<UsmAllocator*>(ptr);
}

UsmAllocator::UsmAllocator(WithFd, std::string_view filename, int fd, size_t size)
  : filename_(filename.empty() ? "usmalloc" : filename)
  , size_(0)
#ifndef _WIN32
  , fd_(fd) // Note: ownership of externally provided fd depends on caller contract
#endif
{
#ifdef _WIN32
  TORCH_CHECK(false, "UsmAllocator is not supported on Windows");
#else
  if (size == 0) return;

  // --------------------------------------------------------
  // 1. Allocate Memory (Platform Specific)
  // --------------------------------------------------------
#ifdef __APPLE__
  AllocResult ar = allocate_macos(size);
#else
  AllocResult ar = allocate_linux(size);
#endif
  
  base_ptr_ = ar.ptr;
  size_ = ar.size;

  // --------------------------------------------------------
  // 2. Initialize Physical Memory
  // --------------------------------------------------------
  memset(base_ptr_, 0, size_);

  // --------------------------------------------------------
  // 3. IO: Read File Content (Direct IO / NoCache)
  // --------------------------------------------------------
  
  // Determine valid file descriptor
  int reading_fd = -1;
  bool close_fd_after_read = false;

  if (fd >= 0) {
    reading_fd = fd;
    close_fd_after_read = false; // Caller owns the FD
  } else if (!filename_.empty() && filename_ != "usmalloc") {
    // Try to open with Direct IO flags first where applicable
#ifdef __APPLE__
    // macOS: Standard open first, F_NOCACHE is applied via fcntl later
    reading_fd = open(filename_.data(), O_RDONLY);
#else
    // Linux: Try O_DIRECT immediately
    reading_fd = open(filename_.data(), O_RDONLY | O_DIRECT);
    if (reading_fd < 0 && errno == EINVAL) {
      // O_DIRECT fallback: failed (e.g., tmpfs), try standard open
      reading_fd = open(filename_.data(), O_RDONLY);
    }
#endif
    if (reading_fd < 0) {
      // If we cannot open the file, clean up memory and fail
      UsmAllocator::close();
      TORCH_CHECK(false, "USM: Failed to open file ", filename_, ": ", strerror(errno));
    }
    close_fd_after_read = true;
  }

  // Perform the Read if a file is present
  if (reading_fd >= 0) {
    size_t bytes_to_read = std::min(size, size_);
    ssize_t r = -1;

#ifdef __APPLE__
    // ------------------- macOS DIO Strategy -------------------
    // F_NOCACHE disables the Unified Buffer Cache (UBC) for this file.
    if (fcntl(reading_fd, F_NOCACHE, 1) == -1) {
       TORCH_WARN("USM: Failed to set F_NOCACHE on macOS: ", strerror(errno));
    }
    
    r = read_full(reading_fd, base_ptr_, bytes_to_read);

#else 
    // ------------------- Linux DIO Strategy -------------------
    // Check if O_DIRECT is active
    int fl = fcntl(reading_fd, F_GETFL);
    bool is_direct = (fl >= 0) && (fl & O_DIRECT);

    if (is_direct) {
      // O_DIRECT requirements:
      // 1. Memory address must be aligned (base_ptr_ is 2MB aligned, so OK).
      // 2. Read length usually needs sector alignment (512 bytes).
      // We align the read request up to the next 512 bytes.
      size_t aligned_read_len = (bytes_to_read + 511) & ~511;
      
      // Ensure we don't overflow the buffer size
      if (aligned_read_len > size_) aligned_read_len = size_;

      r = read_full(reading_fd, base_ptr_, aligned_read_len);
      
      // Fallback: If O_DIRECT read fails (e.g. EINVAL due to filesystem constraints)
      if (r < 0 && (errno == EINVAL || errno == EFAULT || errno == EIO)) {
         TORCH_WARN("USM: O_DIRECT read failed, falling back to buffered IO.");
         
         // Disable O_DIRECT
         fcntl(reading_fd, F_SETFL, fl & ~O_DIRECT);
         lseek(reading_fd, 0, SEEK_SET);
         
         // Retry with standard read
         r = read_full(reading_fd, base_ptr_, bytes_to_read);
      }
    } else {
      // Standard buffered read
      r = read_full(reading_fd, base_ptr_, bytes_to_read);
    }
#endif

    // Check read result
    if (r < 0) {
       int saved_errno = errno;
       if (close_fd_after_read) ::close(reading_fd);
       UsmAllocator::close(); // Cleanup memory
       TORCH_CHECK(false, "USM: Read failed for ", filename_, ": ", strerror(saved_errno));
    }

    if (close_fd_after_read) {
      ::close(reading_fd);
    }
  }

  // Report to PyTorch profiler
  c10::reportMemoryUsageToProfiler(base_ptr_, static_cast<int64_t>(size_), 0, 
                                   static_cast<size_t>(size_), c10::Device(c10::DeviceType::CPU));
#endif
}

UsmAllocator::UsmAllocator(std::string_view filename, size_t size)
  : UsmAllocator(WITH_FD, filename, -1, size)
{}

void UsmAllocator::close() {
  if (closed_) return;
  closed_ = true;
  if (!base_ptr_) return;

#ifdef __APPLE__
  deallocate_macos(base_ptr_, size_);
#else
  deallocate_linux(base_ptr_, size_);
#endif
  base_ptr_ = nullptr;
}

UsmAllocator* UsmAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<UsmAllocator>(&deleteUsmAllocator);
}

at::DataPtr UsmAllocator::makeDataPtr(std::string_view filename, size_t size, size_t* actual_size_out) {
  auto* context = new UsmAllocator(filename, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteUsmAllocator, at::DeviceType::CPU};
}

at::DataPtr UsmAllocator::makeDataPtr(WithFd, const char *filename, int fd, size_t size, size_t* actual_size_out) {
  auto* context = new UsmAllocator(WITH_FD, filename ? std::string_view(filename) : std::string_view(""), fd, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteUsmAllocator, at::DeviceType::CPU};
}

UsmAllocator::~UsmAllocator() {
  if (base_ptr_) {
     c10::reportMemoryUsageToProfiler(base_ptr_, -static_cast<ptrdiff_t>(size_), 0, 0, c10::Device(c10::DeviceType::CPU));
  }
  UsmAllocator::close();
}

} // namespace at
