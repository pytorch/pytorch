#include <TH/THAllocator.h>

#include <atomic>
#if ATOMIC_INT_LOCK_FREE == 2
#define TH_ATOMIC_IPC_REFCOUNT 1
#endif

#include <c10/core/CPUAllocator.h>
#include <c10/util/Unicode.h>

/* stuff for mapped files */
#ifdef _WIN32
#include <c10/util/win32-headers.h>
#endif

#if defined(HAVE_MMAP)
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
/* end of stuff for mapped files */

at::Allocator* getTHDefaultAllocator() {
  return c10::GetCPUAllocator();
}

#define TH_ALLOC_ALIGNMENT 64

#if defined(_WIN32) || defined(HAVE_MMAP)

typedef struct {
  std::atomic<int> refcount;
} THMapInfo;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char * unknown_filename = "filename not specified";
#ifdef _WIN32
const char * unknown_eventname = "eventname not specified";
#endif

THMapAllocator::THMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size)
  : filename_(filename ? filename : unknown_filename)
  , flags_(0) // to be filled later
  , size_(0) // to be filled later
#ifdef _WIN32
  , handle_(INVALID_HANDLE_VALUE) // to be filled later
  , event_(INVALID_HANDLE_VALUE) // to be filled later
  , eventname_(filename ? std::string(filename) + "_event" : unknown_eventname)
#else
  , fd_(fd)
#endif
  , base_ptr_(nullptr)
{

  if (!(flags & TH_ALLOCATOR_MAPPED_SHARED) && !(flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
    flags &= ~TH_ALLOCATOR_MAPPED_NOCREATE;
  }
  if ((flags ^ TH_ALLOCATOR_MAPPED_EXCLUSIVE) == 0) {
    AT_ERROR("TH_ALLOCATOR_MAPPED_EXCLUSIVE flag requires opening the file in shared mode");
  }
#ifdef _WIN32
  if (fd != -1) {
    AT_ERROR("THMapAllocator_newWithFd is unsupported on Windows");
  }
#endif
  flags_ = flags;

  // OK, now do the allocation

  if (size == 0) {
    return;
  }

#ifdef _WIN32
  if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
    // Shadowing
    const wchar_t *filename;
    const wchar_t *eventname;
    const std::wstring wFilename = c10::u8u16(filename_);
    const std::wstring wEventname = c10::u8u16(eventname_);
    LARGE_INTEGER hfilesz;

    if (filename_[0] == '/') {
      filename = wFilename.c_str() + 1;
      eventname = wEventname.c_str() + 1;
    } else {
      filename = wFilename.c_str();
      eventname = wEventname.c_str();
    }

    hfilesz.QuadPart = size;

    if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
      event_ = CreateEventW(nullptr, FALSE, FALSE, eventname);
    } else if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
      event_ = OpenEventW(EVENT_ALL_ACCESS, FALSE, eventname);
    } else {
      AT_ERROR("Expected either TH_ALLOCATOR_MAPPED_EXCLUSIVE or TH_ALLOCATOR_MAPPED_NOCREATE");
    }

    if (event_ == nullptr) {
      AT_ERROR("Couldn't open shared event: <", eventname, ">, error code: <", GetLastError(), ">");
    }

    if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
      handle_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, filename);
    } else if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
      handle_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, filename);
    } else {
      AT_ERROR("Expected either TH_ALLOCATOR_MAPPED_EXCLUSIVE or TH_ALLOCATOR_MAPPED_NOCREATE");
    }

    if (handle_ == nullptr) {
      AT_ERROR("Couldn't open shared file mapping: <", filename, ">, error code: <", GetLastError(), ">");
    }

    size_ = size;
    base_ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!base_ptr_) {
      AT_ERROR("Couldn't map view of shared file <", filename, ">, error code: <", GetLastError(), ">");
    }
  } else {

    HANDLE hfile;
    HANDLE hmfile;
    LARGE_INTEGER hfilesz;

    if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
      AT_ERROR("exclusive file mapping is not supported on Windows");
    }
    if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
      AT_ERROR("file mapping without creation is not supported on Windows");
    }
    if (flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) {
      AT_ERROR("TH_ALLOCATOR_MAPPED_KEEPFD not supported on Windows");
    }
    if (flags_ & TH_ALLOCATOR_MAPPED_FROMFD) {
      AT_ERROR("TH_ALLOCATOR_MAPPED_FROMFD not supported on Windows");
    }

    // Shadowing
    const wchar_t *filename;
    const std::wstring wFilename = c10::u8u16(filename_);

    filename = wFilename.c_str();

    /* open file */
    /* FILE_FLAG_RANDOM_ACCESS ? */
    if (flags_) {
      hfile = CreateFileW(filename, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE) {
        AT_ERROR("could not open file <", filename_, "> in read-write mode; error code: <", GetLastError(), ">");
      }
    } else {
      hfile = CreateFileW(filename, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE) {
        AT_ERROR("could not open file <", filename_, "> in read-only mode; error code: <", GetLastError(), ">");
      }
    }

    if (GetFileSizeEx(hfile, &hfilesz) == 0) {
      AT_ERROR("could not get file size: <", filename_, ">; error code: <", GetLastError(), ">");
    }

    if (size > 0) {
      if (size > hfilesz.QuadPart) {
        if (flags_) {
          hfilesz.QuadPart = size;
          if (SetFilePointerEx(hfile, hfilesz, NULL, FILE_BEGIN) == 0) {
            CloseHandle(hfile);
            AT_ERROR("unable to stretch file <", filename_, "> to the right size; error code: <", GetLastError(), ">", filename_);
          }
          if (SetEndOfFile(hfile) == 0) {
            CloseHandle(hfile);
            AT_ERROR("unable to write to file <", filename_, ">; error code: <", GetLastError(), ">");
          }
        } else {
          CloseHandle(hfile);
          AT_ERROR("file <", filename_, "> size is smaller than the required mapping size <", size, ">; error code: <", GetLastError(), ">");
        }
      }
    } else {
      size = hfilesz.QuadPart;
    }

    size_ = size; /* if we are here, it must be the right size */

    hfilesz.QuadPart = size_;

    /* get map handle */
    if (flags_) {
      if ( (hmfile = CreateFileMappingW(hfile, NULL, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL ) {
        AT_ERROR("could not create a map on file <", filename_, ">; error code: <", GetLastError(), ">");
      }
    } else {
      if ( (hmfile = CreateFileMappingW(hfile, NULL, PAGE_WRITECOPY, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL ) {
        AT_ERROR("could not create a map on file <", filename_, ">; error code: <", GetLastError(), ">");
      }
    }

    /* map the stuff */
    if(flags_) {
      base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    } else {
      base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0);
    }

    CloseHandle(hfile);
    CloseHandle(hmfile);
  }
#else /* _WIN32 */
  {
    /* open file */
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int fd;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int flags; // shadow
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    struct stat file_stat;

    if (flags_ & (TH_ALLOCATOR_MAPPED_SHARED | TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
      flags = O_RDWR | O_CREAT;
    } else {
      flags = O_RDONLY;
    }

    if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
      flags |= O_EXCL;
    }
    if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
      flags &= ~O_CREAT;
    }

    if (!(flags_ & TH_ALLOCATOR_MAPPED_FROMFD)) {
      if (flags_ & TH_ALLOCATOR_MAPPED_SHARED) {
        if ((fd = open(filename_.c_str(), flags, (mode_t)0600)) == -1) {
          AT_ERROR("unable to open file <", filename_, "> in read-write mode");
        }
      } else if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_OPEN
        if((fd = shm_open(filename_.c_str(), flags, (mode_t)0600)) == -1) {
          AT_ERROR("unable to open shared memory object <", filename_, "> in read-write mode");
        }
#else
        AT_ERROR("unable to open file <", filename_, "> in sharedmem mode, shm_open unavailable on this platform");
#endif
      } else {
        if ((fd = open(filename_.c_str(), O_RDONLY)) == -1) {
          AT_ERROR("unable to open file <", filename_, "> in read-only mode");
        }
      }
    } else {
      fd = fd_;
    }

    if (fstat(fd, &file_stat) == -1) {
      if (!(flags_ & TH_ALLOCATOR_MAPPED_FROMFD)) {
        ::close(fd);
      }
      AT_ERROR("unable to stat the file <", filename_, ">");
    }

    if (size > 0) {
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      if (size > file_stat.st_size) {
        if (flags_) {
          if (ftruncate(fd, size) == -1) {
            AT_ERROR("unable to resize file <", filename_, "> to the right size");
          }
          // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
          if (fstat(fd, &file_stat) == -1 || file_stat.st_size < size) {
            ::close(fd);
            AT_ERROR("unable to stretch file <", filename_, "> to the right size");
          }
/* on macOS write returns with errno 45 (Opperation not supported) when used
 * with a file descriptor obtained via shm_open
 */
#ifndef __APPLE__
          if ((write(fd, "", 1)) != 1) /* note that the string "" contains the '\0' byte ... */ {
            ::close(fd);
            AT_ERROR("unable to write to file <", filename_, ">");
          }
#endif
        } else {
          ::close(fd);
          AT_ERROR("file <", filename_, "> size is smaller than the required mapping size <", size, ">");
        }
      }
    } else {
      size = file_stat.st_size;
    }

    size_ = size; /* if we are here, it must be the right size */

    /* map it */
    if (flags_ & (TH_ALLOCATOR_MAPPED_SHARED | TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
      base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    } else {
      base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
    }

    if (base_ptr_ == MAP_FAILED) {
      base_ptr_ = nullptr; /* let's be sure it is NULL */
      AT_ERROR("unable to mmap ", size_, " bytes from file <", filename_, ">: ", strerror(errno), " (", errno, ")");
    }

    if (flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) {
      fd_ = fd;
    } else {
      if (::close(fd) == -1) {
        AT_ERROR("Error closing file <", filename_, ">");
      }
      fd_ = -1;
    }

    if (flags_ & TH_ALLOCATOR_MAPPED_UNLINK) {
      if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_UNLINK
        if (shm_unlink(filename_.c_str()) == -1) {
          AT_ERROR("could not unlink the shared memory file ", filename_);
        }
#else
        AT_ERROR("could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif
      } else {
        if (unlink(filename_.c_str()) == -1)
          AT_ERROR("could not unlink file %s", filename_);
      }
    }

    if (base_ptr_ == MAP_FAILED) {
      AT_ERROR("$ Torch: unable to mmap memory: you tried to mmap ", size_/1073741824, " GB.");
    }
  }
#endif
  c10::reportMemoryUsageToProfiler(base_ptr_, size_, c10::Device(c10::DeviceType::CPU));
}

THMapAllocator::THMapAllocator(const char *filename, int flags, size_t size)
  : THMapAllocator(WITH_FD, filename, -1, flags, size)
{}

#ifdef _WIN32
typedef struct{
  HANDLE event;
  HANDLE handle;
  HANDLE wait;
} ReleaseContext;
static void CALLBACK WaitForReleaseHandle(PVOID lpParam, BOOLEAN TimerOrWaitFired)
{
  if (lpParam) {
    ReleaseContext *ctx = (ReleaseContext *)lpParam;

    SetEvent(ctx->event);
    CloseHandle(ctx->event);
    CloseHandle(ctx->handle);

    UnregisterWait(ctx->wait);

    THFree(ctx);
  }
}
#endif

void THMapAllocator::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  if (base_ptr_ == nullptr) {
    return;
  }
#ifdef _WIN32
  if ((flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) || (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM))
    CloseHandle(handle_);
  if(UnmapViewOfFile(base_ptr_) == 0)
    AT_ERROR("could not unmap the shared memory file");
#else /* _WIN32 */
  if (flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) {
    if (::close(fd_) == -1) {
      AT_ERROR("could not close file descriptor ", fd_);
    }
  }

  if (munmap(base_ptr_, size_)) {
    AT_ERROR("could not unmap the shared memory file");
  }

  if (!(flags_ & (TH_ALLOCATOR_MAPPED_FROMFD | TH_ALLOCATOR_MAPPED_UNLINK))) {
    if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_UNLINK
      if (shm_unlink(filename_.c_str()) == -1) {
        AT_ERROR("could not unlink the shared memory file ", filename_);
      }
#else
      AT_ERROR("could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif
    }
  }
#endif /* _WIN32 */
}

#else /* defined(_WIN32) || defined(HAVE_MMAP) */

THMapAllocator::THMapAllocator(const char *filename, int flags, size_t size) {
  AT_ERROR("file mapping not supported on your system");
}

THMapAllocator::THMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size) {
  AT_ERROR("file mapping not supported on your system");
}

void THMapAllocator::close() { }

#endif

#if (defined(_WIN32) || defined(HAVE_MMAP)) && defined(TH_ATOMIC_IPC_REFCOUNT)

THRefcountedMapAllocatorArgCheck::THRefcountedMapAllocatorArgCheck(int flags) {
  if (flags & TH_ALLOCATOR_MAPPED_FROMFD) {
    AT_ERROR("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_FROMFD flag");
  }
  if (flags & TH_ALLOCATOR_MAPPED_KEEPFD) {
    AT_ERROR("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_KEEPFD flag");
  }
  if (flags & TH_ALLOCATOR_MAPPED_UNLINK) {
    AT_ERROR("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_UNLINK flag");
  }
  if (!(flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
    AT_ERROR("THRefcountedMapAllocator requires TH_ALLOCATOR_MAPPED_SHAREDMEM flag");
  }
}

THRefcountedMapAllocator::THRefcountedMapAllocator(const char *filename, int flags, size_t size)
  : THRefcountedMapAllocatorArgCheck(flags)
  , THMapAllocator(filename, flags, size + TH_ALLOC_ALIGNMENT) {

    initializeAlloc();
}
THRefcountedMapAllocator::THRefcountedMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size)
  : THRefcountedMapAllocatorArgCheck(flags)
  , THMapAllocator(WITH_FD, filename, flags, fd, size + TH_ALLOC_ALIGNMENT) {

    initializeAlloc();
}

void THRefcountedMapAllocator::initializeAlloc() {
  THMapInfo *map_info = (THMapInfo*)base_ptr_;

#ifdef _WIN32
  ReleaseContext* r_ctx = (ReleaseContext *) THAlloc(sizeof(ReleaseContext));
  r_ctx->handle = handle_;
  r_ctx->event = event_;
  r_ctx->wait = NULL;
  BOOL can_wait = RegisterWaitForSingleObject(&r_ctx->wait, event_, WaitForReleaseHandle, (PVOID)r_ctx, INFINITE, WT_EXECUTEONLYONCE);
  if (!can_wait) {
    AT_ERROR("Couldn't register wait on event, error code: <", GetLastError(), ">");
  }
#endif

  if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
    new (&map_info->refcount) std::atomic<int>(1);
  } else {
    map_info->refcount++;
  }
}

void THRefcountedMapAllocator::close() {
  if (closed_) {
    return;
  }
  closed_ = true;

  void* data = base_ptr_;

#ifdef _WIN32
  THMapInfo *info = (THMapInfo*)data;
  if (--info->refcount == 0) {
    SetEvent(event_);
  }
  if(UnmapViewOfFile(data) == 0) {
    AT_ERROR("could not unmap the shared memory file");
  }
#else /* _WIN32 */

  THMapInfo *info = (THMapInfo*)(data);
  if (--info->refcount == 0) {
#ifdef HAVE_SHM_UNLINK
    if (shm_unlink(filename_.c_str()) == -1) {
      AT_ERROR("could not unlink the shared memory file ", filename_);
    }
#else
    AT_ERROR("could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif /* HAVE_SHM_UNLINK */
  }
  if (munmap(info, size_)) {
    AT_ERROR("could not unmap the shared memory file ", filename_);
  }
#endif /* _WIN32 */
}

void THRefcountedMapAllocator::incref()
{
  THMapInfo *map_info = static_cast<THMapInfo*>(base_ptr_);
  ++map_info->refcount;
}

int THRefcountedMapAllocator::decref()
{
  THMapInfo *map_info = static_cast<THMapInfo*>(base_ptr_);
  return --map_info->refcount == 0;
}

#else


THRefcountedMapAllocatorArgCheck::THRefcountedMapAllocatorArgCheck(int flags) {}

THRefcountedMapAllocator::THRefcountedMapAllocator(const char *filename, int flags, size_t size)
  : THRefcountedMapAllocatorArgCheck(flags),
    THMapAllocator(filename, flags, size + TH_ALLOC_ALIGNMENT)
{
  AT_ERROR("refcounted file mapping not supported on your system");
}

THRefcountedMapAllocator::THRefcountedMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size)
  : THRefcountedMapAllocatorArgCheck(flags),
    THMapAllocator(WITH_FD, filename, flags, fd, size + TH_ALLOC_ALIGNMENT)
{
  AT_ERROR("refcounted file mapping not supported on your system");
}

void THRefcountedMapAllocator::initializeAlloc() {}

void THRefcountedMapAllocator::close() {}

#endif

static void deleteTHMapAllocator(void* ptr) {
  delete static_cast<THMapAllocator*>(ptr);
}

static void deleteTHRefcountedMapAllocator(void* ptr) {
  delete static_cast<THRefcountedMapAllocator*>(ptr);
}

THMapAllocator* THMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<THMapAllocator>(&deleteTHMapAllocator);
}

THRefcountedMapAllocator* THRefcountedMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<THRefcountedMapAllocator>(&deleteTHRefcountedMapAllocator);
}

at::DataPtr THMapAllocator::makeDataPtr(const char *filename, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new THMapAllocator(filename, flags, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteTHMapAllocator, at::DeviceType::CPU};
}

at::DataPtr THMapAllocator::makeDataPtr(WithFd, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new THMapAllocator(WITH_FD, filename, fd, flags, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteTHMapAllocator, at::DeviceType::CPU};
}

at::DataPtr THRefcountedMapAllocator::makeDataPtr(const char *filename, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new THRefcountedMapAllocator(filename, flags, size);
  if (actual_size_out) *actual_size_out = context->size() - TH_ALLOC_ALIGNMENT;
  return {context->data(), context, &deleteTHRefcountedMapAllocator, at::DeviceType::CPU};
}

at::DataPtr THRefcountedMapAllocator::makeDataPtr(WithFd, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new THRefcountedMapAllocator(WITH_FD, filename, fd, flags, size);
  if (actual_size_out) *actual_size_out = context->size() - TH_ALLOC_ALIGNMENT;
  return {context->data(), context, &deleteTHRefcountedMapAllocator, at::DeviceType::CPU};
}

void* THRefcountedMapAllocator::data() const {
  return static_cast<void*>(static_cast<char*>(base_ptr_) + TH_ALLOC_ALIGNMENT);
}

THMapAllocator::~THMapAllocator() {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  close();
  c10::reportMemoryUsageToProfiler(base_ptr_, -size_, c10::Device(c10::DeviceType::CPU));
}
