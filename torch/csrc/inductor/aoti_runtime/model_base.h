#pragma once
#ifdef _WIN32
#include <torch/headeronly/util/win32-headers.h>
#include <functional> // std::function
#ifdef USE_MMAP_SELF
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>

#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4

#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FAILED ((void*)-1)

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

struct Dl_info {
  char dli_fname[MAX_PATH]; /**< Filename of defining object */
  void* dli_fbase; /**< Load address of that object */
  const char* dli_sname; /**< Name of nearest lower symbol */
  void* dli_saddr; /**< Exact value of nearest symbol */
};
typedef struct Dl_info Dl_info;

int dladdr(const void* addr, Dl_info* info) {
  // only returns filename, FWIW.
  CHAR tpath[MAX_PATH];
  MEMORY_BASIC_INFORMATION mbi;
  char* path;
  char* tmp;
  size_t length;
  int ret = 0;

  if (!info)
    return 0;

  HMODULE hModule;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)addr,
          &hModule) ||
      hModule == NULL)
    return 0;

  ret = GetModuleFileNameA(hModule, (LPSTR)&tpath, MAX_PATH);
  if (!ret)
    return 0;

  path = tpath;

  length = strlen(path);
  if (length >= MAX_PATH) {
    length = MAX_PATH - 1;
    path[MAX_PATH - 1] = '\0';
  }

  tmp = path;
  while (*tmp) {
    if (*tmp == '\\')
      *tmp = '/';
    tmp++;
  }

  memcpy(info->dli_fname, path, length + 1);
  info->dli_fbase = hModule;
  info->dli_sname = NULL;
  info->dli_saddr = NULL;
  return 1;
}

static DWORD get_creation_disposition(int flags) {
  if (flags & O_CREAT) {
    if (flags & O_EXCL)
      return CREATE_NEW;
    if (flags & O_TRUNC)
      return CREATE_ALWAYS;
    return OPEN_ALWAYS;
  }
  if (flags & O_TRUNC)
    return TRUNCATE_EXISTING;
  return OPEN_EXISTING;
}

#define O_ACCMODE 03
#define O_RDONLY 00
#define O_WRONLY 01
#define O_RDWR 02

static DWORD get_access_mode(int flags) {
  switch (flags & O_ACCMODE) {
    case O_RDONLY:
      return GENERIC_READ;
    case O_WRONLY:
      return GENERIC_WRITE;
    case O_RDWR:
      return GENERIC_READ | GENERIC_WRITE;
    default:
      return GENERIC_READ;
  }
}
#ifndef O_DSYNC
#define O_DSYNC 00010000 /* used to be O_SYNC, see below */
#endif

#ifndef O_SYNC
#define __O_SYNC 04000000
#define O_SYNC (__O_SYNC | O_DSYNC)
#endif

int open(char* pathname, int flags) {
  DWORD dwDesiredAccess = get_access_mode(flags);
  DWORD dwCreationDisposition = get_creation_disposition(flags);
  DWORD dwShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
  DWORD dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;

  if (flags & O_SYNC) {
    dwFlagsAndAttributes |= FILE_FLAG_WRITE_THROUGH;
  }

  if (flags & O_SEQUENTIAL) {
    dwFlagsAndAttributes |= FILE_FLAG_SEQUENTIAL_SCAN;
  }

  if (flags & O_RANDOM) {
    dwFlagsAndAttributes |= FILE_FLAG_RANDOM_ACCESS;
  }

  HANDLE hFile = CreateFileA(
      pathname,
      dwDesiredAccess,
      dwShareMode,
      NULL,
      dwCreationDisposition,
      dwFlagsAndAttributes,
      NULL);

  if (hFile == INVALID_HANDLE_VALUE) {
    switch (GetLastError()) {
      case ERROR_FILE_NOT_FOUND:
        errno = ENOENT;
        break;
      case ERROR_PATH_NOT_FOUND:
        errno = ENOTDIR;
        break;
      case ERROR_ACCESS_DENIED:
        errno = EACCES;
        break;
      case ERROR_FILE_EXISTS:
        errno = EEXIST;
        break;
      case ERROR_TOO_MANY_OPEN_FILES:
        errno = EMFILE;
        break;
      default:
        errno = EIO;
    }
    return -1;
  }

  int fd = _open_osfhandle((intptr_t)hFile, flags);
  if (fd == -1) {
    CloseHandle(hFile);
    errno = EMFILE;
    return -1;
  }

  if (flags & O_APPEND) {
    if (_lseeki64(fd, 0, SEEK_END) == -1) {
      int saved_errno = errno;
      _close(fd);
      errno = saved_errno;
      return -1;
    }
  }

  return fd;
}

int close(int fd) {
  return _close(fd);
}

void* mmap(
    void* addr,
    size_t length,
    int prot,
    int flags,
    int fd,
    off_t offset) {
  HANDLE hFile = (HANDLE)_get_osfhandle(fd);
  if (hFile == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return MAP_FAILED;
  }

  DWORD flProtect;
  if (prot & PROT_WRITE) {
    flProtect = PAGE_READWRITE;
  } else if (prot & PROT_READ) {
    flProtect = PAGE_READONLY;
  } else {
    flProtect = PAGE_NOACCESS;
  }

  flProtect = PAGE_READONLY;

  DWORD dwDesiredAccess = 0;
  if (prot & PROT_READ)
    dwDesiredAccess |= FILE_MAP_READ;
  if (prot & PROT_WRITE)
    dwDesiredAccess |= FILE_MAP_WRITE;
  if (prot & PROT_EXEC)
    dwDesiredAccess |= FILE_MAP_EXECUTE;

  dwDesiredAccess = FILE_MAP_READ;

  SYSTEM_INFO SysInfo;
  GetSystemInfo(&SysInfo);
  DWORD dwSysGran = SysInfo.dwAllocationGranularity;

  DWORD dwFileMapStart = (offset / dwSysGran) * dwSysGran;
  DWORD dwMapViewSize = (offset % dwSysGran) + length;
  DWORD dwFileMapSize = offset + length;
  int iViewDelta = offset - dwFileMapStart;

  HANDLE hMapping =
      CreateFileMapping(hFile, NULL, flProtect, 0, dwFileMapSize, NULL);

  if (!hMapping) {
    DWORD dwErrCode = GetLastError();
    errno = EACCES;
    return MAP_FAILED;
  }

  void* lpMapAddress = MapViewOfFileEx(
      hMapping, dwDesiredAccess, 0, dwFileMapStart, dwMapViewSize, addr);
  if (!lpMapAddress) {
    DWORD dwErrCode = GetLastError();
    errno = EINVAL;
  }

  void* pData = (char*)lpMapAddress + iViewDelta;

  CloseHandle(hMapping);

  if (!lpMapAddress) {
    return MAP_FAILED;
  }

  return pData;
}

int munmap(void* addr, size_t length) {
  if (!UnmapViewOfFile(addr)) {
    errno = EINVAL;
    return -1;
  }
  return 0;
}
#endif // USE_MMAP_SELF
#else // !_WIN32
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#endif // _WIN32

#include <fcntl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#ifdef USE_MPS
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#endif // USE_MPS
#ifdef USE_XPU
#include <torch/csrc/inductor/aoti_runtime/utils_xpu.h>
#else
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#endif // USE_XPU
#include <torch/csrc/inductor/aoti_runtime/constant_type.h>

// At codegen time, we write out a binary file called constants.bin.
// We then turn the raw binary to an object file that exposes this
// symbol and link it into the final .so.
// For information on the binary format, see `man objcopy`, under
// the "binary-architecture" flag:
// https://man7.org/linux/man-pages/man1/objcopy.1.html
// todo: use #embed in C++ 23 once available
// The constants are NOT readonly because they may be mutated.
// NOLINTNEXTLINE(*array*)
extern uint8_t _binary_constants_bin_start[];
// NOLINTNEXTLINE(*array*)
extern uint8_t _binary_constants_bin_end[];

#if defined(USE_CUDA) || defined(USE_XPU)
// Compute required blob size with 64-alignment if on GPU.
#define AOTI_CONST_ALIGNMENT 64
#else
// Use 64-alignment (use something >=64)for better performance on CPU.
#define AOTI_CONST_ALIGNMENT 64
#endif

namespace torch::aot_inductor {
inline void setUsePinnedAsyncConstantsCopy(bool enabled);
inline bool usePinnedAsyncConstantsCopy();
inline void setPinnedAsyncConstantsCopyStageBufferBytes(size_t bytes);
inline size_t pinnedAsyncConstantsCopyStageBufferBytes();
} // namespace torch::aot_inductor

// S646785: env-gated logging for the AOTI constant-load pipeline. Emits when
// the AOTI_LOG_LOADING env var is set so [AOTI_LOAD] markers are greppable in
// trainer logs without rebuilding.
#define AOTI_LOG_LOADING(msg)                                                  \
  do {                                                                         \
    static const bool _aoti_log_ = std::getenv("AOTI_LOG_LOADING") != nullptr; \
    if (_aoti_log_) {                                                          \
      std::cerr << "[AOTI_LOAD] " << msg << std::endl;                         \
    }                                                                          \
  } while (0)

namespace {

using RAIIDataPtr = std::unique_ptr<void, std::function<void(void*)>>;
// PinnedStagingPool (below) owns pinned host blocks via this RAII type from
// the surrounding torch::aot_inductor namespace. Bring it in by name so the
// anonymous-namespace class body can use it unqualified.
using torch::aot_inductor::RAIIAtenTensorHandle;

#ifdef USE_CUDA

// RAII ping-pong pinned staging pool for non-blocking H2D copies of AOTI
// constants without triggering CUDA/HIP's device-wide implicit sync.
//
// Synchronous cudaMemcpy(H2D) from pageable host memory (the .so-embedded
// constants region, mmap'd by dlopen) performs a device-wide implicit
// synchronize per the CUDA/HIP spec, stalling concurrent inference streams.
//
// We avoid that by staging pageable -> pinned (CPU memcpy) and issuing
// cudaMemcpyAsync(pinned -> device) on a dedicated non-blocking stream.
// Two pinned staging buffers ping-pong via cudaEvents so CPU fill of one
// buffer overlaps GPU H2D from the other.
//
// Pinned host buffers are allocated via the stable C ABI
// aoti_torch_empty_strided_pinned, which routes through ATen's cached
// pinned host allocator (at::getPinnedMemoryAllocator). That allocator
// keeps freed blocks in a process-wide pool, so back-to-back model loads
// reuse buffers instead of paying cudaHostAlloc/cudaFreeHost per call.
//
// Total host-locked memory is bounded at 2 * AOTI_COPY_STAGE_BUFFER_BYTES
// (default 2 * 64 MiB = 128 MiB) independent of model size.
class PinnedStagingPool {
 public:
  static constexpr size_t kDefaultBufferBytes = 64ULL * 1024 * 1024;

  // Returns nullptr on pinned-host alloc / stream / event creation failure
  // so callers can fall back to the synchronous copy path.
  static std::unique_ptr<PinnedStagingPool> tryCreate(size_t buffer_bytes) {
    std::unique_ptr<PinnedStagingPool> pool(new PinnedStagingPool());
    pool->buffer_bytes_ = buffer_bytes;
    cudaError_t rc =
        cudaStreamCreateWithFlags(&pool->stream_, cudaStreamNonBlocking);
    if (rc != cudaSuccess) {
      (void)cudaGetLastError();
      AOTI_LOG_LOADING(
          "PinnedStagingPool: cudaStreamCreateWithFlags failed rc="
          << rc << " (" << cudaGetErrorString(rc)
          << "); falling back to sync copy");
      return nullptr;
    }
    const int64_t sizes = static_cast<int64_t>(buffer_bytes);
    const int64_t strides = 1;
    for (int i = 0; i < 2; ++i) {
      rc = cudaEventCreateWithFlags(&pool->events_[i], cudaEventDisableTiming);
      if (rc != cudaSuccess) {
        (void)cudaGetLastError();
        AOTI_LOG_LOADING(
            "PinnedStagingPool: cudaEventCreateWithFlags failed rc="
            << rc << "; falling back to sync copy");
        return nullptr;
      }
      AtenTensorHandle handle = nullptr;
      AOTITorchError trc = aoti_torch_empty_strided_pinned(
          /*ndim=*/1,
          &sizes,
          &strides,
          aoti_torch_dtype_uint8(),
          aoti_torch_device_type_cpu(),
          /*device_index=*/0,
          &handle);
      if (trc != AOTI_TORCH_SUCCESS || handle == nullptr) {
        AOTI_LOG_LOADING(
            "PinnedStagingPool: aoti_torch_empty_strided_pinned("
            << buffer_bytes << ") failed trc=" << trc
            << "; falling back to sync copy");
        return nullptr;
      }
      pool->stage_tensors_[i] = RAIIAtenTensorHandle(handle);
      trc = aoti_torch_get_data_ptr(
          pool->stage_tensors_[i].get(), &pool->stage_[i]);
      if (trc != AOTI_TORCH_SUCCESS || pool->stage_[i] == nullptr) {
        AOTI_LOG_LOADING(
            "PinnedStagingPool: aoti_torch_get_data_ptr failed trc="
            << trc << "; falling back to sync copy");
        return nullptr;
      }
    }
    AOTI_LOG_LOADING(
        "PinnedStagingPool: allocated 2x"
        << (buffer_bytes / (1024 * 1024))
        << " MiB pinned staging buffers via cached pinned host allocator");
    return pool;
  }

  // Chunked copy of a host source range through the ping-pong pinned
  // buffers. Multiple back-to-back calls keep the H2D stream filled.
  // Caller-owned dst must remain valid until the destructor synchronizes.
  void copyH2DViaStage(void* dst, const void* src, size_t total) {
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    size_t offset = 0;
    while (offset < total) {
      const size_t chunk = std::min(buffer_bytes_, total - offset);
      // Wait for GPU to release this staging buffer.
      AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(events_[buf_]));
      memcpy(stage_[buf_], src_bytes + offset, chunk);
      AOTI_RUNTIME_CUDA_CHECK(cudaMemcpyAsync(
          dst_bytes + offset,
          stage_[buf_],
          chunk,
          cudaMemcpyHostToDevice,
          stream_));
      AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(events_[buf_], stream_));
      buf_ ^= 1;
      offset += chunk;
    }
  }

  cudaStream_t stream() const {
    return stream_;
  }

  // Synchronize the H2D stream and surface any error from a previously issued
  // async copy (cudaMemcpyAsync failures are reported lazily at the sync).
  // Call before destroying the pool so a failed load is not silently swallowed
  // by the no-throw destructor and passed downstream as a populated buffer.
  void finish() {
    if (stream_ != nullptr) {
      AOTI_RUNTIME_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

  ~PinnedStagingPool() {
    // Best-effort cleanup: never throw from a destructor, but log rc on
    // failure so leaks are visible in production. Sync the stream FIRST so
    // no in-flight async H2D is still reading from the pinned buffers when
    // their RAIIAtenTensorHandle members run (which return the blocks to
    // ATen's cached pinned host allocator pool). Members are destroyed after
    // this body returns, in reverse declaration order; stage_tensors_ is
    // last-declared so it runs first among members.
    if (stream_ != nullptr) {
      cudaError_t rc = cudaStreamSynchronize(stream_);
      if (rc != cudaSuccess) {
        AOTI_LOG_LOADING(
            "~PinnedStagingPool: cudaStreamSynchronize failed rc="
            << rc << " (" << cudaGetErrorString(rc) << ")");
      }
    }
    for (int i = 0; i < 2; ++i) {
      if (events_[i] != nullptr) {
        cudaError_t rc = cudaEventDestroy(events_[i]);
        if (rc != cudaSuccess) {
          AOTI_LOG_LOADING(
              "~PinnedStagingPool: cudaEventDestroy buf="
              << i << " failed rc=" << rc << " (" << cudaGetErrorString(rc)
              << ")");
        }
      }
    }
    if (stream_ != nullptr) {
      cudaError_t rc = cudaStreamDestroy(stream_);
      if (rc != cudaSuccess) {
        AOTI_LOG_LOADING(
            "~PinnedStagingPool: cudaStreamDestroy failed rc="
            << rc << " (" << cudaGetErrorString(rc) << ")");
      }
    }
    // Clear any error left by best-effort cleanup.
    (void)cudaGetLastError();
  }

  PinnedStagingPool(const PinnedStagingPool&) = delete;
  PinnedStagingPool& operator=(const PinnedStagingPool&) = delete;

 private:
  PinnedStagingPool() = default;

  cudaStream_t stream_{nullptr};
  cudaEvent_t events_[2]{nullptr, nullptr};
  // Cached borrowed pointers into stage_tensors_[i] to avoid a shim call
  // per chunk in the hot path.
  void* stage_[2]{nullptr, nullptr};
  size_t buffer_bytes_{0};
  int buf_{0};
  // Owns the pinned host allocations via ATen's cached pinned host
  // allocator. Declared last so it is destroyed first among members
  // (after the destructor body has synchronized the stream).
  RAIIAtenTensorHandle stage_tensors_[2]{};
};

inline bool envFlagIsEnabled(const char* env) {
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  std::string value(env);
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value != "0" && value != "false";
}

// Returns a PinnedStagingPool when pinned async constant copies are enabled
// and host pinning succeeds. Returns nullptr otherwise so callers fall back to
// the synchronous copy path. Per-buffer size comes from
// AOTI_COPY_STAGE_BUFFER_BYTES (default 64 MiB).
inline std::unique_ptr<PinnedStagingPool> tryMakeConstantsStagingPool() {
  if (!torch::aot_inductor::usePinnedAsyncConstantsCopy()) {
    return nullptr;
  }
  const size_t explicit_buffer_bytes =
      torch::aot_inductor::pinnedAsyncConstantsCopyStageBufferBytes();
  if (explicit_buffer_bytes > 0) {
    return PinnedStagingPool::tryCreate(explicit_buffer_bytes);
  }
  // Resolve the env var once into a cached value, not the raw getenv pointer.
  // Per POSIX that pointer may be invalidated by setenv/putenv/unsetenv.
  static const size_t env_buffer_bytes = [] {
    const char* env = std::getenv("AOTI_COPY_STAGE_BUFFER_BYTES");
    size_t bytes = PinnedStagingPool::kDefaultBufferBytes;
    if (env != nullptr && env[0] != '\0') {
      try {
        auto parsed = std::stoull(env);
        if (parsed > 0) {
          bytes = static_cast<size_t>(parsed);
        }
      } catch (...) {
        // Ignore parse errors; use default.
      }
    }
    return bytes;
  }();
  return PinnedStagingPool::tryCreate(env_buffer_bytes);
}

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
RAIIDataPtr RAII_gpuMalloc(size_t num_bytes) {
#ifdef AOT_INDUCTOR_USE_CACHING_ALLOCATOR
  // Use caching allocator for allocating GPU memory
  void* data_ptr = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_cuda_caching_allocator_raw_alloc(num_bytes, &data_ptr));
  auto deleter = [](void* ptr) noexcept {
    auto err = aoti_torch_cuda_caching_allocator_raw_delete(ptr);
    if (err != AOTI_TORCH_SUCCESS) {
      std::cerr
          << "Failed to free GPU memory in AOTInductor model (caching allocator): error code "
          << err << '\n';
    }
  };
  return RAIIDataPtr(data_ptr, deleter);
#else
  // Use cudaMalloc directly for allocating GPU memory
  void* data_ptr = nullptr;
  AOTI_RUNTIME_CUDA_CHECK(cudaMalloc((void**)&data_ptr, num_bytes));
  auto deleter = [](void* ptr) noexcept {
    auto code = cudaFree(ptr);
    if (code != cudaSuccess) {
      std::cerr << "Failed to free GPU memory in AOTInductor model: "
                << cudaGetErrorString(code) << '\n';
    }
  };
  return RAIIDataPtr(data_ptr, deleter);
#endif
}

#elif defined(USE_XPU)

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
RAIIDataPtr RAII_gpuMalloc(size_t num_bytes) {
  sycl::queue* queue_ptr = nullptr;
  aoti_torch_get_current_sycl_queue((void**)&queue_ptr);
  void* data_ptr = sycl::malloc_device(num_bytes, *queue_ptr);
  auto deleter = [queue_ptr](void* ptr) { sycl::free(ptr, *queue_ptr); };
  return RAIIDataPtr(data_ptr, deleter);
}

#elif defined(USE_MPS)

RAIIDataPtr RAII_gpuMalloc(size_t num_bytes) {
  void* data_ptr = nullptr;
  aoti_torch_mps_malloc(&data_ptr, num_bytes);
  auto deleter = [](void* ptr) { aoti_torch_mps_free(ptr); };
  return RAIIDataPtr(data_ptr, deleter);
}

#endif // USE_CUDA

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
RAIIDataPtr RAII_cpuMalloc(size_t num_bytes) {
  void* data_ptr = std::malloc(num_bytes);
  if (!data_ptr) {
    throw std::bad_alloc();
  }
  auto deleter = [](void* ptr) { std::free(ptr); };
  return RAIIDataPtr(data_ptr, deleter);
}
} // anonymous namespace

namespace torch::aot_inductor {

inline std::atomic<int>& pinnedAsyncConstantsCopySetting() {
  // -1: use env fallback, 0: disabled, 1: enabled.
  static std::atomic<int> setting{-1};
  return setting;
}

inline void setUsePinnedAsyncConstantsCopy(bool enabled) {
  pinnedAsyncConstantsCopySetting().store(
      enabled ? 1 : 0, std::memory_order_relaxed);
}

inline bool usePinnedAsyncConstantsCopy() {
  const int setting =
      pinnedAsyncConstantsCopySetting().load(std::memory_order_relaxed);
  if (setting != -1) {
    return setting == 1;
  }
#ifdef USE_CUDA
  static const bool enabled_from_env = [] {
    const char* env = std::getenv("AOTI_COPY_USE_PINNED_ASYNC");
    return envFlagIsEnabled(env);
  }();
  return enabled_from_env;
#else
  return false;
#endif
}

inline std::atomic<size_t>& pinnedAsyncConstantsCopyStageBufferBytesSetting() {
  // 0: use env/default fallback. Nonzero values are bytes per staging buffer.
  static std::atomic<size_t> bytes{0};
  return bytes;
}

inline void setPinnedAsyncConstantsCopyStageBufferBytes(size_t bytes) {
  pinnedAsyncConstantsCopyStageBufferBytesSetting().store(
      bytes, std::memory_order_relaxed);
}

inline size_t pinnedAsyncConstantsCopyStageBufferBytes() {
  return pinnedAsyncConstantsCopyStageBufferBytesSetting().load(
      std::memory_order_relaxed);
}

using ConstantMap =
    std::unordered_map<std::string, MaybeOwningAtenTensorHandle>;

// valid device strs are: cpu, cuda, cuda:0, cuda:1, ...
// Update the list here if more devices are supported in the future
inline void parse_device_str(
    const std::string& device_str,
    int32_t& device_type,
    int32_t& device_idx) {
  std::regex re("(cpu|cuda|xpu|mps)(:([0-9]+))?");
  std::smatch sm;
  bool matched = std::regex_match(device_str, sm, re);
  AOTI_RUNTIME_CHECK(matched, "Invalid device: " + device_str);

  if (sm[1].str() == "cpu") {
    device_type = aoti_torch_device_type_cpu();
  } else if (sm[1].str() == "cuda") {
    device_type = aoti_torch_device_type_cuda();
#ifdef USE_XPU
  } else if (sm[1].str() == "xpu") {
    device_type = aoti_torch_device_type_xpu();
#endif
#ifdef USE_MPS
  } else if (sm[1].str() == "mps") {
    device_type = aoti_torch_device_type_mps();
#endif
  } else {
    AOTI_RUNTIME_CHECK(false, "Invalid device: " + device_str);
  }

  if (sm[3].matched) {
    device_idx = stoi(sm[3].str());
  } else {
    device_idx = -1;
  }
}

#ifdef USE_CUDA
struct AOTICudaMemcpyThrottleConfig {
  size_t chunk_size;
  int sleep_us;

  static const AOTICudaMemcpyThrottleConfig& get() {
    static AOTICudaMemcpyThrottleConfig instance = read_from_env();
    return instance;
  }

 private:
  static AOTICudaMemcpyThrottleConfig read_from_env() {
    AOTICudaMemcpyThrottleConfig cfg;
    const char* chunk_env = std::getenv("AOTI_CUDA_COPY_CHUNK_SIZE");
    const char* sleep_env = std::getenv("AOTI_CUDA_COPY_SLEEP_US");
    try {
      cfg.chunk_size = chunk_env ? std::stoull(chunk_env) : 0;
    } catch (...) {
      cfg.chunk_size = 0;
    }
    try {
      cfg.sleep_us = sleep_env ? std::stoi(sleep_env) : 0;
    } catch (...) {
      cfg.sleep_us = 0;
    }
    return cfg;
  }
};

inline void aoti_cuda_memcpy_throttled(
    void* dst,
    const void* src,
    size_t size,
    cudaMemcpyKind kind) {
  const auto& cfg = AOTICudaMemcpyThrottleConfig::get();
  if (cfg.chunk_size == 0 || cfg.chunk_size >= size) {
    AOTI_RUNTIME_CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
    return;
  }
  auto* dst_bytes = static_cast<uint8_t*>(dst);
  auto* src_bytes = static_cast<const uint8_t*>(src);
  for (size_t off = 0; off < size; off += cfg.chunk_size) {
    size_t copy_size = std::min(cfg.chunk_size, size - off);
    AOTI_RUNTIME_CUDA_CHECK(
        cudaMemcpy(dst_bytes + off, src_bytes + off, copy_size, kind));
    if (cfg.sleep_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(cfg.sleep_us));
    }
  }
}
#endif

// Defines the base class for AOTInductorModel, which is generated by the
// AOTInductor cpp codegen. Since we do not need dynamic dispatch, we rely
// on curiously recurring template pattern (CRTP) to save some runtime
// v-table overhead. The generated AOTInductorModel is specialized with
// methods such as run_impl.
template <typename Model>
class AOTInductorModelBase {
 public:
  AOTInductorModelBase(
      size_t num_inputs,
      size_t num_outputs,
      size_t num_constants,
      const std::string& device_str,
      std::optional<std::string> cubin_dir,
      bool include_weights = true)
      : inputs_info_(num_inputs),
        outputs_info_(num_outputs),
        constants_info_(num_constants),
        cubin_dir_(std::move(cubin_dir)),
        include_weights(include_weights) {
    parse_device_str(device_str, device_type_, device_idx_);

#ifdef USE_CUDA
    if (device_idx_ == -1) {
      AOTI_RUNTIME_CUDA_CHECK(cudaGetDevice(&device_idx_));
    } else {
      // If device_idx_ is passed in, we need to set the current device to it
      AOTI_RUNTIME_CUDA_CHECK(cudaSetDevice(device_idx_));
    }
#endif // USE_CUDA
#ifdef USE_XPU
    if (device_idx_ == -1) {
      aoti_torch_get_current_xpu_device(&device_idx_);
    } else {
      aoti_torch_set_current_xpu_device(device_idx_);
    }
#endif // USE_XPU
#ifdef USE_MPS
    if (device_idx_ == -1) {
      device_idx_ = 0;
    }
#endif // USE_MPS
  }

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~AOTInductorModelBase() {
#ifdef USE_CUDA
    if (run_finished_) {
      auto code = cudaEventDestroy(*run_finished_);
      if (code != cudaSuccess) {
        std::cerr << "Failed to destroy CUDA event in AOTInductor model: "
                  << cudaGetErrorString(code) << '\n';
      }
    }
#endif // USE_CUDA
#ifdef USE_XPU
    if (run_finished_) {
      (*run_finished_)->wait_and_throw();
      delete *run_finished_;
    }
#endif // USE_XPU
  }

  AOTInductorModelBase(AOTInductorModelBase&&) = delete;
  AOTInductorModelBase& operator=(AOTInductorModelBase&&) = delete;
  AOTInductorModelBase(const AOTInductorModelBase&) = delete;
  AOTInductorModelBase& operator=(const AOTInductorModelBase&) = delete;

  void run(
      AtenTensorHandle*
          input_handles, // array of input AtenTensorHandle; handles
                         // are stolen; the array itself is borrowed
      AtenTensorHandle*
          output_handles, // array for writing output AtenTensorHandle; handles
                          // will be stolen by the caller; the array itself is
                          // borrowed
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
#ifdef USE_CUDA
    if (!run_finished_) {
      cudaEvent_t run_finished = nullptr;
      AOTI_RUNTIME_CUDA_CHECK(cudaEventCreate(&run_finished));
      run_finished_.emplace(run_finished);
    }
#elif defined(USE_XPU)
    if (run_finished_) {
      (*run_finished_)->wait_and_throw();
      delete *run_finished_;
      run_finished_.reset();
    }
    if (stream == nullptr) {
      aoti_torch_get_current_xpu_stream(this->device_idx_, (void**)&stream);
    }
#else // !USE_CUDA && !USE_XPU
    run_finished_ = false;
#endif

    auto* model = static_cast<Model*>(this);
    model->run_impl(input_handles, output_handles, stream, proxy_executor);

#ifdef USE_CUDA
    AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(*run_finished_, stream));
#elif defined(USE_XPU)
    run_finished_ = std::make_optional<sycl::event*>(new sycl::event(
        static_cast<sycl::queue*>(stream)->ext_oneapi_submit_barrier()));
#else // !USE_CUDA && !USE_XPU
    run_finished_ = true;
#endif // USE_CUDA
  }

  // Non-thread-aware variant of run(). Obviously unsafe to use in a threaded
  // environment :)
  void run_single_threaded(
      AtenTensorHandle*
          input_handles, // array of input AtenTensorHandle; handles
                         // are stolen; the array itself is borrowed
      AtenTensorHandle*
          output_handles, // array for writing output AtenTensorHandle; handles
                          // will be stolen by the caller; the array itself is
                          // borrowed
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor) {
    // don't bother with any of the run_finished stuff; this is unsafe to call
    // in a threaded context
    auto* model = static_cast<Model*>(this);
    model->run_impl(input_handles, output_handles, stream, proxy_executor);
  }

  std::unordered_map<std::string, AtenTensorHandle> run_const_fold(
      DeviceStreamType stream,
      AOTIProxyExecutorHandle proxy_executor,
      bool initialization = false) {
#ifdef USE_CUDA
    if (!run_finished_) {
      cudaEvent_t run_finished = nullptr;
      AOTI_RUNTIME_CUDA_CHECK(cudaEventCreate(&run_finished));
      run_finished_.emplace(run_finished);
    }
#elif defined(USE_XPU)
    if (run_finished_) {
      (*run_finished_)->wait_and_throw();
      delete *run_finished_;
      run_finished_.reset();
    }
    if (stream == nullptr) {
      aoti_torch_get_current_xpu_stream(this->device_idx_, (void**)&stream);
    }
#else // !USE_CUDA && !USE_XPU
    run_finished_ = false;
#endif

    auto* model = static_cast<Model*>(this);
    auto folded_constants =
        model->const_run_impl(stream, proxy_executor, initialization);

    // const_run_impl returns owning raw AtenTensorHandles in the map. The
    // fallible calls below (cudaEventRecord / XPU barrier /
    // wait_for_completion) can throw; without cleanup the map's destructor
    // drops those raw handles without freeing the underlying tensors, leaking
    // folded-constant GPU memory (the container catches and keeps serving, so
    // it accumulates). Free the still-owned handles on the error path only.
    // This header is compiled into model.so, so use only the stable C ABI (no
    // c10 scope-guard).
    try {
#ifdef USE_CUDA
      AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(*run_finished_, stream));
#elif defined(USE_XPU)
      run_finished_ = std::make_optional<sycl::event*>(new sycl::event(
          static_cast<sycl::queue*>(stream)->ext_oneapi_submit_barrier()));

#else // !USE_CUDA && !USE_XPU
      run_finished_ = true;
#endif // USE_CUDA

      // Wait for the constant folding kernels to complete. The folded
      // constants may be read by inference on a different stream after
      // swap_constant_buffer(), which has no GPU synchronization.
      wait_for_completion();
    } catch (...) {
      for (auto& kv : folded_constants) {
        if (kv.second != nullptr) {
          (void)aoti_torch_delete_tensor_object(kv.second);
          kv.second = nullptr;
        }
      }
      throw;
    }

    return folded_constants;
  }

  void update_constants_from_blob(const uint8_t* weight_blob_ptr) {
#if defined(USE_MMAP_EXTERNAL)
    user_managed_mmap = const_cast<uint8_t*>(weight_blob_ptr);
    load_constants(true);
#endif
  }

  void load_constants(bool force = false) {
    did_call_load_constants_ = true;
    size_t num_constants = this->num_constants();
    size_t num_folded_constants = this->num_folded_constants();
    constants_map_->reserve(num_constants);

    // A CUDA model can still have constants on CPU,
    // so we need a separate secondary blob for them.
    std::vector<size_t> constants_internal_offset(
        num_constants - num_folded_constants);
    std::vector<size_t> aux_cpu_constants_internal_offset(
        num_constants - num_folded_constants);
    size_t blob_size = 0;
    size_t aux_cpu_blob_size = 0;
    compute_constant_blob(
        blob_size,
        constants_internal_offset,
        aux_cpu_blob_size,
        aux_cpu_constants_internal_offset);

    if (!force && !include_weights) {
      return;
    }

    // Allocate main blob
    if (blob_size > 0) {
#if defined(USE_CUDA) || defined(USE_XPU) || defined(USE_MPS)
      constant_blob_ = RAII_gpuMalloc(blob_size);
#else
      constant_blob_ = RAII_cpuMalloc(blob_size);
#endif
    }

    // Allocate secondary blob on CPU
    if (aux_cpu_blob_size > 0) {
      aux_cpu_constant_blob_ = RAII_cpuMalloc(aux_cpu_blob_size);
    }

    size_t bytes_read = 0;
    size_t main_blob_idx = 0;
    size_t aux_cpu_blob_idx = 0;

#ifdef USE_CUDA
    // Opt-in pinned async staging pool for the constant H2D copies below.
    // nullptr (default / on allocation failure) keeps the throttled
    // synchronous path.
    auto staging_pool = tryMakeConstantsStagingPool();
    PinnedStagingPool* pool_raw = staging_pool.get();
#endif

    auto _copy_start = std::chrono::steady_clock::now();
    AOTI_LOG_LOADING(
        "load_constants: starting H2D copy of " << num_constants
                                                << " constants");

    for (size_t i = 0; i < num_constants; i++) {
      bool from_folded = this->constant_from_folded(i);
      if (from_folded) {
        continue;
      }
      std::string name = this->constant_name(i);
      size_t data_size = this->constant_data_size(i);
      int32_t const_device_type = this->constant_device_type(i);
      bool device_type_matches = const_device_type == device_type_;

      // Mixed-device constants are only supported when the secondary device is
      // CPU. If a constant was compiled for a non-CPU device but we're loading
      // on a different device, we cannot safely create the tensor.
      AOTI_RUNTIME_CHECK(
          device_type_matches ||
              const_device_type == aoti_torch_device_type_cpu(),
          "Mixed-device constants are only supported when the secondary "
          "device is CPU. Constant '" +
              name +
              "' was compiled for a non-CPU device. "
              "Hint: This can happen if you compiled on GPU but are loading "
              "on CPU, which is not supported. In AOTI, you must compile and "
              "load on the same device type.");

      uint8_t* internal_ptr = nullptr;
      if (data_size != 0) {
        if (device_type_matches) {
          internal_ptr = constant_ptr(
              constants_internal_offset[main_blob_idx],
              bytes_read,
              data_size,
              /* skip_copy = */ false
#ifdef USE_CUDA
              ,
              pool_raw
#endif
          );
        } else {
          auto* aux_cpu_constants_ptr =
              static_cast<uint8_t*>(aux_cpu_constant_blob_.get());
          internal_ptr = aux_cpu_constants_ptr +
              aux_cpu_constants_internal_offset[aux_cpu_blob_idx];
          memcpy(internal_ptr, _get_constants_start() + bytes_read, data_size);
        }
      }

      // Always increment blob indices to stay in sync with
      // compute_constant_blob(), even for zero-size constants.
      if (device_type_matches) {
        main_blob_idx++;
      } else {
        aux_cpu_blob_idx++;
      }

      bytes_read += data_size;

      // Create at::Tensor from copied memory.
      auto dtype = this->constant_dtype(i);
      auto ndim = this->constant_ndim(i);
      auto size = this->constant_shape(i);
      auto stride = this->constant_stride(i);
#ifdef USE_MPS
      auto offset = this->constant_offset(i) +
          (constants_internal_offset[i] / aoti_torch_dtype_element_size(dtype));
#else
      auto offset = this->constant_offset(i);
#endif
      auto layout = this->constant_layout(i);
      auto opaque_metadata_ptr = this->opaque_metadata(i);
      auto opaque_metadata_size = this->opaque_metadata_size(i);

      AtenTensorHandle tensor_handle = nullptr;
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob_v2(
          internal_ptr,
          ndim,
          size,
          stride,
          offset,
          dtype,
          const_device_type,
          device_type_matches ? device_idx_ : 0,
          &tensor_handle,
          layout,
          opaque_metadata_ptr,
          opaque_metadata_size));
      constants_map_->emplace(std::move(name), tensor_handle);
    }
#ifdef USE_CUDA
    // Synchronize the staging stream (surfacing any async copy error) and
    // release the pinned buffers before the loaded constants are observed by
    // callers.
    if (staging_pool != nullptr) {
      staging_pool->finish();
    }
    staging_pool.reset();
#endif
    auto _copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - _copy_start)
                        .count();
    AOTI_LOG_LOADING(
        "load_constants: H2D copy completed in " << _copy_ms << " ms");
    if (constants_map_) {
      this->update_constants_array_from_map();
    }
  }

  void set_constant_blob_releasable(bool releasable) {
    constant_blob_releasable_ = releasable;
  }

  RAIIDataPtr release_constant_blob() {
    if (!constant_blob_releasable_) {
      return RAIIDataPtr{};
    }
    return std::move(constant_blob_);
  }

  RAIIDataPtr release_aux_cpu_constant_blob() {
    if (!constant_blob_releasable_) {
      return RAIIDataPtr{};
    }
    return std::move(aux_cpu_constant_blob_);
  }

  std::shared_ptr<std::vector<ConstantHandle>> get_constants_array() {
    return constants_;
  }

  int32_t get_device_type() const {
    return device_type_;
  }

  int32_t get_device_idx() const {
    return device_idx_;
  }

  uint8_t* constant_ptr(
      size_t constant_offset,
      size_t bytes_read,
      size_t data_size,
      bool skip_copy
#ifdef USE_CUDA
      ,
      PinnedStagingPool* staging_pool = nullptr
#endif
  ) {
    auto* constants_ptr = static_cast<uint8_t*>(constant_blob_.get());
    uint8_t* internal_ptr = constants_ptr + constant_offset;
    // TODO: Handle shared storage case.
    if (!skip_copy) {
#ifdef USE_XPU
      sycl::queue* queue_ptr = nullptr;
      aoti_torch_get_current_sycl_queue((void**)&queue_ptr);
      queue_ptr
          ->memcpy(internal_ptr, _get_constants_start() + bytes_read, data_size)
          .wait();
#elif USE_CUDA
      // Prefer the pinned async staging path when the caller supplied a pool
      // (AOTI_COPY_USE_PINNED_ASYNC). Otherwise fall back to the throttled
      // synchronous copy (master default).
      if (staging_pool != nullptr) {
        staging_pool->copyH2DViaStage(
            internal_ptr, _get_constants_start() + bytes_read, data_size);
      } else {
        aoti_cuda_memcpy_throttled(
            internal_ptr,
            _get_constants_start() + bytes_read,
            data_size,
            cudaMemcpyHostToDevice);
      }
#elif USE_MPS
      aoti_torch_mps_memcpy(
          constants_ptr,
          constant_offset,
          bytes_read,
          data_size,
          _get_constants_start());
      return constants_ptr;
#else
      memcpy(internal_ptr, _get_constants_start() + bytes_read, data_size);
#endif
    }
    return internal_ptr;
  }

  void compute_constant_blob(
      size_t& blob_size,
      std::vector<size_t>& constants_internal_offset,
      size_t& aux_cpu_blob_size,
      std::vector<size_t>& aux_cpu_constants_internal_offset) {
    size_t num_constants = this->num_constants();
    blob_size = 0;
    aux_cpu_blob_size = 0;
    size_t main_idx = 0;
    size_t aux_idx = 0;

    for (size_t i = 0; i < num_constants; i++) {
      if (this->constant_from_folded(i)) {
        continue;
      }

      size_t data_size = this->constant_data_size(i);
      // ok to use same AOTI_CONST_ALIGNMENT for both main and auxiliary blobs
      if (data_size % AOTI_CONST_ALIGNMENT) {
        data_size = AOTI_CONST_ALIGNMENT +
            (data_size / AOTI_CONST_ALIGNMENT) * AOTI_CONST_ALIGNMENT;
      }

      if (this->constant_device_type(i) == device_type_) {
        constants_internal_offset[main_idx++] = blob_size;
        blob_size += data_size;
      } else {
        aux_cpu_constants_internal_offset[aux_idx++] = aux_cpu_blob_size;
        aux_cpu_blob_size += data_size;
      }
    }
  }

  size_t num_inputs() const {
    return inputs_info_.size();
  }

  size_t num_outputs() const {
    return outputs_info_.size();
  }

  size_t num_constants() const {
    return constants_info_.size();
  }

  size_t num_folded_constants() const {
    size_t total_consts = this->num_constants();
    size_t folded_consts = 0;
    for (size_t i = 0; i < total_consts; i++) {
      if (this->constant_from_folded(i)) {
        folded_consts++;
      }
    }
    return folded_consts;
  }

  const char* input_name(int64_t idx) const {
    return inputs_info_.at(idx).name;
  }

  const char* output_name(int64_t idx) const {
    return outputs_info_.at(idx).name;
  }

  const char* constant_name(int64_t idx) const {
    return constants_info_.at(idx).name;
  }

  size_t constant_ndim(int64_t idx) {
    return constants_info_.at(idx).shape.size();
  }

  const int64_t* constant_shape(int64_t idx) const {
    return constants_info_.at(idx).shape.data();
  }

  const int64_t* constant_stride(int64_t idx) const {
    return constants_info_.at(idx).stride.data();
  }

  int32_t constant_dtype(int64_t idx) const {
    return constants_info_.at(idx).dtype;
  }

  int32_t constant_layout(int64_t idx) const {
    return constants_info_.at(idx).layout;
  }

  size_t constant_offset(int64_t idx) const {
    return constants_info_.at(idx).offset;
  }

  size_t constant_data_size(int64_t idx) const {
    return constants_info_.at(idx).data_size;
  }

  const char* constant_original_fqn(int64_t idx) const {
    return constants_info_.at(idx).original_fqn;
  }

  const uint8_t* opaque_metadata(int64_t idx) const {
    return constants_info_.at(idx).opaque_metadata.data();
  }

  size_t opaque_metadata_size(int64_t idx) {
    return constants_info_.at(idx).opaque_metadata.size();
  }

  bool constant_from_folded(int64_t idx) const {
    return constants_info_.at(idx).from_folded;
  }

  int32_t constant_type(int64_t idx) const {
    return constants_info_.at(idx).type;
  }

  int32_t constant_device_type(int64_t idx) const {
    return constants_info_.at(idx).device_type;
  }

  const char* get_in_spec() const {
    return in_spec_.c_str();
  }

  const char* get_out_spec() const {
    return out_spec_.c_str();
  }

  uint64_t constant_blob_size() const {
#if defined(USE_MMAP_SELF) || defined(USE_MMAP_EXTERNAL)
    const uint64_t weights_size =
        reinterpret_cast<const uint64_t*>(_binary_constants_bin_start)[0];
    return weights_size;
#else
    throw std::runtime_error{
        "constant blob size is only available for mmap'd weights"};
#endif
  }

  bool did_call_load_constants() const {
    return did_call_load_constants_;
  }

  void update_constants_array_from_map() {
    if (!constants_map_) {
      throw std::runtime_error{
          "constants_map_ was not ready when constants_ is trying to be constructed from it!"};
    }
    if (!constants_) {
      constants_ =
          std::make_shared<std::vector<ConstantHandle>>(constants_info_.size());
    } else {
      constants_->resize(constants_info_.size());
    }
    int idx = 0;
    for (const auto& info : constants_info_) {
      const auto it = constants_map_->find(info.name);
      if (it != constants_map_->end()) {
        constants_->at(idx) = ConstantHandle(it->second);
      }
      idx++;
    }
  }

  void update_constants_map(
      std::shared_ptr<ConstantMap> constants_map,
      bool remap_constants_array = true) {
    constants_map_ = std::move(constants_map);
    if (remap_constants_array) {
      update_constants_array_from_map();
    }
  }

  // This function allows us to update the constants_ that is used to look up
  // the corresponding constant tensor during runtime.
  void update_constants_array(
      std::shared_ptr<std::vector<ConstantHandle>> constants_array) {
    constants_ = std::move(constants_array);
  }

  /// Returns true if the model is complete.
  bool is_finished() {
#ifdef USE_CUDA
    if (!run_finished_) {
      throw std::runtime_error{"Model CUDA event was not initialized"};
    }

    auto event_status = cudaEventQuery(*run_finished_);
    if (event_status == cudaSuccess) {
      return true;
    } else if (event_status == cudaErrorNotReady) {
      return false;
    }

    throw std::runtime_error(
        std::string("The model did not finish successfully. Error: ") +
        cudaGetErrorString(cudaGetLastError()));
#elif defined(USE_XPU)
    if (!run_finished_) {
      throw std::runtime_error{"Model XPU event was not initialized"};
    }
    using namespace sycl::info;
    return (*run_finished_)->get_info<event::command_execution_status>() ==
        event_command_status::complete;

#else // !USE_CUDA && !USE_XPU
    return run_finished_;
#endif // USE_CUDA
  }

  /// Synchronizes completion event.
  void wait_for_completion() {
#ifdef USE_CUDA
    if (!run_finished_) {
      throw std::runtime_error{"Model event was not initialized"};
    }

    AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(*run_finished_));
#endif // USE_CUDA
#ifdef USE_XPU
    if (!run_finished_) {
      throw std::runtime_error{"Model event was not initialized"};
    }
    (*run_finished_)->wait_and_throw();
#endif
  }

 protected:
  uint8_t* _get_constants_start() {
#if defined(USE_MMAP_EXTERNAL)
    if (!user_managed_mmap) {
      throw std::runtime_error{
          "Constants are not mmap'd. Use AOTInductorModelUpdateConstantsBlob to initialize the constants first."};
    }
    // Mapped memory for weights
    return user_managed_mmap;
#endif

#ifndef USE_MMAP_SELF
    // NOLINTNEXTLINE(*const-cast*)
    return const_cast<uint8_t*>(_binary_constants_bin_start);
#else
    if (self_mmap) {
      return self_mmap;
    }
    Dl_info dl_info;
    // get pointer to constant which are appended to the binary
    AOTI_RUNTIME_CHECK(
        dladdr(__func__, &dl_info), "Can't find shared library name");
    int fd = open(dl_info.dli_fname, O_RDONLY);
    AOTI_RUNTIME_CHECK(fd >= 0, "Shared library file cannot be opened");
#ifdef _WIN32
    auto seek_result = _lseeki64(fd, 0, SEEK_END);
#else
    auto seek_result = lseek(fd, 0, SEEK_END);
#endif
    AOTI_RUNTIME_CHECK(
        seek_result >= 0, "Failed to seek to end of shared library file");
    auto weights_size =
        reinterpret_cast<const uint64_t*>(_binary_constants_bin_start)[0];
    auto magic_number =
        reinterpret_cast<const uint64_t*>(_binary_constants_bin_start)[1];
    uint64_t fsize = static_cast<uint64_t>(seek_result);
    AOTI_RUNTIME_CHECK(
        fsize >= weights_size,
        "Shared library file is smaller than embedded weights size");
    auto weights_offset = fsize - weights_size;
    AOTI_RUNTIME_CHECK(
        (weights_offset & 0x3fff) == 0,
        "weights_offset must be aligned to 16K boundary");
    auto ptr = mmap(
        NULL,
        weights_size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE,
        fd,
        weights_offset);
    close(fd);
    AOTI_RUNTIME_CHECK(ptr != MAP_FAILED, "mmap() failed");
    self_mmap = static_cast<uint8_t*>(ptr);
    AOTI_RUNTIME_CHECK(
        reinterpret_cast<uint64_t*>(
            self_mmap + weights_size - sizeof(uint64_t))[0] == magic_number,
        "Weights data seems corrupt");
    return self_mmap;
#endif
  }

  struct ParamInfo {
    const char* name = nullptr;
  };

  struct ConstInfo {
    const char* name = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int32_t dtype{};
    int32_t device_type{};
    int64_t offset{};
    size_t data_size{};
    int32_t layout{};
    std::vector<uint8_t> opaque_metadata;
    int64_t opaque_metadata_size{};
    const char* original_fqn = nullptr;
    bool from_folded{};
    int32_t type{};
  };

  std::vector<ParamInfo> inputs_info_;
  std::vector<ParamInfo> outputs_info_;
  std::vector<ConstInfo> constants_info_;
  std::string in_spec_;
  std::string out_spec_;

  std::shared_ptr<ConstantMap> constants_map_;
  std::shared_ptr<std::vector<ConstantHandle>> constants_;

  // Holds the blob storage for constants' at::Tensor.
  RAIIDataPtr constant_blob_;
  // For mixed-device models, aux_cpu_constant_blob_ holds CPU constants
  RAIIDataPtr aux_cpu_constant_blob_;

#if defined(USE_MMAP_SELF)
  // Mapped memory for weights
  uint8_t* self_mmap = NULL;
#endif

#if defined(USE_MMAP_EXTERNAL)
  // Mapped memory for weights
  uint8_t* user_managed_mmap = NULL;
#endif

  // A directory with CUDA binary files, e.g. compiled kernels, etc.
  const std::optional<std::string> cubin_dir_;

  // This is the flag that implies whether the weight is included in the model.
  // If True, we would prepare the weight when loading the model, otherwise the
  // model will be loaded without weights, and need to be provided by the user.
  bool include_weights;
  bool did_call_load_constants_{false};
  // External-constants containers borrow tensor handles from the caller; their
  // release_* calls intentionally return empty handles.
  bool constant_blob_releasable_{true};

  // Record if the model finishes an inference run so that its owning
  // AOTModelContainer can reuse this instance.
#ifdef USE_CUDA
  std::optional<cudaEvent_t> run_finished_;
#elif defined(USE_XPU)
  std::optional<sycl::event*> run_finished_;
#else // !USE_CUDA
  bool run_finished_{};
#endif

  // Generated model uses this device index to create CUDA guards.
  int32_t device_type_{};
  int32_t device_idx_{};
};

// Codegen-ed classes can derive from this to keep pointers to loaded kernels.
class AOTInductorModelKernelsBase {
 public:
  // NOLINTNEXTLINE(modernize-use-equals-default)
  virtual ~AOTInductorModelKernelsBase() {
#ifdef USE_CUDA
    for (auto mod : loaded_modules_) {
      if (mod) {
        auto err = cuModuleUnload(mod);
        if (err != CUDA_SUCCESS) {
          const char* msg = nullptr;
          cuGetErrorString(err, &msg);
          std::cerr << "Failed to unload CUDA module in AOTInductor model: "
                    << (msg ? msg : "unknown error") << '\n';
        }
      }
    }
#endif // USE_CUDA
  }

#ifdef USE_CUDA
  // Tracks CUmodule handles loaded by loadKernel() so they can be
  // properly unloaded when the model is destroyed, preventing GPU
  // code object leaks.
  std::vector<CUmodule> loaded_modules_;
#endif // USE_CUDA
};

} // namespace torch::aot_inductor
