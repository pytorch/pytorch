#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <string>
#include <cstring>
#include <mutex>

#include <c10/util/Exception.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "env.cuh"

namespace at::native {

// Elementwise kernel observability: enabled by setting
// PYTORCH_GPU_ELEMENTWISE_KERNEL_LOG_DIR to a non-empty writable directory path.
// When set, logs kernel call info (CSV) right before non-vectorized
// kernel launches.
static inline bool elementwise_info_enabled() {
  static bool enabled = []() {
    const char* v = std::getenv("PYTORCH_GPU_ELEMENTWISE_KERNEL_LOG_DIR");
    return v != nullptr && v[0] != '\0';
  }();
  return enabled;
}

// A header-local static can be instantiated by more than one translation unit.
// Make every instance own and close its FILE instead of leaking descriptors at
// process exit; construction is still lazy and once per such instance.
class ElementwiseLogStream final {
 public:
  explicit ElementwiseLogStream(FILE* stream) : stream_(stream) {}
  ElementwiseLogStream(const ElementwiseLogStream&) = delete;
  ElementwiseLogStream& operator=(const ElementwiseLogStream&) = delete;
  ~ElementwiseLogStream() {
    if (stream_ != nullptr) {
      std::fclose(stream_);
    }
  }

  FILE* get() const {
    return stream_;
  }

 private:
  FILE* stream_;
};

// Log destination: resolved once per process. Only called when
// PYTORCH_GPU_ELEMENTWISE_KERNEL_LOG_DIR is set (elementwise_info_enabled).
// Directory and elementwise.log are created on demand.
static inline FILE* elementwise_log_stream() {
  static ElementwiseLogStream stream_holder = []() {
    const char* dir = std::getenv("PYTORCH_GPU_ELEMENTWISE_KERNEL_LOG_DIR");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    TORCH_CHECK(!ec, "Failed to create directory ", dir, ": ", ec.message());

    const std::string path = std::string(dir) + "/elementwise.log";
    const bool existed = std::filesystem::exists(path, ec);
    TORCH_CHECK(!ec, "Failed to inspect ", path, ": ", ec.message());
    FILE* f = std::fopen(path.c_str(), "ab");
    TORCH_CHECK(f != nullptr, "Failed to open ", path, ": ", std::strerror(errno));
#ifndef _WIN32
    if (!existed) {
      using perms = std::filesystem::perms;
      std::filesystem::permissions(
          path, perms::owner_read | perms::owner_write,
          std::filesystem::perm_options::replace, ec);
      TORCH_CHECK(!ec, "Failed to set owner-only permissions on ", path, ": ", ec.message());
    }
#endif
    return ElementwiseLogStream(f);
  }();
  return stream_holder.get();
}

static inline void warn_elementwise_log_lock_once(const char* message) {
  static bool emitted = false;
  if (!emitted) {
    emitted = true;
    std::fprintf(stderr, "GPU elementwise log: %s\\n", message);
  }
}

template <typename func_t>
inline void log_elementwise_info(TensorIteratorBase& iter, const char* path_tag, const func_t& f) {
  if (C10_UNLIKELY(elementwise_info_enabled())) {
    FILE* out = elementwise_log_stream();
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);

    int ninputs = iter.ninputs();
    int ndim = iter.ndim();
    int ntensors = 1 + ninputs;

    std::string kernel_name = typeid(f).name();
    std::replace(kernel_name.begin(), kernel_name.end(), ',', '_');

    std::string str;
    str += local_rank();
    str += ",";
    str += path_tag;
    str += ",";
    str += kernel_name;
    str += ",";
    str += std::to_string(ninputs);
    str += ",";
    str += std::to_string(ndim);
    str += ",";
    for (int d = 0; d < ndim; d++) {
      str += std::to_string(iter.shape()[d]);
      str += ",";
    }
    for (int t = 0; t < ntensors; t++) {
      for (int d = 0; d < ndim; d++) {
        str += std::to_string(iter.strides(t)[d]);
        str += ",";
      }
    }
    // Per-operand compact flags: original dim, cpu_scalar, and the raw
    // pre-normalization shape/strides. After TensorIterator normalizes and
    // coalesces, 0-dim scalars and 1-dim stride-0 have identical
    // shape/stride (both shape=64,stride=0), making them indistinguishable,
    // and the normalized blocks above keep only the coalesced common shape
    // and broadcast-zero byte strides, losing each operand's own layout
    // (e.g. [2,3] contiguous coalesces to [6]). odim tells 0-dim scalars
    // apart; oshape/ostrides keep the pre-normalization per-operand shape
    // and byte strides. Uses tensor_base() (safe on CUDA;
    // original_tensor() not set there).
    // No tag prefix: values at fixed positions after strides (odim group,
    // then oshape group, then ostrides group, then cpuscal group; oshape
    // and ostrides have odim[t] values per operand in the operand's own
    // dim order, strides in bytes same unit as the normalized stride
    // block); column names live in the pattern CSV header.
    for (int t = 0; t < ntensors; t++) {
      str += std::to_string(iter.operand(t).tensor_base().dim());
      str += ",";
    }
    for (int t = 0; t < ntensors; t++) {
      const auto& tb = iter.operand(t).tensor_base();
      for (auto s : tb.sizes()) {
        str += std::to_string(s);
        str += ",";
      }
    }
    for (int t = 0; t < ntensors; t++) {
      const auto& tb = iter.operand(t).tensor_base();
      for (auto s : tb.strides()) {
        str += std::to_string(s * (int64_t)tb.element_size());
        str += ",";
      }
    }
    for (int t = 0; t < ntensors; t++) {
      str += iter.is_cpu_scalar(t) ? "1" : "0";
      str += ",";
    }
    for (int t = 0; t < ntensors; t++) {
      str += c10::toString(iter.dtype(t));
      str += ",";
    }

    // Cross-process mutual exclusion: with torchrun all ranks share one
    // log file, and writes to a regular file are NOT atomic (PIPE_BUF
    // atomicity only applies to pipes), so concurrent writes can interleave
    // mid-line. fcntl record locks are process-associated and therefore
    // effective even though all ranks have independent fds (flock() would
    // not be). The std::mutex above covers threads within this process.
    //
    // Kernel CSV goes to PYTORCH_GPU_ELEMENTWISE_KERNEL_LOG_DIR/elementwise.log,
    // separating kernel metadata from other framework logs on stdout.
#ifdef _WIN32
    const intptr_t os_handle = _get_osfhandle(_fileno(out));
    OVERLAPPED overlapped{};
    if (os_handle == -1 ||
        !LockFileEx(reinterpret_cast<HANDLE>(os_handle), LOCKFILE_EXCLUSIVE_LOCK,
                    0, MAXDWORD, MAXDWORD, &overlapped)) {
      warn_elementwise_log_lock_once("Windows file lock failed; writing without inter-process lock");
      std::fprintf(out, "%s\n", str.c_str());
      std::fflush(out);
    } else {
      std::fprintf(out, "%s\n", str.c_str());
      std::fflush(out);
      if (!UnlockFileEx(reinterpret_cast<HANDLE>(os_handle), 0, MAXDWORD, MAXDWORD, &overlapped)) {
        warn_elementwise_log_lock_once("Windows file unlock failed");
      }
    }
#else
    const int out_fd = fileno(out);
    struct stat status {};
    const bool regular_file = fstat(out_fd, &status) == 0 && S_ISREG(status.st_mode);
    if (!regular_file) {
      // Pipes and terminals do not support advisory record locks; one stdio
      // write remains acceptable after the in-process mutex above.
      std::fprintf(out, "%s\n", str.c_str());
      std::fflush(out);
    } else {
      struct flock fl {};
      fl.l_type = F_WRLCK;
      fl.l_whence = SEEK_SET;
      fl.l_start = 0;
      fl.l_len = 0;  // whole file
      int lock_result = 0;
      do {
        lock_result = fcntl(out_fd, F_SETLKW, &fl);
      } while (lock_result == -1 && errno == EINTR);
      if (lock_result == 0) {
        std::fprintf(out, "%s\n", str.c_str());
        std::fflush(out);
        fl.l_type = F_UNLCK;
        if (fcntl(out_fd, F_SETLKW, &fl) == -1) {
          warn_elementwise_log_lock_once("POSIX file unlock failed");
        }
      } else {
        warn_elementwise_log_lock_once("POSIX file lock failed on a regular file; writing unlocked");
        std::fprintf(out, "%s\n", str.c_str());
        std::fflush(out);
      }
    }
#endif
  }
}

} // namespace at::native
