#include <torch/nativert/common/FileUtil.h>

#ifdef _WIN32
#include <io.h>
#define open _open
#define read _read
#define write _write
#define fileno _fileno
#define dup _dup
#else
#include <unistd.h>
#endif
#include <cerrno>

#include <c10/util/Exception.h>
#include <fmt/core.h>

namespace torch::nativert {

namespace {

int unistd_close(int fh) {
#ifdef _WIN32
  return ::_close(fh);
#else
  return ::close(fh);
#endif
}

inline void incr(ssize_t) {}
template <typename Offset>
inline void incr(ssize_t n, Offset& offset) {
  offset += static_cast<Offset>(n);
}

// Wrap call to read/pread/write/pwrite(fd, buf, count, offset?) to retry on
// incomplete reads / writes.  The variadic argument magic is there to support
// an additional argument (offset) for pread / pwrite; see the incr() functions
// above which do nothing if the offset is not present and increment it if it
// is.
template <class F, class... Offset>
ssize_t wrapFull(F f, int fd, void* buf, size_t count, Offset... offset) {
  char* b = static_cast<char*>(buf);
  ssize_t totalBytes = 0;
  ssize_t r = -1;
  do {
    r = f(fd, b, count, offset...);
    if (r == -1) {
      if (errno == EINTR) {
        continue;
      }
      return r;
    }

    totalBytes += r;
    b += r;
    count -= r;
    incr(r, offset...);
  } while (r != 0 && count); // 0 means EOF

  return totalBytes;
}

int filterCloseReturn(int r) {
  // Ignore EINTR.  On Linux, close() may only return EINTR after the file
  // descriptor has been closed, so you must not retry close() on EINTR --
  // in the best case, you'll get EBADF, and in the worst case, you'll end up
  // closing a different file (one opened from another thread).
  //
  // Interestingly enough, the Single Unix Specification says that the state
  // of the file descriptor is unspecified if close returns EINTR.  In that
  // case, the safe thing to do is also not to retry close() -- leaking a file
  // descriptor is definitely better than closing the wrong file.
  if (r == -1 && errno == EINTR) {
    return 0;
  }
  return r;
}

//  The following wrapX() functions are private functions for wrapping file-io
//  against interrupt and partial op completions.

// Wrap call to f(args) in loop to retry on EINTR
template <class F, class... Args>
ssize_t wrapNoInt(F f, Args... args) {
  ssize_t r = -1;
  do {
    r = f(std::forward<Args>(args)...);
  } while (r == -1 && errno == EINTR);
  return r;
}

} // namespace

int openNoInt(const char* name, int flags, mode_t mode) {
  // Android NDK bionic with FORTIFY has this definition:
  // https://android.googlesource.com/platform/bionic/+/9349b9e51b/libc/include/bits/fortify/fcntl.h
  // ```
  // __BIONIC_ERROR_FUNCTION_VISIBILITY
  // int open(const char* pathname, int flags, mode_t modes, ...) __overloadable
  //         __errorattr(__open_too_many_args_error);
  // ```
  // This is originally to prevent open() with incorrect parameters.
  //
  // However, combined with folly wrapNotInt, template deduction will fail.
  // In this case, we create a custom lambda to bypass the error.
  // The solution is referenced from
  // https://github.com/llvm/llvm-project/commit/0a0e411204a2baa520fd73a8d69b664f98b428ba
  //
  auto openWrapper = [&] { return open(name, flags, mode); };
  return int(wrapNoInt(openWrapper));
}

int closeNoInt(int fd) {
  return filterCloseReturn(unistd_close(fd));
}

ssize_t writeFull(int fd, const void* buf, size_t count) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return wrapFull(write, fd, const_cast<void*>(buf), count);
}

ssize_t readFull(int fd, void* buf, size_t count) {
  return wrapFull(read, fd, buf, count);
}

File::File(int fd, bool ownsFd) noexcept : fd_(fd), ownsFd_(ownsFd) {
  TORCH_CHECK(fd >= -1, "fd must be -1 or non-negative");
  TORCH_CHECK(fd != -1 || !ownsFd, "cannot own -1");
}

File::File(std::string_view name, int flags, mode_t mode)
    : fd_(::open(std::string(name).c_str(), flags, mode)), ownsFd_(false) {
  TORCH_CHECK(
      fd_ != 1,
      "open(\"",
      name,
      "\", ",
      flags,
      ", 0",
      mode,
      ") returned stdout.")
  ownsFd_ = true;
}

File::File(File&& other) noexcept : fd_(other.fd_), ownsFd_(other.ownsFd_) {
  other.release();
}

File& File::operator=(File&& other) noexcept {
  closeNoThrow();
  swap(other);
  return *this;
}

File::~File() {
  auto fd = fd_;
  if (!closeNoThrow()) { // ignore most errors
    TORCH_CHECK(
        errno != EBADF,
        "closing fd ",
        fd,
        ", it may already ",
        "have been closed. Another time, this might close the wrong FD.");
  }
}

/* static */ File File::temporary() {
  // make a temp file with tmpfile(), dup the fd, then return it in a File.
  FILE* tmpFile = tmpfile();
  TORCH_CHECK(tmpFile != nullptr, "tmpfile() failed");
  auto guard = c10::make_scope_exit([&]() { fclose(tmpFile); });

  int fd = ::dup(fileno(tmpFile));
  TORCH_CHECK(fd != -1, "dup() failed");

  return File(fd, true);
}

int File::release() noexcept {
  int released = fd_;
  fd_ = -1;
  ownsFd_ = false;
  return released;
}

void File::swap(File& other) noexcept {
  using std::swap;
  swap(fd_, other.fd_);
  swap(ownsFd_, other.ownsFd_);
}

void File::close() {
  TORCH_CHECK(closeNoThrow(), "close() failed");
}

[[nodiscard]] bool File::closeNoThrow() {
  int r = ownsFd_ ? unistd_close(fd_) : 0;
  release();
  return r == 0;
}

} // namespace torch::nativert
