#pragma once

/*
 * Ported from folly/FileUtil.h
 */
#include <limits>
#include <string_view>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

// Copied from folly/portability/SysTypes.h
#ifdef _WIN32
#include <basetsd.h>

// This is a massive pain to have be an `int` due to the pthread implementation
// we support, but it's far more compatible with the rest of the windows world
// as an `int` than it would be as a `void*`
using pid_t = int;

using uid_t = int;
using gid_t = int;

// This isn't actually supposed to be defined here, but it's the most
// appropriate place without defining a portability header for stdint.h
// with just this single typedef.
using ssize_t = SSIZE_T;

#ifndef HAVE_MODE_T
#define HAVE_MODE_T 1
// The Windows headers don't define this anywhere, nor do any of the libs
// that Folly depends on, so define it here.
using mode_t = unsigned int;
#endif

// Copied from folly/portability/Fcntl.h
#define O_CLOEXEC _O_NOINHERIT
#endif

#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>

namespace torch::nativert {
class File {
 public:
  /**
   * Creates an empty File object, for late initialization.
   */
  constexpr File() noexcept : fd_(-1), ownsFd_(false) {}

  /**
   * Create a File object from an existing file descriptor.
   *
   * @param fd Existing file descriptor
   * @param ownsFd Takes ownership of the file descriptor if ownsFd is true.
   */
  explicit File(int fd, bool ownsFd = false) noexcept;

  /**
   * Open and create a file object.  Throws on error.
   * Owns the file descriptor implicitly.
   */
  explicit File(
      std::string_view name,
      int flags = O_RDONLY,
      mode_t mode = 0666);

  ~File();

  /**
   * Create and return a temporary, owned file (uses tmpfile()).
   */
  static File temporary();

  /**
   * Return the file descriptor, or -1 if the file was closed.
   */
  int fd() const {
    return fd_;
  }

  /**
   * Returns 'true' iff the file was successfully opened.
   */
  explicit operator bool() const {
    return fd_ != -1;
  }

  /**
   * If we own the file descriptor, close the file and throw on error.
   * Otherwise, do nothing.
   */
  void close();

  /**
   * Closes the file (if owned).  Returns true on success, false (and sets
   * errno) on error.
   */
  bool closeNoThrow();

  /**
   * Returns and releases the file descriptor; no longer owned by this File.
   * Returns -1 if the File object didn't wrap a file.
   */
  int release() noexcept;

  /**
   * Swap this File with another.
   */
  void swap(File& other) noexcept;

  // movable
  File(File&&) noexcept;
  File& operator=(File&&) noexcept;

 private:
  // unique
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  int fd_;
  bool ownsFd_;
};

/**
 * Convenience wrappers around some commonly used system calls.  The *NoInt
 * wrappers retry on EINTR.  The *Full wrappers retry on EINTR and also loop
 * until all data is written.  Note that *Full wrappers weaken the thread
 * semantics of underlying system calls.
 */
int openNoInt(const char* name, int flags, mode_t mode = 0666);
int closeNoInt(int fd);

/**
 * Similar to readFull and preadFull above, wrappers around write() and
 * pwrite() that loop until all data is written.
 *
 * Generally, the write() / pwrite() system call may always write fewer bytes
 * than requested, just like read().  In certain cases (such as when writing to
 * a pipe), POSIX provides stronger guarantees, but not in the general case.
 * For example, Linux (even on a 64-bit platform) won't write more than 2GB in
 * one write() system call.
 *
 * Note that writevFull and pwritevFull require iov to be non-const, unlike
 * writev and pwritev.  The contents of iov after these functions return
 * is unspecified.
 *
 * These functions return -1 on error, or the total number of bytes written
 * (which is always the same as the number of requested bytes) on success.
 */
ssize_t writeFull(int fd, const void* buf, size_t count);

/**
 * Wrapper around read() (and pread()) that, in addition to retrying on
 * EINTR, will loop until all data is read.
 *
 * This wrapper is only useful for blocking file descriptors (for non-blocking
 * file descriptors, you have to be prepared to deal with incomplete reads
 * anyway), and only exists because POSIX allows read() to return an incomplete
 * read if interrupted by a signal (instead of returning -1 and setting errno
 * to EINTR).
 *
 * Note that this wrapper weakens the thread safety of read(): the file pointer
 * is shared between threads, but the system call is atomic.  If multiple
 * threads are reading from a file at the same time, you don't know where your
 * data came from in the file, but you do know that the returned bytes were
 * contiguous.  You can no longer make this assumption if using readFull().
 * You should probably use pread() when reading from the same file descriptor
 * from multiple threads simultaneously, anyway.
 *
 * Note that readvFull and preadvFull require iov to be non-const, unlike
 * readv and preadv.  The contents of iov after these functions return
 * is unspecified.
 */
[[nodiscard]] ssize_t readFull(int fd, void* buf, size_t count);

/**
 * Read entire file (if num_bytes is defaulted) or no more than
 * num_bytes (otherwise) into container *out. The container is assumed
 * to be contiguous, with element size equal to 1, and offer size(),
 * reserve(), and random access (e.g. std::vector<char>, std::string,
 * fbstring).
 *
 * Returns: true on success or false on failure. In the latter case
 * errno will be set appropriately by the failing system primitive.
 */
template <class Container>
bool readFile(
    int fd,
    Container& out,
    size_t num_bytes = std::numeric_limits<size_t>::max()) {
  static_assert(
      sizeof(out[0]) == 1,
      "readFile: only containers with byte-sized elements accepted");

  size_t soFar = 0; // amount of bytes successfully read
  auto guard = c10::make_scope_exit([&]() {
    assert(out.size() >= soFar); // resize better doesn't throw
    out.resize(soFar);
  });

  // Obtain file size:
  struct stat buf;
  if (fstat(fd, &buf) == -1) {
    return false;
  }
  // Some files (notably under /proc and /sys on Linux) lie about
  // their size, so treat the size advertised by fstat under advise
  // but don't rely on it. In particular, if the size is zero, we
  // should attempt to read stuff. If not zero, we'll attempt to read
  // one extra byte.
  constexpr size_t initialAlloc = 1024 * 4;
  out.resize(std::min(
      buf.st_size > 0 ? (size_t(buf.st_size) + 1) : initialAlloc, num_bytes));

  while (soFar < out.size()) {
    const auto actual = readFull(fd, &out[soFar], out.size() - soFar);
    if (actual == -1) {
      return false;
    }
    soFar += actual;
    if (soFar < out.size()) {
      // File exhausted
      break;
    }
    // Ew, allocate more memory. Use exponential growth to avoid
    // quadratic behavior. Cap size to num_bytes.
    out.resize(std::min(out.size() * 3 / 2, num_bytes));
  }

  return true;
}

/**
 * Same as above, but takes in a file name instead of fd
 */
template <class Container>
bool readFile(
    const char* file_name,
    Container& out,
    size_t num_bytes = std::numeric_limits<size_t>::max()) {
  TORCH_CHECK(file_name);

  const auto fd = openNoInt(file_name, O_RDONLY | O_CLOEXEC);
  if (fd == -1) {
    return false;
  }

  auto guard = c10::make_scope_exit([&]() {
    // Ignore errors when closing the file
    closeNoInt(fd);
  });

  return readFile(fd, out, num_bytes);
}

} // namespace torch::nativert
