#include <c10d/FileStore.hpp>

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <system_error>
#include <thread>

#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }

namespace c10d {

namespace {

template <typename F>
typename std::result_of<F()>::type syscall(F fn) {
  while (true) {
    auto rv = fn();
    if (rv == -1) {
      if (errno == EINTR) {
        continue;
      }
    }
    return rv;
  }
}

// For a comprehensive overview of file locking methods,
// see: https://gavv.github.io/blog/file-locks/.
// We stick to flock(2) here because we don't care about
// locking byte ranges and don't want locks to be process-wide.

// RAII wrapper around flock(2)
class Lock {
 public:
  explicit Lock(int fd, int operation) : fd_(fd) {
    flock(operation);
  }

  ~Lock() {
    unlock();
  }

  Lock(const Lock& that) = delete;

  Lock(Lock&& other) noexcept {
    fd_ = other.fd_;
    other.fd_ = -1;
  }

  void unlock() {
    if (fd_ >= 0) {
      flock(LOCK_UN);
      fd_ = -1;
    }
  }

 protected:
  int fd_;

  void flock(int operation) {
    auto rv = syscall(std::bind(::flock, fd_, operation));
    SYSASSERT(rv, "flock");
  }
};

class File {
 public:
  explicit File(
      const std::string& path,
      int flags,
      std::chrono::milliseconds timeout) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      fd_ = syscall(std::bind(::open, path.c_str(), flags, 0644));
      // Only retry when the file doesn't exist, since we are waiting for the
      // file to be created in this case to address the following issue:
      // https://github.com/pytorch/pytorch/issues/13750
      if (fd_ >= 0 || errno != ENOENT) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (timeout != c10d::Store::kNoTimeout && elapsed > timeout) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    SYSASSERT(fd_, "open(" + path + ")");
  }

  ~File() {
    ::close(fd_);
  }

  Lock lockShared() {
    return Lock(fd_, LOCK_SH);
  }

  Lock lockExclusive() {
    return Lock(fd_, LOCK_EX);
  }

  off_t seek(off_t offset, int whence) {
    auto rv = syscall(std::bind(lseek, fd_, offset, whence));
    SYSASSERT(rv, "lseek");
    return rv;
  }

  off_t tell() {
    auto rv = syscall(std::bind(lseek, fd_, 0, SEEK_CUR));
    SYSASSERT(rv, "lseek");
    return rv;
  }

  off_t size() {
    auto pos = tell();
    auto size = seek(0, SEEK_END);
    seek(pos, SEEK_SET);
    return size;
  }

  void write(const void* buf, size_t count) {
    while (count > 0) {
      auto rv = syscall(std::bind(::write, fd_, buf, count));
      SYSASSERT(rv, "write");
      buf = (uint8_t*)buf + count;
      count -= rv;
    }
  }

  void read(void* buf, size_t count) {
    while (count > 0) {
      auto rv = syscall(std::bind(::read, fd_, buf, count));
      SYSASSERT(rv, "read");
      buf = (uint8_t*)buf + count;
      count -= rv;
    }
  }

  void write(const std::string& str) {
    uint32_t len = str.size();
    assert(str.size() <= std::numeric_limits<decltype(len)>::max());
    write(&len, sizeof(len));
    write(str.c_str(), len);
  }

  void write(const std::vector<uint8_t>& data) {
    uint32_t len = data.size();
    assert(data.size() <= std::numeric_limits<decltype(len)>::max());
    write(&len, sizeof(len));
    write(data.data(), len);
  }

  void read(std::string& str) {
    uint32_t len;
    read(&len, sizeof(len));
    std::vector<uint8_t> buf(len);
    read(buf.data(), len);
    str.assign(buf.begin(), buf.end());
  }

  void read(std::vector<uint8_t>& data) {
    uint32_t len;
    read(&len, sizeof(len));
    data.resize(len);
    read(data.data(), len);
  }

 protected:
  int fd_;
};

off_t refresh(
    File& file,
    off_t pos,
    std::unordered_map<std::string, std::vector<uint8_t>>& cache) {
  auto size = file.size();
  if (size != pos) {
    std::string tmpKey;
    std::vector<uint8_t> tmpValue;
    file.seek(pos, SEEK_SET);
    while (size > pos) {
      file.read(tmpKey);
      file.read(tmpValue);
      cache[tmpKey] = std::move(tmpValue);
      pos = file.tell();
    }
  }
  file.seek(0, SEEK_SET);
  return pos;
}

} // namespace

FileStore::FileStore(const std::string& path, int numWorkers)
    : Store(),
      path_(path),
      pos_(0),
      numWorkers_(numWorkers),
      cleanupKey_("cleanup/"),
      regularPrefix_("/") {
  if (numWorkers_ < 1) {
    throw std::runtime_error(
        "Number of workers for FileStore should be greater than zero");
  }
}

FileStore::~FileStore() {
  // cleanup key will be different from all rest keys since all rest keys will
  // have a regular prefix.
  auto numFinishedWorker = addHelper(cleanupKey_, 1);
  // The last worker cleans up the file
  if (numFinishedWorker == numWorkers_) {
    // Best effort removal without checking the return
    std::remove(path_.c_str());
  }
}

void FileStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  std::string regKey = regularPrefix_ + key;
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDWR | O_CREAT, timeout_);
  auto lock = file.lockExclusive();
  file.seek(0, SEEK_END);
  file.write(regKey);
  file.write(value);
}

std::vector<uint8_t> FileStore::get(const std::string& key) {
  std::string regKey = regularPrefix_ + key;
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    std::unique_lock<std::mutex> l(activeFileOpLock_);
    File file(path_, O_RDONLY, timeout_);
    auto lock = file.lockShared();
    auto size = file.size();
    if (cache_.count(regKey) == 0 && size == pos_) {
      // No new entries; release the shared lock and sleep for a bit
      lock.unlock();
      l.unlock();
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (timeout_ != kNoTimeout && elapsed > timeout_) {
        throw std::runtime_error("Timeout waiting for key: " + key);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    // Always refresh since even though the key exists in the cache,
    // it might be outdated
    pos_ = refresh(file, pos_, cache_);
    if (cache_.count(regKey) != 0) {
      return cache_[regKey];
    }
  }
}

int64_t FileStore::addHelper(const std::string& key, int64_t i) {
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDWR | O_CREAT, timeout_);
  auto lock = file.lockExclusive();
  pos_ = refresh(file, pos_, cache_);

  const auto& value = cache_[key];
  int64_t ti = i;
  if (!value.empty()) {
    auto buf = reinterpret_cast<const char*>(value.data());
    auto len = value.size();
    ti += std::stoll(std::string(buf, len));
  }
  // Always seek to the end to write
  file.seek(0, SEEK_END);
  // File cursor is at the end of the file now, and we have an
  // exclusive lock, so we can write the new value.
  file.write(key);
  file.write(std::to_string(ti));
  return ti;
}

int64_t FileStore::add(const std::string& key, int64_t i) {
  std::string regKey = regularPrefix_ + key;
  return addHelper(regKey, i);
}

bool FileStore::check(const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDONLY, timeout_);
  auto lock = file.lockShared();
  pos_ = refresh(file, pos_, cache_);

  for (const auto& key : keys) {
    std::string regKey = regularPrefix_ + key;
    if (cache_.count(regKey) == 0) {
      return false;
    }
  }

  return true;
}

void FileStore::wait(const std::vector<std::string>& keys) {
  wait(keys, timeout_);
}

void FileStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  // Not using inotify because it doesn't work on many
  // shared filesystems (such as NFS).
  const auto start = std::chrono::steady_clock::now();
  while (!check(keys)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      throw std::runtime_error("Wait timeout");
    }

    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace c10d
