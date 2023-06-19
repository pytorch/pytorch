#include <torch/csrc/distributed/c10d/FileStore.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <cassert>
#include <cstdint>

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#include <fileapi.h>
#include <io.h>
#include <filesystem>
#else
#include <sys/file.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <system_error>
#include <thread>
#include <utility>

#include <c10/util/Exception.h>

#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }

#ifdef _WIN32
#define LOCK_EX 0x00000001
#define LOCK_SH 0x00000010
#define LOCK_UN 0x00000100

int flock_(int fd, int op) {
  HANDLE hdl = (HANDLE)_get_osfhandle(fd);
  DWORD low = 1, high = 0;
  OVERLAPPED offset = {0, 0, 0, 0, NULL};

  if ((intptr_t)hdl < 0)
    return -1;

  switch (op) {
    case LOCK_EX:
      if (LockFileEx(hdl, LOCKFILE_EXCLUSIVE_LOCK, 0, low, high, &offset))
        return 0;
      break;
    case LOCK_SH:
      if (LockFileEx(hdl, 0, 0, low, high, &offset))
        return 0;
      break;
    case LOCK_UN:
      if (UnlockFileEx(hdl, 0, low, high, &offset) != 0)
        return 0;
      break;
    default:
      break;
  }
  errno = EINVAL;
  return -1;
}
#endif

namespace c10d {

namespace {

template <typename F>
typename c10::invoke_result_t<F> syscall(F fn) {
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

  Lock& operator=(Lock&& other) noexcept {
    if (this != &other) {
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  Lock(Lock&& other) noexcept {
    *this = std::move(other);
  }

  void unlock() {
    if (fd_ >= 0) {
      flock(LOCK_UN);
      fd_ = -1;
    }
  }

 protected:
  int fd_{-1};

  void flock(int operation) {
#ifdef _WIN32
    auto rv = syscall(std::bind(::flock_, fd_, operation));
#else
    auto rv = syscall([this, operation] { return ::flock(fd_, operation); });
#endif
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
#ifdef _WIN32
      fd_ = syscall(std::bind(
          ::open, path.c_str(), flags | _O_BINARY, _S_IREAD | _S_IWRITE));
#else
      fd_ = syscall([capture0 = path.c_str(), flags] {
        return ::open(capture0, flags, 0644);
      });
#endif
      // Only retry when the file doesn't exist, since we are waiting for the
      // file to be created in this case to address the following issue:
      // https://github.com/pytorch/pytorch/issues/13750
      if (fd_ >= 0 || errno != ENOENT) {
        break;
      }
#ifdef _WIN32
      // if the parent folder doesn't exist it will never be able to create the
      // file so we can skip the retry
      if (!std::filesystem::exists(std::filesystem::path(path).parent_path())) {
        break;
      }
#endif
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
    auto rv =
        syscall([this, offset, whence] { return lseek(fd_, offset, whence); });
    SYSASSERT(rv, "lseek");
    return rv;
  }

  off_t tell() {
    auto rv = syscall([this] { return lseek(fd_, 0, SEEK_CUR); });
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
      auto rv =
          syscall([this, buf, count] { return ::write(fd_, buf, count); });
      SYSASSERT(rv, "write");
      buf = (uint8_t*)buf + rv;
      count -= rv;
    }
  }

  void read(void* buf, size_t count) {
    while (count > 0) {
      auto rv = syscall([this, buf, count] { return ::read(fd_, buf, count); });
      SYSASSERT(rv, "read");
      buf = (uint8_t*)buf + rv;
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
    uint32_t len = 0;
    read(&len, sizeof(len));
    std::vector<uint8_t> buf(len);
    read(buf.data(), len);
    str.assign(buf.begin(), buf.end());
  }

  void read(std::vector<uint8_t>& data) {
    uint32_t len = 0;
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
    std::unordered_map<std::string, std::vector<uint8_t>>& cache,
    const std::string deletePrefix) {
  auto size = file.size();
  if (size != pos) {
    std::string tmpKey;
    std::vector<uint8_t> tmpValue;
    file.seek(pos, SEEK_SET);
    while (size > pos) {
      file.read(tmpKey);
      file.read(tmpValue);
      if (tmpKey.compare(0, deletePrefix.size(), deletePrefix) == 0) {
        cache.erase(tmpKey.substr(deletePrefix.size()));
      } else {
        cache[tmpKey] = std::move(tmpValue);
      }
      pos = file.tell();
    }
  }
  file.seek(0, SEEK_SET);
  return pos;
}

} // namespace

FileStore::FileStore(std::string path, int numWorkers)
    : Store(),
      path_(std::move(path)),

      numWorkers_(numWorkers),
      cleanupKey_("cleanup/"),
      refCountKey_("refcount/"),
      regularPrefix_("/"),
      deletePrefix_("-") {
  addHelper(refCountKey_, 1);
}

FileStore::~FileStore() {
  // If the file does not exist - exit.
  // This can happen when FileStore is invoked from python language which has
  // GC. If python code has directory cleanup procedure, the race condition may
  // occur between that code and this deconstructor. As a result, we check for
  // file existense before cleanup
#ifdef _WIN32
  int res = syscall(std::bind(::_access, path_.c_str(), 0));
#else
  int res =
      syscall([filepath = path_.c_str()] { return ::access(filepath, F_OK); });
#endif
  if (res == -1) {
    return;
  }

  // cleanup key will be different from all rest keys since all rest keys will
  // have a regular prefix.
  auto numFinishedWorker = addHelper(cleanupKey_, 1);
  auto refCount = addHelper(refCountKey_, -1);
  // The last worker cleans up the file. If numWorkers was not initialized to
  // a specific postive value (i.e. meaning that there was not a fixed number
  // of workers), we don't attempt to clean.
  // Clean up the file if number of references is 0.
  if (refCount == 0 && numWorkers_ >= 0 && numFinishedWorker >= numWorkers_) {
    // Best effort removal without checking the return
    ::remove(path_.c_str());
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

std::vector<uint8_t> FileStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  std::string regKey = regularPrefix_ + key;
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDWR | O_CREAT, timeout_);
  auto lock = file.lockExclusive();
  // Always refresh since even though the key exists in the cache,
  // it might be outdated
  pos_ = refresh(file, pos_, cache_, deletePrefix_);
  if ((cache_.count(regKey) == 0 && expectedValue.empty()) ||
      (cache_.count(regKey) != 0 && cache_[regKey] == expectedValue)) {
    // if the key does not exist and currentValue arg is empty or
    // the key does exist and current value is what is expected, then set it
    file.seek(0, SEEK_END);
    file.write(regKey);
    file.write(desiredValue);
    return desiredValue;
  } else if (cache_.count(regKey) == 0) {
    // if the key does not exist
    return expectedValue;
  }
  // key exists but current value is not expected
  return cache_[regKey];
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
        auto err = c10::str(
            "Timeout waiting for key: ",
            key,
            " after ",
            timeout_.count(),
            " ms");
        TORCH_CHECK(false, err);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    // Always refresh since even though the key exists in the cache,
    // it might be outdated
    pos_ = refresh(file, pos_, cache_, deletePrefix_);
    if (cache_.count(regKey) != 0) {
      return cache_[regKey];
    }
  }
}

int64_t FileStore::addHelper(const std::string& key, int64_t i) {
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDWR | O_CREAT, timeout_);
  auto lock = file.lockExclusive();
  pos_ = refresh(file, pos_, cache_, deletePrefix_);

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

int64_t FileStore::add(const std::string& key, int64_t value) {
  std::string regKey = regularPrefix_ + key;
  return addHelper(regKey, value);
}

int64_t FileStore::getNumKeys() {
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDONLY, timeout_);
  auto lock = file.lockShared();
  pos_ = refresh(file, pos_, cache_, deletePrefix_);
  return cache_.size();
}

bool FileStore::deleteKey(const std::string& key) {
  std::string deleteKey = deletePrefix_ + regularPrefix_ + key;
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDWR, timeout_);
  auto lock = file.lockExclusive();
  file.seek(0, SEEK_END);
  file.write(deleteKey);
  file.write(std::vector<uint8_t>{});
  return true;
}

bool FileStore::check(const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> l(activeFileOpLock_);
  File file(path_, O_RDONLY, timeout_);
  auto lock = file.lockShared();
  pos_ = refresh(file, pos_, cache_, deletePrefix_);

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
      TORCH_CHECK(false, "Wait timeout");
    }

    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace c10d
