/*
 * This implements a machine-wide mutex to be used
 * to synchronize CUDA calls (memory allocation and frees) and
 * NCCL calls. This prevents a potential deadlock that
 * can occur.
 *
 * The implementation has a few caveats:
 *   - it assumes that PID are not reused
 *   - there is a possible race between the creation (shm_open followed
 *     by ftruncate) and the spin on 'isInitialized' (if the memory region is
 *     not all zeroes).
 *
 * There are two implementations of the mutex and they vary mostly by how
 * they wait:
 *   - The ShmTicketMutex_t is a simple ticket based lock and processes will
 *     queue up and only attempt to grab the lock when it is their turn
 *   - The ShmTTSetMutex_t is a simple test-test-and-set mutex. It is possibly
 *     faster for low contention.
 *
 * Use both as you would use any std::mutex. Both mutexes support try_lock as
 * well.
 */
#pragma once

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <climits>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>

#include "caffe2/core/logging.h"

const int kTicketDelay = 1000;
const int kTimeout = 1000;

class ShmProcessMutexCheck {
 public:
  static ShmProcessMutexCheck& getInstance();
  ShmProcessMutexCheck(const ShmProcessMutexCheck&) = delete;
  ShmProcessMutexCheck& operator=(const ShmProcessMutexCheck&) = delete;

  bool addLock(const std::string& name);
  bool removeLock(const std::string& name);

 protected:
  ShmProcessMutexCheck() = default;
  std::mutex m_;
  std::unordered_set<std::string> shmLocks_;
};

template <class Derived>
struct shm_traits;

struct ShmBaseHeader {
  std::atomic<bool> isInitialized;
  std::atomic<int> countMapped;
  std::atomic<pid_t> owner;
};

template <class Impl>
class ShmProcessMutex {
 public:
  using header_t = typename shm_traits<Impl>::header_t;

  explicit ShmProcessMutex(const char* name)
      : name_(name), check_(ShmProcessMutexCheck::getInstance()) {
    CAFFE_ENFORCE(check_.addLock(name_), "Creating duplicate lock: ", name_);
    myPid_ = getpid();
    // Try to open and map the shared memory location
    int fd = -1;
    while (true) {
      fd = shm_open(name, O_RDWR, 0);
      if (fd == -1) {
        CAFFE_ENFORCE(
            errno == ENOENT,
            "shm_open failed with not ENOENT: ",
            strerror(errno));

        // Create new object
        fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0700);
        if (fd == -1 && errno == EEXIST) {
          // Some other process created first; loop around to re-open
          continue;
        }
        CAFFE_ENFORCE(
            fd != -1, "shm_open failed with create: ", strerror(errno));
        // At this point, we are the creator of the shared object.
        // Initialize the header_ (it's all 0 right now)
        auto rv = ftruncate(fd, sizeof(header_t));
        CAFFE_ENFORCE(rv != -1, "ftruncate: ", strerror(errno));

        // Map memory and initialize
        header_ = (header_t*)mmap(
            nullptr,
            sizeof(header_t),
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            0);
        CAFFE_ENFORCE(header_ != MAP_FAILED, "mmap: ", strerror(errno));

        header_->countMapped = 1;
        header_->owner = 0;
        header_->isInitialized.store(true, std::memory_order_release);
        close(fd);
        break;
      } else {
        // Object exists, we just map it
        header_ = (header_t*)mmap(
            nullptr,
            sizeof(header_t),
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            0);
        CAFFE_ENFORCE(header_ != MAP_FAILED, "mmap: ", strerror(errno));

        // Wait for memory to be initialized
        while (header_->isInitialized.load(std::memory_order_acquire) == 0) {
          // Spin; should be done soon
        }
        // Now check if we can register ourself by incrementing countMapped.
        // If we are "locked-out" (shared object being destroyed), retry
        if (header_->countMapped.fetch_add(1, std::memory_order_relaxed) < 0) {
          header_->countMapped.fetch_sub(1, std::memory_order_relaxed);
          int rv = munmap(header_, sizeof(header_t));
          CAFFE_ENFORCE(rv == 0, "munmap (to retry) failed: ", strerror(errno));
          close(fd);
          continue;
        }
        close(fd);
        break;
      }
    }
  }

  ~ShmProcessMutex() {
    if (header_ != nullptr) {
      // We are participating in a lock. Destroy
      internalDestroy();
    }
  }

  // Copy and assignment operator are implicitly deleted

  ShmProcessMutex(ShmProcessMutex&& toMove) noexcept
      : header_(toMove.header_),
        myPid_(toMove.myPid_),
        name_(toMove.name_),
        check_(toMove.check_) {
    toMove.header_ = nullptr;
    toMove.myPid_ = -1;
  }

  ShmProcessMutex& operator=(ShmProcessMutex&& toMove) {
    CAFFE_ENFORCE(toMove.myPid_ == this->myPid_);
    if (&toMove != this) {
      internalDestroy();
      header_ = toMove.header_;
      name_ = toMove.name_;
      toMove.header_ = nullptr;
      toMove.myPid_ = -1;
    }
    return *this;
  }

  void lock() {
    pid_t expectedPid = 0;
    while (not header_->owner.compare_exchange_weak(
        expectedPid,
        myPid_,
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
      if (expectedPid == 0) {
        continue;
      }
      // Someone else has the lock. We check if that process is
      // still alive
      if (kill(expectedPid, 0) < 0 && errno == ESRCH) {
        // The process no longer exists. Try to "steal" the lock
        continue;
      }
      while (true) {
        if (static_cast<Impl*>(this)->waitForLock()) {
          return;
        }
        expectedPid = header_->owner.load(std::memory_order_relaxed);
        if (expectedPid == 0 || (kill(expectedPid, 0) < 0 && errno == ESRCH)) {
          break;
        }
      }
    }
  }

  bool try_lock() {
    pid_t expectedPid = 0;
    bool firstTry = true;
    while (not header_->owner.compare_exchange_weak(
        expectedPid,
        myPid_,
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
      if (expectedPid == 0) {
        continue;
      }
      // Someone else has the lock. We check if that process is
      // still alive
      if (firstTry && kill(expectedPid, 0) < 0 && errno == ESRCH) {
        firstTry = false;
        // The process no longer exists. Try to "steal" the lock once
        continue;
      }
      return false;
    }
    return true;
  }

  void unlock() noexcept {
    header_->owner.store(0, std::memory_order_relaxed);
    static_cast<Impl*>(this)->subUnlock();
  }

 protected:
  header_t* header_;
  pid_t myPid_;
  std::string name_;

  ShmProcessMutexCheck& check_;

 private:
  void internalDestroy() {
    CAFFE_ENFORCE(header_ != nullptr, "Internal error");
    CAFFE_ENFORCE(check_.removeLock(name_), "Double free of lock: ", name_);
    // Unmap the memory. If we are the last one, "lock" the
    // shared memory and free it if successful
    int oldCount = header_->countMapped.fetch_sub(1, std::memory_order_relaxed);
    bool doUnlink = false;
    if (oldCount == 1) {
      // We were the last one. We attempt to lock out
      // future processes by exchanging with something very negative
      // This simplifies the checks when checking for lock out
      oldCount = 0;
      if (header_->countMapped.compare_exchange_strong(
              oldCount,
              INT_MIN,
              std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        doUnlink = true;
      }
    }
    int rv = munmap(header_, sizeof(header_t));
    CAFFE_ENFORCE(rv == 0, "munmap failed: ", strerror(errno));
    if (doUnlink) {
      rv = shm_unlink(name_.c_str());
      CAFFE_ENFORCE(rv == 0, "shm_unlink failed: ", strerror(errno));
    }
  }
};

template <class T>
class ShmTTSetMutex : public ShmProcessMutex<ShmTTSetMutex<T>> {
 public:
  friend class ShmProcessMutex<ShmTTSetMutex<T>>;
  explicit ShmTTSetMutex(const char* name, int timeout = kTimeout)
      : ShmProcessMutex<ShmTTSetMutex>(name), timeout_(timeout) {}

 protected:
  bool waitForLock() {
    int delay = timeout_;
    pid_t expectedPid = 0;
    while (--delay > 0 &&
           this->header_->owner.load(std::memory_order_relaxed)) {
      // Empty loop
      __asm__ __volatile__("");
    }
    return this->header_->owner.compare_exchange_strong(
        expectedPid, this->myPid_, std::memory_order_relaxed);
  }

  void subUnlock() noexcept {}
  int timeout_;
};

template <class T>
class ShmTicketMutex : public ShmProcessMutex<ShmTicketMutex<T>> {
 public:
  friend class ShmProcessMutex<ShmTicketMutex<T>>;
  explicit ShmTicketMutex(const char* name, int delay = kTicketDelay)
      : ShmProcessMutex<ShmTicketMutex>(name), delay_(delay) {}

 protected:
  bool waitForLock() {
    pid_t expectedPid = 0;
    int slot = this->header_->ticket.fetch_add(1, std::memory_order_relaxed);
    for (;;) {
      int spintime =
          (slot - this->header_->now.load(std::memory_order_relaxed)) * delay_;
      for (int i = 0; i < spintime; i++) {
        // Empty loop
        __asm__ __volatile__("");
      }
      if (this->header_->now.load(std::memory_order_relaxed) == slot) {
        break;
      }
    }
    return this->header_->owner.compare_exchange_strong(
        expectedPid, this->myPid_, std::memory_order_relaxed);
  }

  void subUnlock() noexcept {
    this->header_->now.fetch_add(1, std::memory_order_relaxed);
  }

  int delay_;
};

template <class T>
struct shm_traits<ShmTTSetMutex<T>> {
  using header_t = T;
};

template <class T>
struct shm_traits<ShmTicketMutex<T>> {
  using header_t = T;
};

struct TicketStruct : ShmBaseHeader {
  std::atomic<unsigned> ticket;
  std::atomic<unsigned> now;
};

template class ShmTicketMutex<TicketStruct>;
template class ShmTTSetMutex<ShmBaseHeader>;

using ShmTicketMutex_t = ShmTicketMutex<TicketStruct>;
using ShmTTSetMutex_t = ShmTTSetMutex<ShmBaseHeader>;
