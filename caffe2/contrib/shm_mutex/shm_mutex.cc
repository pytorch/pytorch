#include "shm_mutex.h"

ShmProcessMutexCheck& ShmProcessMutexCheck::getInstance() {
  static ShmProcessMutexCheck singleton;
  return singleton;
}

bool ShmProcessMutexCheck::addLock(const std::string& name) {
  std::lock_guard<std::mutex> l(m_);
  auto p = shmLocks_.emplace(name);
  return p.second;
}

bool ShmProcessMutexCheck::removeLock(const std::string& name) {
  std::lock_guard<std::mutex> l(m_);
  return shmLocks_.erase(name) == 1;
}
