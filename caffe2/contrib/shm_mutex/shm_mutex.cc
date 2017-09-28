/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
