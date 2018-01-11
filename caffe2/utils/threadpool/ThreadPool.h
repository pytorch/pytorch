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

#ifndef CAFFE2_UTILS_THREADPOOL_H_
#define CAFFE2_UTILS_THREADPOOL_H_

#include "ThreadPoolCommon.h"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

//
// A work-stealing threadpool loosely based off of pthreadpool
//

namespace caffe2 {

class Task;
class WorkersPool;

constexpr size_t kCacheLineSize = 64;

class alignas(kCacheLineSize) ThreadPool {
 public:
  // Constructs a work-stealing threadpool with the given number of
  // threads
  static std::unique_ptr<ThreadPool> defaultThreadPool();
  ThreadPool(int numThreads);
  ~ThreadPool();
  // Returns the number of threads currently in use
  int getNumThreads() const;

  // Sets the minimum work size (range) for which to invoke the
  // threadpool; work sizes smaller than this will just be run on the
  // main (calling) thread
  void setMinWorkSize(size_t size);
  size_t getMinWorkSize() const { return minWorkSize_; }
  void run(const std::function<void(int, size_t)>& fn, size_t range);

private:
  mutable std::mutex executionMutex_;
  size_t minWorkSize_;
  size_t numThreads_;
  std::shared_ptr<WorkersPool> workersPool_;
  std::vector<std::shared_ptr<Task>> tasks_;
};

} // namespace caffe2

#endif // CAFFE2_UTILS_THREADPOOL_H_
