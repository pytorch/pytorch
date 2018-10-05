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

#include "caffe2/core/net_async_base.h"

C10_DEFINE_int(
    caffe2_threads_per_hip_gpu,
    1,
    "Number of CPU threads per AMD HIP GPU");

namespace caffe2 {

std::shared_ptr<TaskThreadPool>
GetAsyncNetHIPThreadPool(int hip_gpu_id, int pool_size, bool create_new) {
  // For GPU, use per device thread pools of predefined constant size
  if (pool_size != c10::FLAGS_caffe2_threads_per_hip_gpu) {
    LOG(INFO) << "Overriding AMD HIP GPU pool size: using "
              << c10::FLAGS_caffe2_threads_per_hip_gpu << " threads per GPU";
  }
  static std::unordered_map<int, std::weak_ptr<TaskThreadPool>> pools;
  static std::mutex pool_mutex;

  if (create_new) {
    LOG(INFO) << "Created new AMD HIP GPU pool, size: "
              << c10::FLAGS_caffe2_threads_per_hip_gpu
              << "; GPU id: " << hip_gpu_id;
    return std::make_shared<TaskThreadPool>(
        c10::FLAGS_caffe2_threads_per_hip_gpu);
  } else {
    std::lock_guard<std::mutex> lock(pool_mutex);

    std::shared_ptr<TaskThreadPool> shared_pool = nullptr;
    if (pools.count(hip_gpu_id)) {
      shared_pool = pools.at(hip_gpu_id).lock();
    }
    if (!shared_pool) {
      LOG(INFO) << "Created shared AMD HIP GPU pool, size: "
                << c10::FLAGS_caffe2_threads_per_hip_gpu
                << "; GPU id: " << hip_gpu_id;
      shared_pool = std::make_shared<TaskThreadPool>(
          c10::FLAGS_caffe2_threads_per_hip_gpu);
      pools[hip_gpu_id] = shared_pool;
    }
    return shared_pool;
  }
}

C10_REGISTER_CREATOR(ThreadPoolRegistry, HIP, GetAsyncNetHIPThreadPool);

} // namespace caffe2
