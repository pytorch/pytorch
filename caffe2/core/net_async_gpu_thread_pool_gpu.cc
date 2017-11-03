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

#include "caffe2/core/net_async_gpu_thread_pool.h"

#include "caffe2/core/context_gpu.h"

CAFFE2_DEFINE_int(caffe2_threads_per_gpu, 1, "Number of CPU threads per GPU");

namespace caffe2 {

namespace {
std::shared_ptr<TaskThreadPool> AsyncNetGPUThreadPoolCreator(
    const DeviceOption& device_option) {
  CAFFE_ENFORCE_EQ(
      device_option.device_type(),
      CUDA,
      "Unexpected device type for CUDA thread pool");
  return GetAsyncNetGPUThreadPool(device_option.cuda_gpu_id());
}
} // namespace

CAFFE_REGISTER_CREATOR(ThreadPoolRegistry, CUDA, AsyncNetGPUThreadPoolCreator);

std::shared_ptr<TaskThreadPool> GetAsyncNetGPUThreadPool(int gpu_id) {
  static std::unordered_map<int, std::weak_ptr<TaskThreadPool>> pools;
  static std::mutex pool_mutex;
  std::lock_guard<std::mutex> lock(pool_mutex);

  std::shared_ptr<TaskThreadPool> shared_pool = nullptr;
  if (pools.count(gpu_id)) {
    shared_pool = pools.at(gpu_id).lock();
  }
  if (!shared_pool) {
    shared_pool =
        std::make_shared<TaskThreadPool>(FLAGS_caffe2_threads_per_gpu);
    pools[gpu_id] = shared_pool;
  }
  return shared_pool;
}

} // namespace caffe2
