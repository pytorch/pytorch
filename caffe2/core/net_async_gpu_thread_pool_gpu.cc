#include "caffe2/core/net_async_gpu_thread_pool.h"

#include "caffe2/core/context_gpu.h"

C10_DEFINE_int(caffe2_threads_per_gpu, 1, "Number of CPU threads per GPU");

namespace caffe2 {

C10_REGISTER_CREATOR(ThreadPoolRegistry, CUDA, GetAsyncNetGPUThreadPool);

std::shared_ptr<TaskThreadPool>
GetAsyncNetGPUThreadPool(int gpu_id, int pool_size, bool create_new) {
  // For GPU, use per device thread pools of predefined constant size
  if (pool_size != c10::FLAGS_caffe2_threads_per_gpu) {
    LOG(INFO) << "Overriding GPU pool size: using "
              << c10::FLAGS_caffe2_threads_per_gpu << " threads per GPU";
  }
  static std::unordered_map<int, std::weak_ptr<TaskThreadPool>> pools;
  static std::mutex pool_mutex;

  if (create_new) {
    LOG(INFO) << "Created new GPU pool, size: "
              << c10::FLAGS_caffe2_threads_per_gpu << "; GPU id: " << gpu_id;
    return std::make_shared<TaskThreadPool>(c10::FLAGS_caffe2_threads_per_gpu);
  } else {
    std::lock_guard<std::mutex> lock(pool_mutex);

    std::shared_ptr<TaskThreadPool> shared_pool = nullptr;
    if (pools.count(gpu_id)) {
      shared_pool = pools.at(gpu_id).lock();
    }
    if (!shared_pool) {
      LOG(INFO) << "Created shared GPU pool, size: "
                << c10::FLAGS_caffe2_threads_per_gpu << "; GPU id: " << gpu_id;
      shared_pool =
          std::make_shared<TaskThreadPool>(c10::FLAGS_caffe2_threads_per_gpu);
      pools[gpu_id] = shared_pool;
    }
    return shared_pool;
  }
}

} // namespace caffe2
