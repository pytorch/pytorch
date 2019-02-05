#ifndef CAFFE2_UTILS_THREAD_POOL_H_
#define CAFFE2_UTILS_THREAD_POOL_H_

#include "ATen/core/thread_pool.h"
#include "caffe2/core/numa.h"
#include "caffe2/utils/thread_name.h"

namespace caffe2 {

class CAFFE2_API TaskThreadPool : public c10::ThreadPool {
 public:
  explicit TaskThreadPool(
      std::size_t pool_size,
      int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id) {}

  // TODO move this to ATen/core/thread_pool.h
  void init_thread() override {
    setThreadName("CaffeTaskThread");
    NUMABind(numa_node_id_);
  }
};

} // namespace caffe2

#endif // CAFFE2_UTILS_THREAD_POOL_H_
