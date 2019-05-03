#pragma once
#include <ATen/Parallel.h>
#include <c10/core/thread_pool.h>

namespace at {

// Checks external number of threads setting and returns
// default value (hardware_concurrency()) if it's negative
inline size_t check_and_get_pool_size(int nthreads) {
  if (nthreads > 0) {
    return nthreads;
  } else {
    return std::thread::hardware_concurrency();
  }
}

class CAFFE2_API PTThreadPool : public c10::ThreadPool {
 public:
  explicit PTThreadPool(
      std::size_t pool_size,
      int numa_node_id = -1)
    : c10::ThreadPool(pool_size, numa_node_id) {}

  void init_thread() override {
    c10::setThreadName("PTThreadPool");
    at::init_num_threads();
  }
};

} // namespace at
