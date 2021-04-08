#pragma once

#include <ATen/Parallel.h>
#include <c10/core/thread_pool.h>

namespace at {

class TORCH_API PTThreadPool : public c10::ThreadPool {
public:
  explicit PTThreadPool(
      int pool_size,
      int numa_node_id = -1)
    : c10::ThreadPool(pool_size, numa_node_id, [](){
        c10::setThreadName("PTThreadPool");
        at::init_num_threads();
      }) {}
};

} // namespace at
