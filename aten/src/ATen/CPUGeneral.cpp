#include <ATen/CPUGeneral.h>
#include <atomic>
#include <memory>
#include <thread>

namespace at {
// Lock free atomic type
std::atomic<int> num_threads(-1);

void set_num_threads(int num_threads_) {
  if (num_threads_ >= 0)
    num_threads.store(num_threads_);
}

int get_num_threads() { return num_threads.load(); }
}
