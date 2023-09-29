#include <atomic>

#include <deque>
#include <memory>
#include <mutex>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <torch/csrc/distributed/c10d/Hooks.hpp>
namespace c10d {

namespace {

std::mutex callbacks_lock;
std::vector<CollectiveEventCallback> callbacks_list;
} // namespace

TORCH_API void register_collective_callback(
    CollectiveEventCallback&& callback) {
  std::unique_lock<std::mutex> lock(callbacks_lock);
  callbacks_list.push_back(std::move(callback));
}

namespace details {

void enqueue_c10d_event(EventInfo&& evt) {
  std::unique_lock<std::mutex> lock(callbacks_lock);
  for (auto& cb : callbacks_list) {
    cb(evt);
  }
}

} // namespace details
} // namespace c10d
