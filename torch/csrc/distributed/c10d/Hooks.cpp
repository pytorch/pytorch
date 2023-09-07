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

std::atomic<bool> event_queue_enabled = false;
int sync_pipe;
std::mutex event_queue_lock;
std::deque<details::EventInfo> event_queue;

} // namespace

void enable_event_collection(int pipe) {
  sync_pipe = pipe;
  event_queue_enabled.store(true);
}

namespace details {

bool dequeue_c10d_event(EventInfo& evt) {
  std::unique_lock<std::mutex> lock(event_queue_lock);
  if (event_queue.size() == 0) {
    return false;
  }
  evt = event_queue.front();
  event_queue.pop_front();
  return true;
}

void enqueue_c10d_event(EventInfo&& evt) {
  if (!event_queue_enabled.load())
    return;

  std::unique_lock<std::mutex> lock(event_queue_lock);
  event_queue.push_back(std::move(evt));
  char m = 'x';
  write(sync_pipe, &m, 1);
}

} // namespace details
} // namespace c10d
