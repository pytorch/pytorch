#include <torch/csrc/profiler/collection.h>

#include <algorithm>

#include <ATen/record_function.h>

namespace torch {
namespace profiler {
namespace impl {
namespace {
// See `RecordQueue::getSubqueue()` for an overview of this cache.
struct SubQueueThreadCache {
  uint32_t key_;
  ThreadLocalSubqueue* ref_;
};

// The astute observer will note that this leaves a dangling reference; nothing
// in the teardown of `RecordQueue` or `ThreadLocalSubqueue` clears this value.
// (And the raw pointer in `SubQueueThreadCache` will not extend the lifetime
// of `*ref_`.) This is safe, however, because `getSubqueue` will check
// `sub_queue_cache_.key_` before attempting to access `ref_`, and if `key_`
// does not match the RecordQueue's *unique* `id_` it will evict
// `sub_queue_cache_` and fall back to a different mechanism.
std::atomic<uint32_t> queue_id_{0};
thread_local SubQueueThreadCache sub_queue_cache_{0, nullptr};
} // namespace

RecordQueue::RecordQueue() : id_(++queue_id_) {}

ThreadLocalSubqueue* RecordQueue::getSubqueue() {
  // In the most common case, a thread will want to write to the same sub-queue
  // that it wrote to last call. The only time that isn't true is if:
  //  A) The profiler context has ended and we are in a new one.
  //  B) Two profilers are active in different TLS contexts, and this thread
  //     is a worker helping with intra-op parallelism.
  // Since we expect this to be the OVERWHELMINGLY common case (>99%), we add a
  // special thread_local cache so that we can skip the overall `flat_hash_map`
  // (and corresponding lock).
  if (id_ == sub_queue_cache_.key_) {
    return sub_queue_cache_.ref_;
  }

  const auto tid = at::RecordFunction::currentThreadId();
  std::lock_guard<std::mutex> guard(sub_queue_mutex_);
  auto it = sub_queues_.find(tid);
  if (it == sub_queues_.end()) {
    it =
        sub_queues_.emplace(tid, std::make_unique<ThreadLocalSubqueue>()).first;
  }

  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

std::deque<OpEventData> RecordQueue::getRecords(
    std::function<time_t(approx_time_t)> time_converter) {
  auto converter = [&](approx_time_t t) {
    return t == std::numeric_limits<approx_time_t>::min()
        ? std::numeric_limits<int64_t>::min()
        : time_converter(t) / 1000; // ns to ms
  };
  std::deque<OpEventData> out;
  for (auto& subqueue_it : sub_queues_) {
    for (auto& i : subqueue_it.second->data_) {
      if (!i.backend_.has_value()) {
        i.start_time_.us_ = converter(i.start_time_.count_);
        i.end_time_.us_ = converter(i.end_time_.count_);
      }
      out.emplace_back(std::move(i));
    }
    subqueue_it.second->data_.clear();
  }

  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.start_time_.us_ < b.start_time_.us_;
  });
  return out;
}

} // namespace impl
} // namespace profiler
} // namespace torch
