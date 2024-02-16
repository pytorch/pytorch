
#pragma once

#include <shared_mutex>

#include <torch/csrc/autograd/function.h>

namespace c10d {

// `RankLocal` maintains a unique instance of T for each non-autograd thread.
// For non-autograd threads, `RankLocal<T>::get()` functions similar to
// thread_local. For autograd threads, `RankLocal<T>::get()` returns the
// instance of T corresponding to the enqueuing non-autograd thread. The
// mechanism allows for rank-specific context shared between forward and
// backward. It works for both the one-rank-per-process and one-rank-per-thread
// scenarios.
//
// NOTE: RankLocal doesn't make the underlying objects thread-safe.
template <typename T>
class RankLocal {
 public:
  RankLocal(const RankLocal&) = delete;
  RankLocal& operator=(const RankLocal&) = delete;

  static T& get() {
    // Fast path: non-autograd threads can simply return
    // the object reference cached in TLS.
    if (cached_ != nullptr) {
      return *cached_;
    }
    const auto node = torch::autograd::get_current_node();
    auto fwd_thread_id = node == nullptr ? at::RecordFunction::currentThreadId()
                                         : node->thread_id();
    // Optimistically aquire the read lock first, since most likely we are in
    // an autograd thread and the object has already been constructed.
    {
      std::shared_lock read_lock(lock_);
      auto it = thread_id_to_rank_local_.find(fwd_thread_id);
      if (it != thread_id_to_rank_local_.end()) {
        // Cache for non-autograd threads
        if (node == nullptr) {
          cached_ = &it->second;
        }
        return it->second;
      }
    }

    std::unique_lock write_lock(lock_);
    auto [it, _] = thread_id_to_rank_local_.try_emplace(fwd_thread_id);
    // Cache for non-autograd threads
    if (node == nullptr) {
      cached_ = &it->second;
    }
    return it->second;
  }

 private:
  RankLocal(){};
  thread_local static T* cached_;
  static std::unordered_map<uint64_t, T> thread_id_to_rank_local_;
  static std::shared_mutex lock_;
};

template <typename T>
thread_local T* RankLocal<T>::cached_ = nullptr;

template <typename T>
std::unordered_map<uint64_t, T> RankLocal<T>::thread_id_to_rank_local_;

template <typename T>
std::shared_mutex RankLocal<T>::lock_;

} // namespace c10d
