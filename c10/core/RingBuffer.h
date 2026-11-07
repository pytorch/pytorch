#pragma once

#include <c10/macros/Macros.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <vector>

#if C10_ASAN_ENABLED
#include <sanitizer/lsan_interface.h>
#endif

namespace c10 {

template <class T>
class RingBuffer {
 public:
  RingBuffer() {
    // alloc_trace is a heap pointer we intentionally never free: it can hold
    // references to Python state that is already destroyed by the time exit
    // handlers run, so freeing it then is unsafe. The owning allocator impl is
    // itself heap-allocated and destroyed at exit, which makes this allocation
    // unreachable and thus flagged by LeakSanitizer; annotate it as a
    // deliberate leak so LSan ignores it (and everything it transitively holds,
    // e.g. recorded Python contexts).
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    alloc_trace = new std::deque<T>();
#if C10_ASAN_ENABLED
    __lsan_ignore_object(alloc_trace);
#endif
  }

  void setMaxEntries(size_t size) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_max_entries_ = std::max(static_cast<size_t>(1), size);
    while (alloc_trace->size() > alloc_trace_max_entries_) {
      alloc_trace->pop_front();
    }
  }

  void insertEntries(const T& entry) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->push_back(entry);
    } else {
      alloc_trace->pop_front();
      alloc_trace->push_back(entry);
    }
  }

  void getEntries(std::vector<T>& result) const {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    result.reserve(result.size() + alloc_trace->size());
    result.insert(result.end(), alloc_trace->begin(), alloc_trace->end());
  }

  void clear() {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace->clear();
    alloc_trace->shrink_to_fit();
  }

 private:
  size_t alloc_trace_max_entries_ = 1;

  mutable std::mutex alloc_trace_lock;
  std::deque<T>* alloc_trace;
};

} // namespace c10
