// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <thread>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <c10/util/Semaphore.h>
#include <c10/util/irange.h>

namespace c10 {

/*
  the general structure:
  1. we have an aligned buffer of size (N * sizeof(T))
  2. we have a head pointer which points to the next 'acquirable' slot
  3. each value has a next pointer which points the slot available
     after the current head is taken

  it's essentially a managed linked list that behaves like a stack
  where each stack element can only be acquired once, and must be
  released before it can be acquired again; the user must ensure that
  the usage of the acquired item is thread-safe, since this structure
  makes no guarantees here. there are no allocations after initialization.

  upon acquiring, we
  1. set the value of the head
     to its 'next' value

  upon releasing, we
  1. set the next value of the released
     value to the current head
  2. set the head to the releaseed value
*/

template <typename T, bool use_cv_on_empty = false>
class BufferedAtomicStack {
 private:
  C10_ALWAYS_INLINE size_t aligned_buffer_bytes(size_t N) {
    return ((sizeof(T) * N) + std::alignment_of_v<T> - 1) &
        ~(std::alignment_of_v<T> - 1);
  }

  using deleter_fn_ptr = void (*)(T*);
  static void default_deleter(T*) {}

 public:
  template <typename... Args>
  BufferedAtomicStack(size_t N, Args... init_values)
      : values_(
            // thinking this might be equivalent to 'new T[N]'
            // but need to confirm
            reinterpret_cast<T*>(std::aligned_alloc(
                std::alignment_of_v<T>,
                aligned_buffer_bytes(N))),
            [](void* ptr) { std::free(ptr); }),
        next_(N) {
    for (const auto i : c10::irange(N)) {
      values_[i] = T(init_values...);
      next_[i] = i + 1;
    }
  }

  BufferedAtomicStack(const BufferedAtomicStack&) = delete;
  BufferedAtomicStack& operator=(const BufferedAtomicStack&) = delete;
  BufferedAtomicStack operator=(BufferedAtomicStack&&) = delete;
  BufferedAtomicStack(BufferedAtomicStack&&) = delete;

  ~BufferedAtomicStack() {
    if (!std::is_pointer_v<T>) {
      for (const size_t i :
           c10::irange(next_.size() /* equivalent to N in ctor */)) {
        values_[i].~T();
      }
    }
  }

  class ItemPtr;

  /* NOTE: the value will be released once ItemPtr goes out of scope */
  ItemPtr acquire_scoped(deleter_fn_ptr deleter_fn = &default_deleter) {
    return ItemPtr(acquire_inner(/* spin= */ true), this, deleter_fn);
  }

  /* NOTE: if acquired successfully (i.e., ret != nullptr), the value will be
   * released once ItemPtr goes out of scope */
  ItemPtr try_acquire_scoped(deleter_fn_ptr deleter_fn = &default_deleter) {
    return ItemPtr(acquire_inner(/* spin= */ false), this, deleter_fn);
  }

  /* NOTE: caller is responsible for releasing the value */
  T* acquire() {
    return acquire_inner(/* spin= */ true);
  }

  /* NOTE: if acquired successfully (i.e., ret != nullptr), caller is
   * responsible for releasing the value */
  T* try_acquire() {
    return acquire_inner(/* spin= */ false);
  }

  void release(T* v) {
    if (C10_UNLIKELY(v == nullptr)) {
      return;
    }

    auto* v_addr = &values_[0];
    TORCH_DCHECK_GE(v, v_addr);
    TORCH_DCHECK_LT(v, v_addr + next_.size());
    size_t cur_idx = v - v_addr;

    // set v's next value to the previous head
    next_[cur_idx] = head_.load(std::memory_order_acquire);
    // try to actually do the update, making v the stack head
    while (!head_.compare_exchange_weak(
        next_[cur_idx],
        cur_idx,
        std::memory_order_release,
        std::memory_order_acquire)) {
    };

    if constexpr (use_cv_on_empty) {
      if (next_[cur_idx] == next_.size() /* first item after empty stack -- notify possible waiter */) {
        cv_.notify_one();
      }
    }

    VLOG(1) << "released entry " << cur_idx;
  }

  class ItemPtr {
    friend class BufferedAtomicStack;

   public:
    ItemPtr() = delete;
    ItemPtr(ItemPtr&&) = default;
    ItemPtr& operator=(ItemPtr&&) = default;
    ItemPtr(const ItemPtr&) = delete;
    ItemPtr& operator=(const ItemPtr&) = delete;
    ~ItemPtr() = default;

    bool operator==(const T* other) const {
      return ptr_ == other;
    }

    T& operator*() {
      return *ptr_;
    }

    T* operator->() {
      return ptr_;
    }

   private:
    struct deleter {
      deleter(
          c10::BufferedAtomicStack<T, use_cv_on_empty>* queue,
          deleter_fn_ptr deleter_fn)
          : queue_(queue), deleter_fn_(deleter_fn) {}
      void operator()(T* p) {
        deleter_fn_(p);
        queue_->release(p);
      }

     private:
      c10::BufferedAtomicStack<T, use_cv_on_empty>* queue_;
      deleter_fn_ptr deleter_fn_;
    };

    /* implicit */ ItemPtr(
        T* ptr,
        c10::BufferedAtomicStack<T, use_cv_on_empty>* queue,
        deleter_fn_ptr deleter_fn = default_deleter)
        : ptr_(ptr, deleter{queue, deleter_fn}) {}

    std::unique_ptr<T, deleter> ptr_;
  };

 private:
  T* acquire_inner(
      bool spin = true /* return nullptr if can't acquire and spin == true */) {
    size_t head_idx = head_.load(std::memory_order_acquire);
    do {
      if (head_idx == next_.size() /* empty stack */) {
        if (!spin) {
          break;
        }

        if constexpr (use_cv_on_empty) {
          std::unique_lock<std::mutex> lk(mutex_);
          cv_.wait(lk, [&]() { return head_idx != next_.size(); });
        } else {
          std::this_thread::yield();
        }

        continue;
      }

      // note this doesn't have to be atomic,
      // since it's implied that if we are the
      // ones to acquire this head, then
      // this value can't be mutated by another
      // thread (since it can only be mutated
      // by releasing the head, which cannot be
      // released since it hasn't yet been acquired)
      size_t next_idx = next_[head_idx];

      // swap the head with its next value
      // which gives us ownership of the current
      // head until we release it
      if (head_.compare_exchange_weak(
              head_idx,
              next_idx,
              std::memory_order_release,
              // use relaxed ordering upon failure if we aren't spinning
              spin ? std::memory_order_acquire : std::memory_order_relaxed)) {
        return &values_[head_idx];
      }
    } while (spin);

    return nullptr;
  }

  // this holds the T values...
  // allows us to hold non-copyable
  // non-moveable types unlike std::vector
  std::unique_ptr<T[], void (*)(void*)> values_;

  std::vector<std::size_t> next_;

  // the newest un-owned value in the buffer
  std::atomic_size_t head_{0};

  std::conditional_t<use_cv_on_empty, std::mutex, std::monostate> mutex_;
  std::conditional_t<use_cv_on_empty, std::condition_variable, std::monostate>
      cv_;
};

} // namespace c10
