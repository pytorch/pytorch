#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <utility>

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

namespace torch::profiler::impl {

// ============================================================================
// == AppendOnlyList ==========================================================
// ============================================================================
//   During profiling, we have a very predictable access pattern: we only
// append to the end of the container. We can specialize and outperform both
// std::vector (which must realloc) and std::deque (which performs a double
// indirection), and this class of operation is sufficiently important to the
// profiling hot path to warrant specializing:
//   https://godbolt.org/z/rTjozf1c4
//   https://quick-bench.com/q/mmfuu71ogwaiULDCJyHdKnHZms4    (Prototype #1,
//   int) https://quick-bench.com/q/5vWDW6jjdXVdoffev2zst8D09no    (Prototype
//   #1, int pair) https://quick-bench.com/q/IfEkfAQMeJSNBA52xtMP6Agcl-Q
//   (Prototype #2, int pair)
//   https://quick-bench.com/q/wJV2lKmuXL4XyGJzcI5hs4gEHFg    (Prototype #3, int
//   pair) https://quick-bench.com/q/xiO8ZaBEkYRYUA9dFrMuPLlW9fo    (Full impl,
//   int pair)
// AppendOnlyList has 2x lower emplace overhead compared to more generic STL
// containers.
//
//   The optimal value of `ChunkSize` will vary by use case, but testing shows
// that a value of 1024 does a good job amortizing the `malloc` cost of growth.
// Performance drops off for larger values, so testing on a case-by-case basis
// is recommended if performance is absolutely critical.

template <
    typename T,
    size_t ChunkSize,
    template <typename U, size_t N> class block_t = std::array>
class AppendOnlyList {
 public:
  using array_t = block_t<T, ChunkSize>;
  static_assert(
      std::is_base_of_v<std::array<T, ChunkSize>, array_t>,
      "AppendOnlyList expects raw low level pointer storage.");
  static_assert(ChunkSize > 0, "Block cannot be empty.");

  AppendOnlyList() : buffer_last_{buffer_.before_begin()} {}
  AppendOnlyList(const AppendOnlyList&) = delete;
  AppendOnlyList(AppendOnlyList&&) = delete;
  AppendOnlyList& operator=(const AppendOnlyList&) = delete;
  AppendOnlyList& operator=(AppendOnlyList&&) = delete;
  ~AppendOnlyList() = default;

  size_t size() const {
    return n_blocks_ * ChunkSize - (size_t)(end_ - next_);
  }

  template <class... Args>
  T* emplace_back(Args&&... args) {
    maybe_grow();
    if constexpr (
        std::is_trivially_destructible_v<T> &&
        std::is_trivially_destructible_v<array_t>) {
      ::new ((void*)next_) T{std::forward<Args>(args)...};
    } else {
      *next_ = T{std::forward<Args>(args)...};
    }
    return next_++;
  }

  template <typename T0>
  std::enable_if_t<std::is_same_v<T0, T> && std::is_trivially_copyable_v<T>>
  copy(c10::ArrayRef<T0> src) {
    size_t n = src.size();
    if (C10_UNLIKELY(n == 0)) {
      return;
    }
    maybe_grow();
    if (C10_LIKELY(next_ && (next_ + n <= end_))) {
      std::memcpy((void*)next_, (void*)src.begin(), n * sizeof(T0));
      next_ += n;
    } else {
      // We could chunk this into several `memcpy`s, but because we expect this
      // fallback to be infrequent (n << ChunkSize) the performance impact is
      // negligible.
      for (auto i : src) {
        emplace_back(i);
      }
    }
  }

  void clear() {
    buffer_.clear();
    buffer_last_ = buffer_.before_begin();
    n_blocks_ = 0;
    next_ = nullptr;
    end_ = nullptr;
  }

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;

    Iterator(std::forward_list<array_t>& buffer, const size_t size)
        : block_{buffer.begin()}, size_{size} {}

    // End iterator.
    Iterator() = default;

    bool exhausted() const {
      return current_ >= size_;
    }

    reference operator*() const {
      return *current_ptr(/*checked=*/true);
    }
    pointer operator->() {
      return current_ptr(/*checked=*/true);
    }

    // Prefix increment
    Iterator& operator++() {
      if (!(++current_ % ChunkSize)) {
        block_++;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.current_ptr() == b.current_ptr();
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return a.current_ptr() != b.current_ptr();
    }

    std::pair<array_t*, size_t> address() const {
      if (current_ >= size_) {
        return {nullptr, 0};
      }
      return {&(*block_), current_ % ChunkSize};
    }

   private:
    T* current_ptr(bool checked = false) const {
      auto a = address();
      if (a.first == nullptr) {
        TORCH_INTERNAL_ASSERT(!checked, "Invalid access on AppendOnlyList.");
        return nullptr;
      }
      return a.first->data() + a.second;
    }

    typename std::forward_list<array_t>::iterator block_;
    size_t current_{0};
    size_t size_{0};
  };

  Iterator begin() {
    return Iterator(buffer_, size());
  }
  Iterator end() {
    return Iterator();
  }
  // TODO: cbegin and cend()

 private:
  void maybe_grow() {
    if (C10_UNLIKELY(next_ == end_)) {
      buffer_last_ = buffer_.emplace_after(buffer_last_);
      n_blocks_++;
      next_ = buffer_last_->data();
      end_ = next_ + ChunkSize;
    }
  }

  std::forward_list<array_t> buffer_;

  // We maintain a pointer to the last element of `buffer_` so that we can
  // insert at the end in O(1) time.
  size_t n_blocks_{0};
  T* next_{nullptr};
  T* end_{nullptr};

 protected:
  typename std::forward_list<array_t>::iterator buffer_last_;
};

} // namespace torch::profiler::impl
