#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * Port of std::string_view with methods from C++20.
 * Implemented following the interface definition in
 * https://en.cppreference.com/w/cpp/string/basic_string_view
 * See there for the API documentation.
 *
 * Difference: We don't have a Traits template parameter because
 * std::char_traits isn't constexpr and we'd have to reimplement
 * std::char_traits if we wanted to use it with our constexpr basic_string_view.
 */
template <class CharT>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class basic_string_view final {
 public:
  using value_type = CharT;
  using pointer = CharT*;
  using const_pointer = const CharT*;
  using reference = CharT&;
  using const_reference = const CharT&;
  using const_iterator = const CharT*;
  using iterator = const_iterator;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  static constexpr size_type npos = size_type(-1);

  constexpr basic_string_view() noexcept : begin_(nullptr) {}

  explicit constexpr basic_string_view(const_pointer str, size_type count)
      : begin_(str), size_(count) {}

  /* implicit */ constexpr basic_string_view(const_pointer str)
      : basic_string_view(str, strlen_(str)) {}

  /* implicit */ basic_string_view(const ::std::basic_string<CharT>& str)
      : basic_string_view(str.data(), str.size()) {}

  /* implicit */ constexpr basic_string_view(
      const ::std::basic_string_view<CharT>& str)
      : basic_string_view(str.data(), str.size()) {}

  constexpr basic_string_view(const basic_string_view&) noexcept = default;

  constexpr basic_string_view& operator=(
      const basic_string_view& rhs) noexcept = default;

  constexpr operator ::std::basic_string_view<CharT>() const {
    return ::std::basic_string_view<CharT>(data(), size());
  }

  explicit operator ::std::basic_string<CharT>() const {
    return ::std::basic_string<CharT>(data(), size());
  }

  constexpr const_iterator begin() const noexcept {
    return cbegin();
  }

  constexpr const_iterator cbegin() const noexcept {
    return begin_;
  }

  constexpr const_iterator end() const noexcept {
    return cend();
  }

  constexpr const_iterator cend() const noexcept {
    return begin_ + size_;
  }

  constexpr const_reverse_iterator rbegin() const noexcept {
    return crbegin();
  }

  constexpr const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(this->end());
  }

  constexpr const_reverse_iterator rend() const noexcept {
    return crend();
  }

  constexpr const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(this->begin());
  }

  friend constexpr const_iterator begin(basic_string_view sv) noexcept {
    return sv.begin();
  }

  friend constexpr const_iterator end(basic_string_view sv) noexcept {
    return sv.end();
  }

  constexpr const_reference operator[](size_type pos) const {
    // TODO: split out
    return at_(pos);
  }

  constexpr const_reference at(size_type pos) const {
#if !defined( \
    __CUDA_ARCH__) // CUDA doesn't like std::out_of_range in device code
    return C10_UNLIKELY(pos >= size_)
        ? (throw std::out_of_range(
               "string_view::operator[] or string_view::at() out of range. Index: " +
               std::to_string(pos) + ", size: " + std::to_string(size())),
           at_(0))
        : at_(pos);
#else
    return at_(pos);
#endif
  }

  constexpr const_reference front() const {
    return *begin_;
  }

  constexpr const_reference back() const {
    return *(begin_ + size_ - 1);
  }

  constexpr const_pointer data() const noexcept {
    return begin_;
  }

  constexpr size_type size() const noexcept {
    return size_;
  }

  constexpr size_type length() const noexcept {
    return size();
  }

  constexpr size_type max_size() const noexcept {
    return std::numeric_limits<difference_type>::max();
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size() == 0;
  }

  constexpr void remove_prefix(size_type n) {
    if (n > size()) {
      throw std::out_of_range(
          "basic_string_view::remove_prefix: out of range. PrefixLength: " +
          std::to_string(n) + ", size: " + std::to_string(size()));
    }
    begin_ += n;
    size_ -= n;
  }

  constexpr void remove_suffix(size_type n) {
    if (n > size()) {
      throw std::out_of_range(
          "basic_string_view::remove_suffix: out of range. SuffixLength: " +
          std::to_string(n) + ", size: " + std::to_string(size()));
    }
    size_ -= n;
  }

  constexpr void swap(basic_string_view& sv) noexcept {
    auto tmp = *this;
    *this = sv;
    sv = tmp;
  }

  size_type copy(pointer dest, size_type count, size_type pos = 0) const {
    if (pos > size_) {
      throw std::out_of_range(
          "basic_string_view::copy: out of range. Index: " +
          std::to_string(pos) + ", size: " + std::to_string(size()));
    }
    size_type copy_length = std::min(count, size_ - pos);
    for (auto iter = begin() + pos, end = iter + copy_length; iter != end;) {
      *(dest++) = *(iter++);
    }
    return copy_length;
  }

  constexpr basic_string_view substr(size_type pos = 0, size_type count = npos)
      const {
#if !defined( \
    __CUDA_ARCH__) // CUDA doesn't like std::out_of_range in device code
    return (pos > size_)
        ? (throw std::out_of_range(
               "basic_string_view::substr parameter out of bounds. Index: " +
               std::to_string(pos) + ", size: " + std::to_string(size())),
           substr_())
        : substr_(pos, count);
#else
    return substr_(pos, count);
#endif
  }

  constexpr int compare(basic_string_view rhs) const noexcept {
    // Write it iteratively. This is faster.
    for (size_t i = 0, end = std::min(size(), rhs.size()); i < end; ++i) {
      if (at_(i) < rhs.at_(i)) {
        return -1;
      } else if (at_(i) > rhs.at_(i)) {
        return 1;
      }
    }
    if (size() < rhs.size()) {
      return -1;
    } else if (size() > rhs.size()) {
      return 1;
    }
    return 0;
  }

  constexpr int compare(size_type pos1, size_type count1, basic_string_view v)
      const {
    return substr(pos1, count1).compare(v);
  }

  constexpr int compare(
      size_type pos1,
      size_type count1,
      basic_string_view v,
      size_type pos2,
      size_type count2) const {
    return substr(pos1, count1).compare(v.substr(pos2, count2));
  }

  constexpr int compare(const_pointer s) const {
    return compare(basic_string_view(s));
  }

  constexpr int compare(size_type pos1, size_type count1, const_pointer s)
      const {
    return substr(pos1, count1).compare(basic_string_view(s));
  }

  constexpr int compare(
      size_type pos1,
      size_type count1,
      const_pointer s,
      size_type count2) const {
    return substr(pos1, count1).compare(basic_string_view(s, count2));
  }

  friend constexpr bool operator==(
      basic_string_view lhs,
      basic_string_view rhs) noexcept {
    return lhs.equals_(rhs);
  }

  friend constexpr bool operator!=(
      basic_string_view lhs,
      basic_string_view rhs) noexcept {
    return !(lhs == rhs);
  }

  friend constexpr bool operator<(
      basic_string_view lhs,
      basic_string_view rhs) noexcept {
    return lhs.compare(rhs) < 0;
  }

  friend constexpr bool operator>=(
      basic_string_view lhs,
      basic_string_view rhs) noexcept {
    return !(lhs < rhs);
  }

  friend constexpr bool operator>(
      basic_string_view lhs,
      basic_string_view rhs) noexcept {
    return rhs < lhs;
  }

  friend constexpr bool operator<=(
      basic_string_view lhs,
      basic_string_view rhs) noexcept {
    return !(lhs > rhs);
  }

  constexpr bool starts_with(basic_string_view prefix) const noexcept {
    return (prefix.size() > size()) ? false
                                    : prefix.equals_(substr_(0, prefix.size()));
  }

  constexpr bool starts_with(CharT prefix) const noexcept {
    return !empty() && prefix == front();
  }

  constexpr bool starts_with(const_pointer prefix) const {
    return starts_with(basic_string_view(prefix));
  }

  constexpr bool ends_with(basic_string_view suffix) const noexcept {
    return (suffix.size() > size())
        ? false
        : suffix.equals_(substr_(size() - suffix.size(), suffix.size()));
  }

  constexpr bool ends_with(CharT suffix) const noexcept {
    return !empty() && suffix == back();
  }

  constexpr bool ends_with(const_pointer suffix) const {
    return ends_with(basic_string_view(suffix));
  }

  constexpr size_type find(basic_string_view v, size_type pos = 0)
      const noexcept {
    if (v.size() == 0) {
      return pos <= size() ? pos : npos;
    }

    if (pos + v.size() <= size()) {
      for (size_type cur = pos, end = size() - v.size(); cur <= end; ++cur) {
        if (v.at_(0) == at_(cur) &&
            v.substr_(1).equals_(substr_(cur + 1, v.size() - 1))) {
          return cur;
        }
      }
    }
    return npos;
  }

  constexpr size_type find(CharT ch, size_type pos = 0) const noexcept {
    return find_first_if_(pos, charIsEqual_{ch});
  }

  constexpr size_type find(const_pointer s, size_type pos, size_type count)
      const {
    return find(basic_string_view(s, count), pos);
  }

  constexpr size_type find(const_pointer s, size_type pos = 0) const {
    return find(basic_string_view(s), pos);
  }

  constexpr size_type rfind(basic_string_view v, size_type pos = npos)
      const noexcept {
    // Write it iteratively. This is faster.
    if (v.size() == 0) {
      return pos <= size() ? pos : size();
    }

    if (v.size() <= size()) {
      pos = std::min(size() - v.size(), pos);
      do {
        if (v.at_(0) == at_(pos) &&
            v.substr_(1).equals_(substr_(pos + 1, v.size() - 1))) {
          return pos;
        }
      } while (pos-- > 0);
    }
    return npos;
  }

  constexpr size_type rfind(CharT ch, size_type pos = npos) const noexcept {
    return find_last_if_(pos, charIsEqual_{ch});
  }

  constexpr size_type rfind(const_pointer s, size_type pos, size_type count)
      const {
    return rfind(basic_string_view(s, count), pos);
  }

  constexpr size_type rfind(const_pointer s, size_type pos = npos) const {
    return rfind(basic_string_view(s), pos);
  }

  constexpr size_type find_first_of(basic_string_view v, size_type pos = 0)
      const noexcept {
    return find_first_if_(pos, stringViewContainsChar_{v});
  }

  constexpr size_type find_first_of(CharT ch, size_type pos = 0)
      const noexcept {
    return find_first_if_(pos, charIsEqual_{ch});
  }

  constexpr size_type find_first_of(
      const_pointer s,
      size_type pos,
      size_type count) const {
    return find_first_of(basic_string_view(s, count), pos);
  }

  constexpr size_type find_first_of(const_pointer s, size_type pos = 0) const {
    return find_first_of(basic_string_view(s), pos);
  }

  constexpr size_type find_last_of(basic_string_view v, size_type pos = npos)
      const noexcept {
    return find_last_if_(pos, stringViewContainsChar_{v});
  }

  constexpr size_type find_last_of(CharT ch, size_type pos = npos)
      const noexcept {
    return find_last_if_(pos, charIsEqual_{ch});
  }

  constexpr size_type find_last_of(
      const_pointer s,
      size_type pos,
      size_type count) const {
    return find_last_of(basic_string_view(s, count), pos);
  }

  constexpr size_type find_last_of(const_pointer s, size_type pos = npos)
      const {
    return find_last_of(basic_string_view(s), pos);
  }

  constexpr size_type find_first_not_of(basic_string_view v, size_type pos = 0)
      const noexcept {
    return find_first_if_(pos, stringViewDoesNotContainChar_{v});
  }

  constexpr size_type find_first_not_of(CharT ch, size_type pos = 0)
      const noexcept {
    return find_first_if_(pos, charIsNotEqual_{ch});
  }

  constexpr size_type find_first_not_of(
      const_pointer s,
      size_type pos,
      size_type count) const {
    return find_first_not_of(basic_string_view(s, count), pos);
  }

  constexpr size_type find_first_not_of(const_pointer s, size_type pos = 0)
      const {
    return find_first_not_of(basic_string_view(s), pos);
  }

  constexpr size_type find_last_not_of(
      basic_string_view v,
      size_type pos = npos) const noexcept {
    return find_last_if_(pos, stringViewDoesNotContainChar_{v});
  }

  constexpr size_type find_last_not_of(CharT ch, size_type pos = npos)
      const noexcept {
    return find_last_if_(pos, charIsNotEqual_{ch});
  }

  constexpr size_type find_last_not_of(
      const_pointer s,
      size_type pos,
      size_type count) const {
    return find_last_not_of(basic_string_view(s, count), pos);
  }

  constexpr size_type find_last_not_of(const_pointer s, size_type pos = npos)
      const {
    return find_last_not_of(basic_string_view(s), pos);
  }

 private:
  static constexpr size_type strlen_(const_pointer str) noexcept {
    const_pointer current = str;
    while (*current != '\0') {
      ++current;
    }
    return current - str;
  }

  constexpr const_reference at_(size_type pos) const noexcept {
    return *(begin_ + pos);
  }

  constexpr basic_string_view substr_(size_type pos = 0, size_type count = npos)
      const {
    return basic_string_view{begin_ + pos, std::min(count, size() - pos)};
  }

  template <class Condition>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  constexpr size_type find_first_if_(size_type pos, Condition&& condition)
      const noexcept {
    if (pos + 1 <= size()) {
      for (size_type cur = pos; cur < size(); ++cur) {
        if (condition(at_(cur))) {
          return cur;
        }
      }
    }
    return npos;
  }

  template <class Condition>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  constexpr size_type find_last_if_(size_type pos, Condition&& condition)
      const noexcept {
    // Write it iteratively. This is faster.
    if (size() > 0) {
      pos = std::min(size() - 1, pos);
      do {
        if (condition(at_(pos))) {
          return pos;
        }
      } while (pos-- > 0);
    }
    return npos;
  }

  constexpr bool equals_(basic_string_view rhs) const {
    // We don't use string_view::compare() here but implement it manually
    // because only looking at equality allows for more optimized code.
#if defined(__GNUC__) && !defined(__CUDACC__)
    return size() == rhs.size() &&
        0 == __builtin_memcmp(data(), rhs.data(), size());
#else
    if (size() != rhs.size()) {
      return false;
    }
    // Yes, memcmp would be laster than this loop, but memcmp isn't constexpr
    // and I didn't feel like implementing a constexpr memcmp variant.
    // TODO At some point this should probably be done, including tricks
    // like comparing one machine word instead of a byte per iteration.
    for (typename basic_string_view<CharT>::size_type pos = 0; pos < size();
         ++pos) {
      if (at_(pos) != rhs.at_(pos)) {
        return false;
      }
    }
    return true;
#endif
  }

  struct charIsEqual_ final {
    CharT expected;
    constexpr bool operator()(CharT actual) const noexcept {
      return expected == actual;
    }
  };

  struct charIsNotEqual_ final {
    CharT expected;
    constexpr bool operator()(CharT actual) const noexcept {
      return expected != actual;
    }
  };

  struct stringViewContainsChar_ final {
    basic_string_view expected;
    constexpr bool operator()(CharT ch) const noexcept {
      return npos != expected.find(ch);
    }
  };

  struct stringViewDoesNotContainChar_ final {
    basic_string_view expected;
    constexpr bool operator()(CharT ch) const noexcept {
      return npos == expected.find(ch);
    }
  };

  const_pointer begin_;
  size_type size_{};
};

template <class CharT>
inline std::basic_ostream<CharT>& operator<<(
    std::basic_ostream<CharT>& stream,
    basic_string_view<CharT> sv) {
  // The rules for operator<< are quite complex, so lets defer to the
  // STL implementation.
  using std_string_type = ::std::basic_string_view<CharT>;
  return stream << std_string_type(sv.data(), sv.size());
}

template <class CharT>
constexpr inline void swap(
    basic_string_view<CharT>& lhs,
    basic_string_view<CharT>& rhs) noexcept {
  lhs.swap(rhs);
}
using string_view = std::string_view;
using c10_string_view = basic_string_view<char>;

// NOTE: In C++20, this function should be replaced by string_view.starts_with
constexpr bool starts_with(
    const std::string_view s,
    const std::string_view prefix) noexcept {
  return (prefix.size() > s.size()) ? false
                                    : prefix == s.substr(0, prefix.size());
}

// NOTE: In C++20, this function should be replaced by string_view.starts_with
constexpr bool starts_with(
    const std::string_view s,
    const char prefix) noexcept {
  return !s.empty() && prefix == s.front();
}

// NOTE: In C++20, this function should be replaced by string_view.ends_with
constexpr bool ends_with(
    const std::string_view s,
    const std::string_view suffix) noexcept {
  return (suffix.size() > s.size())
      ? false
      : suffix == s.substr(s.size() - suffix.size(), suffix.size());
}

// NOTE: In C++20, this function should be replaced by string_view.ends_with
constexpr bool ends_with(const std::string_view s, const char prefix) noexcept {
  return !s.empty() && prefix == s.back();
}

} // namespace c10

namespace std {
template <class CharT>
struct hash<::c10::basic_string_view<CharT>> {
  size_t operator()(::c10::basic_string_view<CharT> x) const {
    // The standard says that std::string_view hashing must do the same as
    // std::string hashing but leaves the details of std::string hashing
    // up to the implementer. So, to be conformant, we need to reuse and
    // existing STL type's hash function. The std::string fallback is probably
    // slow but the only way to be conformant.

    using std_string_type = ::std::basic_string_view<CharT>;
    return ::std::hash<std_string_type>{}(std_string_type(x.data(), x.size()));
  }
};
} // namespace std
