// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(OPENTELEMETRY_STL_VERSION)
#  if OPENTELEMETRY_STL_VERSION >= 2017
#    include "opentelemetry/std/string_view.h"
#    define OPENTELEMETRY_HAVE_STD_STRING_VIEW
#  endif
#endif

#if !defined(OPENTELEMETRY_HAVE_STD_STRING_VIEW)
#  include <algorithm>
#  include <cstddef>
#  include <cstring>
#  include <ostream>
#  include <stdexcept>
#  include <string>

#  include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{

using Traits = std::char_traits<char>;

/**
 * Back port of std::string_view to work with pre-cpp-17 compilers.
 *
 * Note: This provides a subset of the methods available on std::string_view but
 * tries to be as compatible as possible with the std::string_view interface.
 */
class string_view
{
public:
  typedef std::size_t size_type;

  static constexpr size_type npos = static_cast<size_type>(-1);

  string_view() noexcept : length_(0), data_(nullptr) {}

  string_view(const char *str) noexcept : length_(std::strlen(str)), data_(str) {}

  string_view(const std::basic_string<char> &str) noexcept
      : length_(str.length()), data_(str.c_str())
  {}

  string_view(const char *str, size_type len) noexcept : length_(len), data_(str) {}

  explicit operator std::string() const { return {data_, length_}; }

  const char *data() const noexcept { return data_; }

  bool empty() const noexcept { return length_ == 0; }

  size_type length() const noexcept { return length_; }

  size_type size() const noexcept { return length_; }

  const char *begin() const noexcept { return data(); }

  const char *end() const noexcept { return data() + length(); }

  const char &operator[](size_type i) { return *(data() + i); }

  string_view substr(size_type pos, size_type n = npos) const
  {
    if (pos > length_)
    {
#  if __EXCEPTIONS
      throw std::out_of_range{"opentelemetry::nostd::string_view"};
#  else
      std::terminate();
#  endif
    }
    n = (std::min)(n, length_ - pos);
    return string_view(data_ + pos, n);
  }

  int compare(string_view v) const noexcept
  {
    size_type len = (std::min)(size(), v.size());
    int result    = Traits::compare(data(), v.data(), len);
    if (result == 0)
      result = size() == v.size() ? 0 : (size() < v.size() ? -1 : 1);
    return result;
  }

  int compare(size_type pos1, size_type count1, string_view v) const
  {
    return substr(pos1, count1).compare(v);
  }

  int compare(size_type pos1,
              size_type count1,
              string_view v,
              size_type pos2,
              size_type count2) const
  {
    return substr(pos1, count1).compare(v.substr(pos2, count2));
  }

  int compare(const char *s) const { return compare(string_view(s)); }

  int compare(size_type pos1, size_type count1, const char *s) const
  {
    return substr(pos1, count1).compare(string_view(s));
  }

  int compare(size_type pos1, size_type count1, const char *s, size_type count2) const
  {
    return substr(pos1, count1).compare(string_view(s, count2));
  }

  size_type find(char ch, size_type pos = 0) const noexcept
  {
    size_type res = npos;
    if (pos < length())
    {
      auto found = Traits::find(data() + pos, length() - pos, ch);
      if (found)
      {
        res = found - data();
      }
    }
    return res;
  }

  bool operator<(const string_view v) const noexcept { return compare(v) < 0; }

  bool operator>(const string_view v) const noexcept { return compare(v) > 0; }

private:
  // Note: uses the same binary layout as libstdc++'s std::string_view
  // See
  // https://github.com/gcc-mirror/gcc/blob/e0c554e4da7310df83bb1dcc7b8e6c4c9c5a2a4f/libstdc%2B%2B-v3/include/std/string_view#L466-L467
  size_type length_;
  const char *data_;
};

inline bool operator==(string_view lhs, string_view rhs) noexcept
{
  return lhs.length() == rhs.length() &&
#  if defined(_MSC_VER)
#    if _MSC_VER >= 1900 && _MSC_VER <= 1911
         // Avoid SCL error in Visual Studio 2015, VS2017 update 1 to update 4
         (std::memcmp(lhs.data(), rhs.data(), lhs.length()) == 0);
#    else
         std::equal(lhs.data(), lhs.data() + lhs.length(), rhs.data());
#    endif
#  else
         std::equal(lhs.data(), lhs.data() + lhs.length(), rhs.data());
#  endif
}

inline bool operator==(string_view lhs, const std::string &rhs) noexcept
{
  return lhs == string_view(rhs);
}

inline bool operator==(const std::string &lhs, string_view rhs) noexcept
{
  return string_view(lhs) == rhs;
}

inline bool operator==(string_view lhs, const char *rhs) noexcept
{
  return lhs == string_view(rhs);
}

inline bool operator==(const char *lhs, string_view rhs) noexcept
{
  return string_view(lhs) == rhs;
}

inline bool operator!=(string_view lhs, string_view rhs) noexcept
{
  return !(lhs == rhs);
}

inline bool operator!=(string_view lhs, const std::string &rhs) noexcept
{
  return !(lhs == rhs);
}

inline bool operator!=(const std::string &lhs, string_view rhs) noexcept
{
  return !(lhs == rhs);
}

inline bool operator!=(string_view lhs, const char *rhs) noexcept
{
  return !(lhs == rhs);
}

inline bool operator!=(const char *lhs, string_view rhs) noexcept
{
  return !(lhs == rhs);
}

inline std::ostream &operator<<(std::ostream &os, string_view s)
{
  return os.write(s.data(), static_cast<std::streamsize>(s.length()));
}
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE

namespace std
{
template <>
struct hash<OPENTELEMETRY_NAMESPACE::nostd::string_view>
{
  std::size_t operator()(const OPENTELEMETRY_NAMESPACE::nostd::string_view &k) const
  {
    // TODO: for C++17 that has native support for std::basic_string_view it would
    // be more performance-efficient to provide a zero-copy hash.
    auto s = std::string(k.data(), k.size());
    return std::hash<std::string>{}(s);
  }
};
}  // namespace std
#endif /* OPENTELEMETRY_HAVE_STD_STRING_VIEW */
