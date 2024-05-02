#pragma once
// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#ifdef USE_VULKAN_API

#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace detail {

struct CompileTimeEmptyString {
  operator const std::string&() const {
    static const std::string empty_string_literal;
    return empty_string_literal;
  }
  operator const char*() const {
    return "";
  }
};

template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

template <size_t N>
struct CanonicalizeStrTypes<char[N]> {
  using type = const char*;
};

inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  ss << t;
  return ss;
}

template <>
inline std::ostream& _str<CompileTimeEmptyString>(
    std::ostream& ss,
    const CompileTimeEmptyString&) {
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
  }
};

template <>
struct _str_wrapper<> final {
  static CompileTimeEmptyString call() {
    return CompileTimeEmptyString();
  }
};

} // namespace detail

template <typename... Args>
inline std::string concat_str(const Args&... args) {
  return detail::_str_wrapper<
      typename detail::CanonicalizeStrTypes<Args>::type...>::call(args...);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
