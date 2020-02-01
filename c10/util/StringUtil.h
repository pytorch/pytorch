#ifndef C10_UTIL_STRINGUTIL_H_
#define C10_UTIL_STRINGUTIL_H_

#include <c10/macros/Macros.h>
#include <c10/util/string_utils.h>
#include <c10/util/string_view.h>

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace c10 {

namespace detail {

// Obtains the base name from a full path.
C10_API std::string StripBasename(const std::string& full_path);

template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

template <size_t N>
struct CanonicalizeStrTypes<char[N]> {
  using type = const char *;
};


inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  ss << t;
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

template <typename... Args>
inline std::string _str_wrapper(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
}

} // namespace detail

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline std::string str(const Args&... args) {
  return detail::_str_wrapper<typename detail::CanonicalizeStrTypes<Args>::type...>(args...);
}

// Specializations for already-a-string types.
template <>
inline std::string str(const std::string& str) {
  return str;
}
inline std::string str(const char* c_str) {
  return c_str;
}

template <class Container>
inline std::string Join(const std::string& delimiter, const Container& v) {
  std::stringstream s;
  int cnt = static_cast<int64_t>(v.size()) - 1;
  for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
    s << (*i) << (cnt ? delimiter : "");
  }
  return s.str();
}

// Replace all occurrences of "from" substring to "to" string.
// Returns number of replacements
size_t C10_API ReplaceAll(std::string& s, const char* from, const char* to);

/// Represents a location in source code (for debugging).
struct C10_API SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

// unix isprint but insensitive to locale
inline static bool isPrint(char s) {
  return s > 0x1f && s < 0x7f;
}

inline void printQuotedString(std::ostream& stmt, const std::string& str) {
  stmt << "\"";
  for (auto s : str) {
    switch (s) {
      case '\\':
        stmt << "\\\\";
        break;
      case '\'':
        stmt << "\\'";
        break;
      case '\"':
        stmt << "\\\"";
        break;
      case '\a':
        stmt << "\\a";
        break;
      case '\b':
        stmt << "\\b";
        break;
      case '\f':
        stmt << "\\f";
        break;
      case '\n':
        stmt << "\\n";
        break;
      case '\r':
        stmt << "\\r";
        break;
      case '\t':
        stmt << "\\t";
        break;
      case '\v':
        stmt << "\\v";
        break;
      default:
        if (isPrint(s)) {
          stmt << s;
        } else {
          // C++ io has stateful formatting settings. Messing with
          // them is probably worse than doing this manually.
          char buf[4] = "000";
          buf[2] += s % 8;
          s /= 8;
          buf[1] += s % 8;
          s /= 8;
          buf[0] += s;
          stmt << "\\" << buf;
        }
        break;
    }
  }
  stmt << "\"";
}

inline std::vector<std::string> split(
    c10::string_view str,
    char delim,
    bool ignoreEmpty) {
  std::vector<std::string> out;

  const size_t strSize = str.size();

  size_t tokenStartPos = 0;
  size_t tokenSize = 0;
  for (size_t i = 0; i <= strSize - 1; ++i) {
    if (str[i] == delim) {
      if (!ignoreEmpty || tokenSize > 0) {
        out.emplace_back(str.substr(tokenStartPos, tokenSize));
      }

      tokenStartPos = i + 1;
      tokenSize = 0;
    } else {
      ++tokenSize;
    }
  }
  tokenSize = strSize - tokenStartPos;
  if (!ignoreEmpty || tokenSize > 0) {
    out.emplace_back(str.substr(tokenStartPos, tokenSize));
  }
  return out;
}

} // namespace c10

#endif // C10_UTIL_STRINGUTIL_H_
