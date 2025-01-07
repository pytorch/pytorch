#include <cerrno>
#include <cstring>

#include <c10/util/error.h>
#include <string>
#include <type_traits>

namespace c10::utils {

// Get an error string in the thread-safe way.
std::string str_error(int errnum) {
  auto old_errno = errno;
  std::string buf(256, '\0');
#if defined(_WIN32)
  auto res [[maybe_unused]] = strerror_s(buf.data(), buf.size(), errnum);
  buf.resize(strlen(buf.c_str()));
#else
  auto res [[maybe_unused]] = strerror_r(errnum, buf.data(), buf.size());
  if constexpr (std::is_same_v<decltype(res), int>) {
    buf.resize(strlen(buf.c_str()));
  } else {
    if (res) {
      buf = res;
    }
  }
#endif
  errno = old_errno;
  return buf;
}

} // namespace c10::utils
