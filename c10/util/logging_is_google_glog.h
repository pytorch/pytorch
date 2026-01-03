#ifndef C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
#define C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_

#include <map>
#include <set>
#include <vector>

#include <iomanip> // because some of the caffe2 code uses e.g. std::setw
// Using google glog. For glog 0.3.2 versions, stl_logging.h needs to be before
// logging.h to actually use stl_logging. Because template magic.
// In addition, we do not do stl logging in .cu files because nvcc does not like
// it. Some mobile platforms do not like stl_logging, so we add an
// overload in that case as well.

#ifdef __CUDACC__
#include <cuda.h>
#endif

#if !defined(__CUDACC__) && !defined(C10_USE_MINIMAL_GLOG)
#include <glog/stl_logging.h>

// Old versions of glog don't declare this using declaration, so help
// them out.  Fortunately, C++ won't complain if you declare the same
// using declaration multiple times.
namespace std {
using ::operator<<;
}

#else // !defined(__CUDACC__) && !defined(C10_USE_MINIMAL_GLOG)

// In the cudacc compiler scenario, we will simply ignore the container
// printout feature. Basically we need to register a fake overload for
// vector/string - here, we just ignore the entries in the logs.

namespace std {
#define INSTANTIATE_FOR_CONTAINER(container)                      \
  template <class... Types>                                       \
  ostream& operator<<(ostream& out, const container<Types...>&) { \
    return out;                                                   \
  }

INSTANTIATE_FOR_CONTAINER(vector)
INSTANTIATE_FOR_CONTAINER(map)
INSTANTIATE_FOR_CONTAINER(set)
#undef INSTANTIATE_FOR_CONTAINER
} // namespace std

#endif

#include <c10/util/logging_common.h>
#include <glog/logging.h>

namespace c10 {

[[noreturn]] void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller);

template <typename T>
T& CheckNotNullCommon(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  if (t == nullptr) {
    MessageLogger(file, line, ::google::GLOG_FATAL, fatal).stream()
        << "Check failed: '" << names << "' must be non NULL. ";
  }
  return t;
}

template <typename T>
T* CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T* t,
    bool fatal) {
  return CheckNotNullCommon(file, line, names, t, fatal);
}

template <typename T>
T& CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  return CheckNotNullCommon(file, line, names, t, fatal);
}

} // namespace c10

// Log with source location information override (to be used in generic
// warning/error handlers implemented as functions, not macros)
//
// Note, we don't respect GOOGLE_STRIP_LOG here for simplicity
#define LOG_AT_FILE_LINE(n, file, line) \
  ::google::LogMessage(file, line, ::google::GLOG_##n).stream()

#endif // C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
