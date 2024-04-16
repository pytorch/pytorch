#include <c10/util/thread_name.h>

#include <algorithm>
#include <array>

#ifndef __GLIBC_PREREQ
#define __GLIBC_PREREQ(x, y) 0
#endif

#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 12) && !defined(__APPLE__) && \
    !defined(__ANDROID__)
#define C10_HAS_PTHREAD_SETNAME_NP
#endif

// Android only, prctl is only used when pthread_setname_np
// and pthread_getname_np are not avilable.
#if defined(__linux__)
#define C10_HAS_PRCTL_PR_SET_NAME 1
#else
#define C10_HAS_PRCTL_PR_SET_NAME 0
#endif

#ifdef C10_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

#if C10_HAS_PRCTL_PR_SET_NAME
#include <sys/prctl.h>
#endif

namespace c10 {

void setThreadName(std::string name) {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  constexpr size_t kMaxThreadName = 15;
  name.resize(std::min(name.size(), kMaxThreadName));

  pthread_setname_np(pthread_self(), name.c_str());
#elif C10_HAS_PRCTL_PR_SET_NAME
  // for Android prctl is used instead of pthread_setname_np
  // if Android NDK version is older than API level 9.
  prctl(PR_SET_NAME, name.c_str(), 0L, 0L, 0L);
#endif
}

std::string getThreadName() {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  std::array<char, 16> buf{};
  pthread_getname_np(pthread_self(), buf.data(), buf.size());
  return std::string(buf.data());
#elif C10_HAS_PRCTL_PR_SET_NAME
  std::array<char, 16> buf{};
  prctl(PR_GET_NAME, buf.data(), 0L, 0L, 0L);
  return std::string(buf.data());
#endif
  return "";
}

} // namespace c10
