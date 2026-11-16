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

#ifdef C10_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

namespace c10 {

#ifdef C10_HAS_PTHREAD_SETNAME_NP
namespace {
// pthreads has a limit of 16 characters including the null termination byte.
constexpr size_t kMaxThreadName = 15;
} // namespace
#endif

void setThreadName(std::string name) {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  name.resize(std::min(name.size(), kMaxThreadName));

  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

std::string getThreadName() {
#ifdef C10_HAS_PTHREAD_SETNAME_NP
  std::array<char, kMaxThreadName + 1> name{};
  pthread_getname_np(pthread_self(), name.data(), name.size());
  return name.data();
#else
  return "";
#endif
}

} // namespace c10
