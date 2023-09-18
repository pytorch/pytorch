#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace c10 {

// to_string, stoi and stod implementation for Android related stuff.
// Note(jiayq): Do not use the CAFFE2_TESTONLY_FORCE_STD_STRING_TEST macro
// outside testing code that lives under common_test.cc
#if defined(__ANDROID__) || defined(CAFFE2_TESTONLY_FORCE_STD_STRING_TEST)
#define CAFFE2_TESTONLY_WE_ARE_USING_CUSTOM_STRING_FUNCTIONS 1
template <typename T>
std::string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

#else
#define CAFFE2_TESTONLY_WE_ARE_USING_CUSTOM_STRING_FUNCTIONS 0
using std::to_string;
#endif // defined(__ANDROID__) || defined(CAFFE2_FORCE_STD_STRING_FALLBACK_TEST)

} // namespace c10

using std::stod;
using std::stoi;
using std::stoll;
using std::stoull;
#if defined(__ANDROID__) && __ANDROID_API__ < 21 && defined(__GLIBCXX__)
#include <cstdlib>
// std::strtoll isn't available on Android NDK platform < 21 when building
// with libstdc++, so bring the global version into std.
namespace std {
using ::strtoll;
}
#endif
