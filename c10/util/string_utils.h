#pragma once

#include <sstream>
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

inline int stoi(const std::string& str, std::size_t* pos = 0) {
  std::stringstream ss;
  int n = 0;
  ss << str;
  ss >> n;
  if (pos) {
    if (ss.tellg() == std::streampos(-1)) {
      *pos = str.size();
    } else {
      *pos = ss.tellg();
    }
  }
  return n;
}

inline uint64_t stoull(const std::string& str) {
  std::stringstream ss;
  uint64_t n = 0;
  ss << str;
  ss >> n;
  return n;
}

inline double stod(const std::string& str, std::size_t* pos = 0) {
  std::stringstream ss;
  ss << str;
  double val = 0;
  ss >> val;
  if (pos) {
    if (ss.tellg() == std::streampos(-1)) {
      *pos = str.size();
    } else {
      *pos = ss.tellg();
    }
  }
  return val;
}

inline long long stoll(const std::string& str) {
  // std::stoll doesn't exist in our Android environment, we need to implement
  // it ourselves.
  std::istringstream s(str);
  long long result = 0;
  s >> result;
  return result;
}

inline long long stoll(const std::string& str, size_t pos, int base) {
  // std::stoll doesn't exist in our Android environment, we need to implement
  // it ourselves.
  std::stringstream s;
  if (str.size() > 0 && str.at(0) == '0') {
    if (str.size() > 1 && (str.at(1) == 'x' || str.at(1) == 'X')) {
      s << std::hex << str;
    } else {
      s << std::oct << str;
    }
  } else {
    s << str;
  }
  long long result = 0;
  s >> result;
  return result;
}

#else
#define CAFFE2_TESTONLY_WE_ARE_USING_CUSTOM_STRING_FUNCTIONS 0
using std::stod;
using std::stoi;
using std::stoull;
using std::stoll;
using std::to_string;
#endif // defined(__ANDROID__) || defined(CAFFE2_FORCE_STD_STRING_FALLBACK_TEST)

} // namespace c10

#if defined(__ANDROID__) && __ANDROID_API__ < 21 && defined(__GLIBCXX__)
#include <cstdlib>
// std::strtoll isn't available on Android NDK platform < 21 when building
// with libstdc++, so bring the global version into std.
namespace std {
  using ::strtoll;
}
#endif
