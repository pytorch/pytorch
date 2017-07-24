/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <sstream>
#include <vector>

namespace gloo {

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <>
inline void MakeStringInternal(
    std::stringstream& ss,
    const std::stringstream& t) {
  ss << t.str();
}

template <typename T, typename... Args>
inline void
MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

template <typename T>
std::string MakeString(const std::vector<T>& v, const std::string& delim=" ") {
  std::stringstream ss;
  for (auto it = v.begin(); it < v.end(); it++) {
    if (it != v.begin()) {
      MakeStringInternal(ss, delim);
    }
    MakeStringInternal(ss, *it);
  }
  return std::string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string& str) {
  return str;
}
inline std::string MakeString(const char* cstr) {
  return std::string(cstr);
}

} // namespace gloo
