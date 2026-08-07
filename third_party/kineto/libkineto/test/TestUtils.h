/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace libkineto::test {

// Owns a uniquely-named temporary file for a test to write a trace into. The
// file is created and its descriptor immediately closed on construction, and
// the file is removed when the owner goes out of scope, so tests never leak
// files under /tmp.
class TempTraceFile {
 public:
  // Creates a file named /tmp/<prefix>XXXXXX<suffix>, where mkstemps replaces
  // the six X characters with a unique component. suffix is preserved after
  // that component so callers can keep a .json extension; pass an empty suffix
  // for a name that ends in the random component.
  explicit TempTraceFile(std::string_view prefix, std::string_view suffix);
  ~TempTraceFile();

  TempTraceFile(const TempTraceFile&) = delete;
  TempTraceFile& operator=(const TempTraceFile&) = delete;

  TempTraceFile(TempTraceFile&& other) noexcept;
  TempTraceFile& operator=(TempTraceFile&& other) noexcept;

  [[nodiscard]] const std::string& path() const {
    return path_;
  }

  [[nodiscard]] const char* c_str() const {
    return path_.c_str();
  }

 private:
  std::string path_;
};

// Creates a self-cleaning temporary trace file; see TempTraceFile for the
// naming scheme. The returned object removes the file when it goes out of
// scope.
[[nodiscard]] TempTraceFile createTempTraceFile(
    std::string_view prefix,
    std::string_view suffix);

// Strips a leading "file://" from url, returning the bare filesystem path.
std::string logUrlToPath(const std::string& url);

// Returns the number of non-overlapping occurrences of substring in source.
size_t countSubstrings(const std::string& source, const std::string& substring);

// Asserts (via gtest) that the file at path exists and holds more than 100
// bytes. No-op on non-Linux platforms.
void checkTracefile(const char* path);

} // namespace libkineto::test
