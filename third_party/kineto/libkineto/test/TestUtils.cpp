/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/TestUtils.h"

#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#endif

namespace libkineto::test {

TempTraceFile::TempTraceFile(std::string_view prefix, std::string_view suffix) {
  std::string nameTemplate;
  nameTemplate.reserve(5 + prefix.size() + 6 + suffix.size());
  nameTemplate += "/tmp/";
  nameTemplate += prefix;
  nameTemplate += "XXXXXX";
  nameTemplate += suffix;

  const int fd = mkstemps(nameTemplate.data(), static_cast<int>(suffix.size()));
  if (fd < 0) {
    throw std::runtime_error("mkstemps failed for " + nameTemplate);
  }
  close(fd);
  path_ = std::move(nameTemplate);
}

TempTraceFile::~TempTraceFile() {
  if (!path_.empty()) {
    unlink(path_.c_str());
  }
}

TempTraceFile::TempTraceFile(TempTraceFile&& other) noexcept
    : path_(std::move(other.path_)) {
  other.path_.clear();
}

TempTraceFile& TempTraceFile::operator=(TempTraceFile&& other) noexcept {
  if (this != &other) {
    if (!path_.empty()) {
      unlink(path_.c_str());
    }
    path_ = std::move(other.path_);
    other.path_.clear();
  }
  return *this;
}

TempTraceFile createTempTraceFile(
    std::string_view prefix,
    std::string_view suffix) {
  return TempTraceFile(prefix, suffix);
}

std::string logUrlToPath(const std::string& url) {
  const std::string prefix = "file://";
  if (url.starts_with(prefix)) {
    return url.substr(prefix.size());
  }
  return url;
}

size_t countSubstrings(
    const std::string& source,
    const std::string& substring) {
  if (source.empty() || substring.empty()) {
    return 0;
  }
  size_t count = 0;
  size_t pos = source.find(substring);
  while (pos != std::string::npos) {
    ++count;
    pos = source.find(substring, pos + substring.length());
  }
  return count;
}

void checkTracefile(const char* path) {
#ifdef __linux__
  // @lint-ignore NULLSAFECLANG callers always pass a non-null path
  int fd = open(path, O_RDONLY);
  ASSERT_GE(fd, 0) << "failed to open " << path;
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
  close(fd);
#endif
}

} // namespace libkineto::test
