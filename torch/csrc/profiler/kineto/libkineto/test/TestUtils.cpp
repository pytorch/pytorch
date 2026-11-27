/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/TestUtils.h"
#include "src/ThrowUtil.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

// Both platforms take the open flags from fcntl.h and the permission bits from
// sys/stat.h. What differs is where the call itself lives: _sopen_s and its
// share mode come from io.h and share.h, POSIX open and close from unistd.h.
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

namespace libkineto::test {

namespace {

// Creates path only if nothing is there yet, which is the property mkstemps
// provides for the name it settles on. Returns false when the name is already
// taken so the caller can pick another, and swallows every other failure the
// same way: the caller reports after exhausting its attempts.
bool createExclusive(const std::string& path) {
#ifdef _WIN32
  int fd = -1;
  const errno_t err = _sopen_s(
      &fd,
      path.c_str(),
      _O_CREAT | _O_EXCL | _O_RDWR,
      _SH_DENYNO,
      _S_IREAD | _S_IWRITE);
  if (err != 0) {
    return false;
  }
  _close(fd);
#else
  const int fd =
      ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    return false;
  }
  ::close(fd);
#endif
  return true;
}

// Stands in for the six X characters mkstemps replaces. Uniqueness does not
// rest on this being unpredictable: exclusive creation is what guarantees the
// caller owns the file, and this only has to keep collisions rare.
std::string randomNameComponent() {
  static constexpr std::string_view kChars =
      "abcdefghijklmnopqrstuvwxyz0123456789";
  static thread_local std::mt19937 engine{std::random_device{}()};
  std::uniform_int_distribution<size_t> pick(0, kChars.size() - 1);

  std::string component(6, '\0');
  for (char& c : component) {
    c = kChars[pick(engine)];
  }
  return component;
}

void removeQuietly(const std::string& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

} // namespace

TempTraceFile::TempTraceFile(std::string_view prefix, std::string_view suffix) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path();

  // Bounded so that a temp directory we cannot write to reports that instead
  // of spinning. Reaching the limit on collisions alone is not realistic.
  constexpr int kMaxAttempts = 100;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    std::string name;
    name.reserve(prefix.size() + 6 + suffix.size());
    name += prefix;
    name += randomNameComponent();
    name += suffix;

    // generic_string keeps forward slashes on Windows, which its file APIs
    // accept. Callers hand this path to Kineto as a log file and read it back
    // out of trace JSON, and a separator that needs no escaping travels
    // through both unchanged.
    const std::string candidate = (dir / name).generic_string();
    if (createExclusive(candidate)) {
      path_ = candidate;
      return;
    }
  }

  KINETO_THROW(
      std::runtime_error,
      "could not create a temporary trace file under " + dir.generic_string());
}

TempTraceFile::~TempTraceFile() {
  if (!path_.empty()) {
    removeQuietly(path_);
  }
}

TempTraceFile::TempTraceFile(TempTraceFile&& other) noexcept
    : path_(std::move(other.path_)) {
  other.path_.clear();
}

TempTraceFile& TempTraceFile::operator=(TempTraceFile&& other) noexcept {
  if (this != &other) {
    if (!path_.empty()) {
      removeQuietly(path_);
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

std::chrono::system_clock::time_point timePointFromNs(int64_t ns) {
  return std::chrono::system_clock::time_point(
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds(ns)));
}

} // namespace libkineto::test
