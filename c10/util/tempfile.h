#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#else // defined(_WIN32)
#include <Windows.h>
#include <fileapi.h>
#endif // defined(_WIN32)

namespace c10 {
namespace detail {
// Creates the filename pattern passed to and completed by `mkstemp`.
// Returns std::vector<char> because `mkstemp` needs a (non-const) `char*` and
// `std::string` only provides `const char*` before C++17.
#if !defined(_WIN32)
inline std::vector<char> make_filename(std::string name_prefix) {
  // The filename argument to `mkstemp` needs "XXXXXX" at the end according to
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
  static const std::string kRandomPattern = "XXXXXX";

  // We see if any of these environment variables is set and use their value, or
  // else default the temporary directory to `/tmp`.
  static const char* env_variables[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};

  std::string tmp_directory = "/tmp";
  for (const char* variable : env_variables) {
    if (const char* path = getenv(variable)) {
      tmp_directory = path;
      break;
    }
  }

  std::vector<char> filename;
  filename.reserve(
      tmp_directory.size() + name_prefix.size() + kRandomPattern.size() + 2);

  filename.insert(filename.end(), tmp_directory.begin(), tmp_directory.end());
  filename.push_back('/');
  filename.insert(filename.end(), name_prefix.begin(), name_prefix.end());
  filename.insert(filename.end(), kRandomPattern.begin(), kRandomPattern.end());
  filename.push_back('\0');

  return filename;
}
#endif // !defined(_WIN32)
} // namespace detail

struct TempFile {
#if !defined(_WIN32)
  TempFile() : fd(-1) {}
  TempFile(std::string name, int fd) : fd(fd), name(std::move(name)) {}
  TempFile(const TempFile&) = delete;
  TempFile(TempFile&& other) noexcept
      : fd(other.fd), name(std::move(other.name)) {
    other.fd = -1;
    other.name.clear();
  }

  TempFile& operator=(const TempFile&) = delete;
  TempFile& operator=(TempFile&& other) noexcept {
    fd = other.fd;
    name = std::move(other.name);
    other.fd = -1;
    other.name.clear();
    return *this;
  }

  ~TempFile() {
    if (fd >= 0) {
      unlink(name.c_str());
      close(fd);
    }
  }

  int fd;
#endif // !defined(_WIN32)

  std::string name;
};

struct TempDir {
  TempDir() = default;
  explicit TempDir(std::string name) : name(std::move(name)) {}
  TempDir(const TempDir&) = delete;
  TempDir(TempDir&& other) noexcept : name(std::move(other.name)) {
    other.name.clear();
  }

  TempDir& operator=(const TempDir&) = delete;
  TempDir& operator=(TempDir&& other) noexcept {
    name = std::move(other.name);
    other.name.clear();
    return *this;
  }

  ~TempDir() {
    if (!name.empty()) {
#if !defined(_WIN32)
      rmdir(name.c_str());
#else // defined(_WIN32)
      RemoveDirectoryA(name.c_str());
#endif // defined(_WIN32)
    }
  }

  std::string name;
};

/// Attempts to return a temporary file or returns `nullopt` if an error
/// occurred.
///
/// The file returned follows the pattern
/// `<tmp-dir>/<name-prefix><random-pattern>`, where `<tmp-dir>` is the value of
/// the `"TMPDIR"`, `"TMP"`, `"TEMP"` or
/// `"TEMPDIR"` environment variable if any is set, or otherwise `/tmp`;
/// `<name-prefix>` is the value supplied to this function, and
/// `<random-pattern>` is a random sequence of numbers.
/// On Windows, `name_prefix` is ignored and `tmpnam` is used.
inline c10::optional<TempFile> try_make_tempfile(
    std::string name_prefix = "torch-file-") {
#if defined(_WIN32)
  return TempFile{std::tmpnam(nullptr)};
#else
  std::vector<char> filename = detail::make_filename(std::move(name_prefix));
  const int fd = mkstemp(filename.data());
  if (fd == -1) {
    return c10::nullopt;
  }
  // Don't make the string from string(filename.begin(), filename.end(), or
  // there will be a trailing '\0' at the end.
  return TempFile(filename.data(), fd);
#endif // defined(_WIN32)
}

/// Like `try_make_tempfile`, but throws an exception if a temporary file could
/// not be returned.
inline TempFile make_tempfile(std::string name_prefix = "torch-file-") {
  if (auto tempfile = try_make_tempfile(std::move(name_prefix))) {
    return std::move(*tempfile);
  }
  TORCH_CHECK(false, "Error generating temporary file: ", std::strerror(errno));
}

/// Attempts to return a temporary directory or returns `nullopt` if an error
/// occurred.
///
/// The directory returned follows the pattern
/// `<tmp-dir>/<name-prefix><random-pattern>/`, where `<tmp-dir>` is the value
/// of the `"TMPDIR"`, `"TMP"`, `"TEMP"` or
/// `"TEMPDIR"` environment variable if any is set, or otherwise `/tmp`;
/// `<name-prefix>` is the value supplied to this function, and
/// `<random-pattern>` is a random sequence of numbers.
/// On Windows, `name_prefix` is ignored and `tmpnam` is used.
inline c10::optional<TempDir> try_make_tempdir(
    std::string name_prefix = "torch-dir-") {
#if defined(_WIN32)
  while (true) {
    const char* dirname = std::tmpnam(nullptr);
    if (!dirname) {
      return c10::nullopt;
    }
    if (CreateDirectoryA(dirname, NULL)) {
      return TempDir(dirname);
    }
    if (GetLastError() != ERROR_ALREADY_EXISTS) {
      return c10::nullopt;
    }
  }
  return c10::nullopt;
#else
  std::vector<char> filename = detail::make_filename(std::move(name_prefix));
  const char* dirname = mkdtemp(filename.data());
  if (!dirname) {
    return c10::nullopt;
  }
  return TempDir(dirname);
#endif // defined(_WIN32)
}

/// Like `try_make_tempdir`, but throws an exception if a temporary directory
/// could not be returned.
inline TempDir make_tempdir(std::string name_prefix = "torch-dir-") {
  if (auto tempdir = try_make_tempdir(std::move(name_prefix))) {
    return std::move(*tempdir);
  }
  TORCH_CHECK(
      false, "Error generating temporary directory: ", std::strerror(errno));
}
} // namespace c10
