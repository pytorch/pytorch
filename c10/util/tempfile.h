#pragma once

#include <c10/macros/Export.h>
#include <optional>
#include <string>
#include <utility>

namespace c10 {
struct C10_API TempFile {
  TempFile(std::string_view name, int fd = -1) noexcept : fd(fd), name(name) {}
  TempFile(const TempFile&) = delete;
  TempFile(TempFile&& other) noexcept
      : fd(other.fd), name(std::move(other.name)) {
    other.fd = -1;
  }

  TempFile& operator=(const TempFile&) = delete;
  TempFile& operator=(TempFile&& other) noexcept {
    fd = other.fd;
    name = std::move(other.name);
    other.fd = -1;
    return *this;
  }
#if defined(_WIN32)
  bool open();
#endif

  ~TempFile();

  int fd;

  std::string name;
};

struct C10_API TempDir {
  TempDir() = delete;
  explicit TempDir(std::string_view name) noexcept : name(name) {}
  TempDir(const TempDir&) = delete;
  TempDir(TempDir&& other) noexcept : name(std::move(other.name)) {
    other.name.clear();
  }

  TempDir& operator=(const TempDir&) = delete;
  TempDir& operator=(TempDir&& other) noexcept {
    name = std::move(other.name);
    return *this;
  }

  ~TempDir();

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
/// On Windows, `name_prefix` is ignored and `tmpnam_s` is used,
/// and no temporary file is opened.
C10_API std::optional<TempFile> try_make_tempfile(
    std::string_view name_prefix = "torch-file-");

/// Like `try_make_tempfile`, but throws an exception if a temporary file could
/// not be returned.
C10_API TempFile make_tempfile(std::string_view name_prefix = "torch-file-");

/// Attempts to return a temporary directory or returns `nullopt` if an error
/// occurred.
///
/// The directory returned follows the pattern
/// `<tmp-dir>/<name-prefix><random-pattern>/`, where `<tmp-dir>` is the value
/// of the `"TMPDIR"`, `"TMP"`, `"TEMP"` or
/// `"TEMPDIR"` environment variable if any is set, or otherwise `/tmp`;
/// `<name-prefix>` is the value supplied to this function, and
/// `<random-pattern>` is a random sequence of numbers.
/// On Windows, `name_prefix` is ignored.
C10_API std::optional<TempDir> try_make_tempdir(
    std::string_view name_prefix = "torch-dir-");

/// Like `try_make_tempdir`, but throws an exception if a temporary directory
/// could not be returned.
C10_API TempDir make_tempdir(std::string_view name_prefix = "torch-dir-");
} // namespace c10
