#include <c10/util/Exception.h>
#include <c10/util/tempfile.h>
#include <fmt/format.h>

#if !defined(_WIN32)
#include <unistd.h>
#include <cerrno>
#else // defined(_WIN32)
#include <Windows.h>
#include <fcntl.h>
#include <fileapi.h>
#include <io.h>
#endif // defined(_WIN32)

// Creates the filename pattern passed to and completed by `mkstemp`.
#if !defined(_WIN32)
static std::string make_filename(std::string_view name_prefix) {
  // The filename argument to `mkstemp` needs "XXXXXX" at the end according to
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
  constexpr const char* kRandomPattern = "XXXXXX";

  // We see if any of these environment variables is set and use their value, or
  // else default the temporary directory to `/tmp`.

  const char* tmp_directory = "/tmp";
  for (const char* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* path = getenv(variable)) {
      tmp_directory = path;
      break;
    }
  }
  return fmt::format("{}/{}{}", tmp_directory, name_prefix, kRandomPattern);
}
#else
static std::string make_filename() {
  char name[L_tmpnam_s]{};
  auto res = tmpnam_s(name, L_tmpnam_s);
  if (res != 0) {
    TORCH_WARN("Error generating temporary file");
    return "";
  }
  return name;
}
#endif // !defined(_WIN32)

namespace c10 {
/// Attempts to return a temporary file or returns `nullopt` if an error
/// occurred.
std::optional<TempFile> try_make_tempfile(std::string_view name_prefix) {
#if defined(_WIN32)
  auto filename = make_filename();
#else
  auto filename = make_filename(name_prefix);
#endif
  if (filename.empty()) {
    return std::nullopt;
  }
#if defined(_WIN32)
  return TempFile(std::move(filename));
#else
  const int fd = mkstemp(filename.data());
  if (fd == -1) {
    return std::nullopt;
  }
  return TempFile(std::move(filename), fd);
#endif // defined(_WIN32)
}

/// Like `try_make_tempfile`, but throws an exception if a temporary file could
/// not be returned.
TempFile make_tempfile(std::string_view name_prefix) {
  if (auto tempfile = try_make_tempfile(name_prefix)) {
    return std::move(*tempfile);
  }
  TORCH_CHECK(false, "Error generating temporary file: ", std::strerror(errno));
}

/// Attempts to return a temporary directory or returns `nullopt` if an error
/// occurred.
std::optional<TempDir> try_make_tempdir(std::string_view name_prefix) {
#if defined(_WIN32)
  for (int i = 0; i < 10; i++) {
    auto dirname = make_filename();
    if (dirname.empty()) {
      return std::nullopt;
    }
    if (CreateDirectoryA(dirname.c_str(), nullptr)) {
      return TempDir(dirname);
    }
    if (GetLastError() == ERROR_SUCCESS) {
      return std::nullopt;
    }
  }
  return std::nullopt;
#else
  auto filename = make_filename(name_prefix);
  const char* dirname = mkdtemp(filename.data());
  if (!dirname) {
    return std::nullopt;
  }
  return TempDir(dirname);
#endif // defined(_WIN32)
}

#if defined(_WIN32)
bool TempFile::open() {
  if (fd != -1) {
    return false;
  }
  auto err = _sopen_s(
      &fd,
      name.c_str(),
      _O_CREAT | _O_TEMPORARY | _O_EXCL | _O_BINARY | _O_RDWR,
      _SH_DENYNO,
      _S_IREAD | _S_IWRITE);
  if (err != 0) {
    fd = -1;
    return false;
  }
  return true;
}
#endif

TempFile::~TempFile() {
  if (!name.empty()) {
#if !defined(_WIN32)
    if (fd >= 0) {
      unlink(name.c_str());
      close(fd);
    }
#else
    if (fd >= 0) {
      _close(fd);
    }
#endif
  }
}

TempDir::~TempDir() {
  if (!name.empty()) {
#if !defined(_WIN32)
    rmdir(name.c_str());
#else // defined(_WIN32)
    RemoveDirectoryA(name.c_str());
#endif // defined(_WIN32)
  }
}

/// Like `try_make_tempdir`, but throws an exception if a temporary directory
/// could not be returned.
TempDir make_tempdir(std::string_view name_prefix) {
  if (auto tempdir = try_make_tempdir(name_prefix)) {
    return std::move(*tempdir);
  }
#if !defined(_WIN32)
  TORCH_CHECK(
      false, "Error generating temporary directory: ", std::strerror(errno));
#else // defined(_WIN32)
  TORCH_CHECK(false, "Error generating temporary directory");
#endif // defined(_WIN32)
}
} // namespace c10
