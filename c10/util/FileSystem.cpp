#include <c10/util/FileSystem.h>

#include <sys/stat.h>
#include <cerrno>

#ifndef _WIN32
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#endif

namespace c10 {
namespace filesystem {

bool exists(const std::string& path) {
#ifdef _WIN32
  return ::_access(path.c_str(), 0) == 0;
#else
  return ::access(path.c_str(), F_OK) == 0;
#endif
}

bool create_directories(const std::string& path) {
  if (path.empty() || exists(path)) {
    return true;
  }

  // Create parent directories first
  std::string parentPath = parent_path(path);
  if (!parentPath.empty() && parentPath != path) {
    if (!create_directories(parentPath)) {
      return false;
    }
  }

  // Create the directory
#ifdef _WIN32
  int result = ::_mkdir(path.c_str());
#else
  int result = ::mkdir(path.c_str(), 0755);
#endif

  return result == 0 || errno == EEXIST;
}

std::string parent_path(const std::string& path) {
#ifdef _WIN32
  const std::string separators("/\\");
#else
  const std::string separators("/");
#endif

  size_t pos = path.find_last_of(separators);
  if (pos == std::string::npos) {
    return "";
  }

  return path.substr(0, pos);
}

} // namespace filesystem
} // namespace c10