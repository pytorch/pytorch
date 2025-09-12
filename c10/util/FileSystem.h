#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace c10 {
namespace filesystem {

// Checks if a file or directory exists
C10_API bool exists(const std::string& path);

// Creates a directory (and parent directories if needed)
C10_API bool create_directories(const std::string& path);

// Gets parent directory path from a full path
C10_API std::string parent_path(const std::string& path);

} // namespace filesystem
} // namespace c10
