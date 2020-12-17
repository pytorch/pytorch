#include <c10/util/StringUtil.h>
#include <c10/util/Exception.h>

#include <cstring>
#include <string>

namespace c10 {

namespace detail {

std::string StripBasename(const std::string& full_path) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

std::string ExcludeFileExtension(const std::string& file_name) {
  const char sep = '.';
  auto end_index = file_name.find_last_of(sep) == std::string::npos
      ? -1
      : file_name.find_last_of(sep);
  return file_name.substr(0, end_index);
}

} // namespace detail

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}

size_t ReplaceAll(std::string& s, const char* from, const char* to) {
  TORCH_CHECK(from && *from, "");
  TORCH_CHECK(to, "");

  size_t numReplaced = 0;
  std::string::size_type lenFrom = std::strlen(from);
  std::string::size_type lenTo = std::strlen(to);
  for (auto pos = s.find(from); pos != std::string::npos;
       pos = s.find(from, pos + lenTo)) {
    s.replace(pos, lenFrom, to);
    numReplaced++;
  }
  return numReplaced;
}

} // namespace c10
