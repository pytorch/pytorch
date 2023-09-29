#include <c10/util/StringUtil.h>

#include <cstring>
#include <string>

namespace c10 {

namespace detail {

std::string StripBasename(const std::string& full_path) {
#ifdef _WIN32
  const std::string separators("/\\");
#else
  const std::string separators("/");
#endif
  size_t pos = full_path.find_last_of(separators);
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

size_t ReplaceAll(std::string& s, c10::string_view from, c10::string_view to) {
  if (from.empty()) {
    return 0;
  }

  size_t numReplaced = 0;
  std::string::size_type last_pos = 0u;
  std::string::size_type cur_pos = 0u;
  std::string::size_type write_pos = 0u;
  const c10::string_view input(s);

  if (from.size() >= to.size()) {
    // If the replacement string is not larger than the original, we
    // can do the replacement in-place without allocating new storage.
    char* s_data = &s[0];

    while ((cur_pos = s.find(from.data(), last_pos, from.size())) !=
           std::string::npos) {
      ++numReplaced;
      // Append input between replaced sub-strings
      if (write_pos != last_pos) {
        std::copy(s_data + last_pos, s_data + cur_pos, s_data + write_pos);
      }
      write_pos += cur_pos - last_pos;
      // Append the replacement sub-string
      std::copy(to.begin(), to.end(), s_data + write_pos);
      write_pos += to.size();
      // Start search from next character after `from`
      last_pos = cur_pos + from.size();
    }

    // Append any remaining input after replaced sub-strings
    if (write_pos != last_pos) {
      std::copy(s_data + last_pos, s_data + input.size(), s_data + write_pos);
      write_pos += input.size() - last_pos;
      s.resize(write_pos);
    }
    return numReplaced;
  }

  // Otherwise, do an out-of-place replacement in a temporary buffer
  std::string buffer;

  while ((cur_pos = s.find(from.data(), last_pos, from.size())) !=
         std::string::npos) {
    ++numReplaced;
    // Append input between replaced sub-strings
    buffer.append(input.begin() + last_pos, input.begin() + cur_pos);
    // Append the replacement sub-string
    buffer.append(to.begin(), to.end());
    // Start search from next character after `from`
    last_pos = cur_pos + from.size();
  }
  if (numReplaced == 0) {
    // If nothing was replaced, don't modify the input
    return 0;
  }
  // Append any remaining input after replaced sub-strings
  buffer.append(input.begin() + last_pos, input.end());
  s = std::move(buffer);
  return numReplaced;
}

} // namespace c10
