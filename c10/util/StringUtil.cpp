#include <c10/util/StringUtil.h>

#include <string>

#ifndef _WIN32
#include <codecvt>
#include <locale>
#else
#include <c10/util/Unicode.h>
#endif

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

// Narrows the wstr argument and then passes it to _str.
// Assumes that the input (wide) text is encoded as UTF-16.
static std::ostream& _strFromWide(
    std::ostream& ss,
    const std::wstring& wString);

#ifndef _WIN32

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
// TODO (huydhn) https://en.cppreference.com/w/cpp/header/codecvt has been
// deprecated in C++17 but there is no alternative yet, so I just ack it
static std::ostream& _strFromWide(
    std::ostream& ss,
    const std::wstring& wString) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return _str(ss, converter.to_bytes(wString));
}
C10_DIAGNOSTIC_POP()

#else // #ifndef _WIN32
// The WIN32 implementation of wstring_convert leaks memory; see
// https://github.com/microsoft/STL/issues/443

static std::ostream& _strFromWide(
    std::ostream& ss,
    const std::wstring& wString) {
  return _str(ss, u16u8(wString));
}

#endif // _WIN32

std::ostream& _str(std::ostream& ss, const wchar_t* wCStr) {
  return _strFromWide(ss, std::wstring(wCStr));
}
std::ostream& _str(std::ostream& ss, const wchar_t& wChar) {
  return _strFromWide(ss, std::wstring(1, wChar));
}
std::ostream& _str(std::ostream& ss, const std::wstring& wString) {
  return _strFromWide(ss, wString);
}

} // namespace detail

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}

size_t ReplaceAll(std::string& s, std::string_view from, std::string_view to) {
  if (from.empty()) {
    return 0;
  }

  size_t numReplaced = 0;
  std::string::size_type last_pos = 0u;
  std::string::size_type cur_pos = 0u;
  std::string::size_type write_pos = 0u;
  const std::string_view input(s);

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

template <>
std::optional<int64_t> tryToNumber<int64_t>(const std::string& symbol) {
  return tryToNumber<int64_t>(symbol.c_str());
}

template <>
std::optional<int64_t> tryToNumber<int64_t>(const char* symbol) {
  // TODO Using strtoll for portability. Consider using std::from_chars in the
  // future. According to https://libcxx.llvm.org/Status/Cxx17.html,
  // std::from_chars is not supported until clang 20. We will need MSVC to also
  // fully support std::from_chars.
  if (!symbol) {
    return std::nullopt;
  }
  char* end = nullptr;
  errno = 0;
  int64_t value = strtoll(symbol, &end, 0);
  if (errno != 0) {
    errno = 0;
    return std::nullopt;
  }
  if (*end != '\0' || end == symbol) {
    return std::nullopt;
  }
  return value;
}

template <>
std::optional<double> tryToNumber<double>(const std::string& symbol) {
  return tryToNumber<double>(symbol.c_str());
}

template <>
std::optional<double> tryToNumber<double>(const char* symbol) {
  // TODO Using strtod for portability. Consider using std::from_chars in the
  // future. According to https://libcxx.llvm.org/Status/Cxx17.html,
  // std::from_chars is not supported until clang 20. We will need MSVC to also
  // fully support std::from_chars.
  if (!symbol) {
    return std::nullopt;
  }
  char* end = nullptr;
  errno = 0;
  double value = strtod(symbol, &end);
  if (errno != 0) {
    errno = 0;
    return std::nullopt;
  }
  if (*end != '\0' || end == symbol) {
    return std::nullopt;
  }
  return value;
}

std::vector<std::string_view> split(std::string_view target, char delimiter) {
  std::vector<std::string_view> atoms;
  std::string_view buffer = target;
  while (!buffer.empty()) {
    auto i = buffer.find(delimiter);
    if (i == std::string_view::npos) {
      atoms.push_back(buffer);
      buffer.remove_prefix(buffer.size());
    } else {
      atoms.push_back(buffer.substr(0, i));
      buffer.remove_prefix(i + 1);
    }
  }
  return atoms;
}
} // namespace c10
