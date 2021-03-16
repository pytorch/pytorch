#pragma once
#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/lexer.h>

namespace torch {
namespace jit {

inline bool isCharCount(char c, const std::string& str, size_t start, int len) {
  // count checks from [start, start + len)
  return start + len <= str.size() &&
      std::count(str.begin() + start, str.begin() + start + len, c) == len;
}

inline c10::optional<char> parseOctal(const std::string& str, size_t pos) {
  //\xxx where x are 0-7
  if (pos + 3 >= str.size())
    return c10::nullopt;
  size_t c = 0;
  for (size_t i = 1, b = 64; i < 4; ++i, b /= 8) {
    int d = str[pos + i];
    if (d < '0' || d > '7')
      return c10::nullopt;
    c += b * (d - '0');
  }
  if (c >= 256)
    return c10::nullopt;
  return c;
}

inline std::string parseStringLiteral(
    const SourceRange& range,
    const std::string& str) {
  int quote_len = isCharCount(str[0], str, 0, 3) ? 3 : 1;
  auto ret_str = str.substr(quote_len, str.size() - quote_len * 2);
  size_t pos = ret_str.find('\\');
  while (pos != std::string::npos) {
    // invariant: pos has to escape a character because it is a valid string
    char c = ret_str[pos + 1];
    size_t to_erase = 2;
    switch (ret_str[pos + 1]) {
      case '\\':
      case '\'':
      case '\"':
      case '\n':
        break;
      case 'a':
        c = '\a';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'v':
        c = '\v';
        break;
      case 't':
        c = '\t';
        break;
      case 'x':
        throw ErrorReport(range) << "unsupported hex specifier";
      case 'u':
      case 'U':
        throw ErrorReport(range) << "unsupported unicode specifier";
      default:
        // octal value in format \nnn, n is [0-7]
        if (auto v = parseOctal(ret_str, pos)) {
          to_erase = 4;
          c = *v;
        } else {
          throw ErrorReport(range) << " ill formed octal specifier";
        }
    }
    ret_str.replace(pos, to_erase, /* num copies */ 1, c);
    pos = ret_str.find('\\', pos + 1);
  }
  return ret_str;
}

} // namespace jit
} // namespace torch
