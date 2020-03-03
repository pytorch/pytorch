#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {
namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// Convert an python index (which may be negative) into an index usable for a
// C++ container
int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

std::string stringSlice(std::string string, int64_t start, int64_t end, int64_t step) {
  TORCH_CHECK(step == 1, "Slicing a string only supports step=1");

  const int64_t size = string.size();

  // Clamp start and end to the bounds of the list
  start = std::max(int64_t(0), normalizeIndex(start, size));
  end = std::min(size, normalizeIndex(end, size));

  if (end <= start) {
    // Slice is empty
    return std::string("");
  }

  std::string result(string.begin() + start, string.begin() + end);
  return result;
}

int64_t stringFindImpl(
    std::string string,
    std::string substr,
    int64_t start,
    int64_t end,
    bool reverse = false) {
  int64_t size = string.size();
  if (start < 0) {
    start = std::max(int64_t(0), int64_t(size + start));
  }
  if (end < 0) {
    end = std::max(int64_t(0), int64_t(size + end + 1));
  }
  if (end > start) {
    string = string.substr(start, end - start);
  } else {
    string = "";
  }

  int64_t result = -1;
  if (string.size() >= substr.size()) {
    auto pos = string.find(substr, 0);
    if (reverse) {
      auto rpos = pos;
      do {
        pos = rpos;
        rpos = string.find(substr, pos + 1);
      } while (rpos != std::string::npos);
    }
    if (pos != std::string::npos) {
      result = pos + start;
    }
  }
  return result;
}

RegisterOperators reg_str_ops({

// python string is methods return false if empty
#define DEFINE_STRING_IS_OP(op_name, char_op)                          \
  Operator(                                                            \
      #op_name "(str self) -> bool",                                   \
      [](Stack& stack) {                                               \
        auto string = pop(stack).toStringRef();                        \
        push(                                                          \
            stack,                                                     \
            string.size() != 0 &&                                      \
                std::all_of(string.begin(), string.end(), [](char c) { \
                  return char_op(c);                                   \
                }));                                                   \
        return 0;                                                      \
      },                                                               \
      aliasAnalysisFromSchema())

    DEFINE_STRING_IS_OP(aten::isdigit, ::isdigit),
    DEFINE_STRING_IS_OP(aten::isspace, ::isspace),
    DEFINE_STRING_IS_OP(aten::isalnum, ::isalnum),
    DEFINE_STRING_IS_OP(aten::isalpha, ::isalpha),
    DEFINE_STRING_IS_OP(aten::isdecimal, ::isdigit),
    DEFINE_STRING_IS_OP(aten::isnumeric, ::isdigit),

#define DEFINE_STRING_CHAR_MAP_OP(op_name, char_op) \
  Operator(                                         \
      #op_name "(str self) -> str",                 \
      [](Stack& stack) {                            \
        auto string = pop(stack).toStringRef();     \
        std::stringstream ss;                       \
        for (char c : string) {                     \
          ss << static_cast<char>(char_op(c));      \
        }                                           \
        push(stack, ss.str());                      \
        return 0;                                   \
      },                                            \
      aliasAnalysisFromSchema())

    DEFINE_STRING_CHAR_MAP_OP(aten::upper, ::toupper),
    DEFINE_STRING_CHAR_MAP_OP(aten::lower, ::tolower),
    DEFINE_STRING_CHAR_MAP_OP(aten::swapcase, ([](char c) {
                                if (c == static_cast<char>(::toupper(c))) {
                                  return static_cast<char>(::tolower(c));
                                } else {
                                  return static_cast<char>(::toupper(c));
                                }
                              }))

});

auto reg_str_ops_2 =
    torch::RegisterOperators()
        .op("aten::splitlines(str self, bool keepends=False) -> str[]",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, bool keepends) {
                  std::string delimiters =
                      "\n\r\r\n\v\x0b\f\x0c\x1c\x1d\x1e\x85\u2028\u2029";
                  c10::List<std::string> splits;

                  std::string::size_type prev_pos = 0;
                  std::string::size_type pos = 0;
                  while ((pos = string.find_first_of(delimiters, pos)) !=
                         std::string::npos) {
                    splits.emplace_back(
                        string.substr(prev_pos, pos - prev_pos));
                    if (keepends) {
                      splits.emplace_back(string.substr(pos, 1));
                    }
                    pos++;
                    prev_pos = pos;
                  }
                  if (prev_pos != string.size()) {
                    splits.emplace_back(
                        string.substr(prev_pos, string.size() - prev_pos));
                  }

                  return splits;
                }))
        .op("aten::slice.str(str string, int start, int end=9223372036854775807, int step=1) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel<decltype(stringSlice), &stringSlice>())

        // upper and lower require there to be at least one alpha character,
        // and ignore all other characters
        .op("aten::isupper(str self) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  bool found_alpha = false;
                  bool is_upper = true;
                  for (size_t i = 0; i < string.size() && is_upper; ++i) {
                    char c = string[i];
                    found_alpha |= static_cast<bool>(::isalpha(c));
                    is_upper &= (!::isalpha(c) || ::isupper(c));
                  }
                  return found_alpha && is_upper;
                }))
        .op("aten::islower(str self) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  bool found_alpha = false;
                  bool is_lower = true;
                  for (size_t i = 0; i < string.size() && is_lower; ++i) {
                    char c = string[i];
                    found_alpha |= static_cast<bool>(::isalpha(c));
                    is_lower &= (!::isalpha(c) || ::islower(c));
                  }
                  return found_alpha && is_lower;
                }))

        .op("aten::capitalize(str self) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  std::stringstream ss;
                  auto first_char = true;
                  for (char c : string) {
                    if (first_char) {
                      ss << static_cast<char>(::toupper(c));
                      first_char = false;
                    } else {
                      ss << static_cast<char>(::tolower(c));
                    }
                  }
                  return ss.str();
                }))

        .op("aten::title(str self) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  std::stringstream ss;
                  bool prev_is_nonalpha = true;
                  for (char c : string) {
                    if (prev_is_nonalpha) {
                      ss << static_cast<char>(::toupper(c));
                    } else {
                      ss << static_cast<char>(::tolower(c));
                    }
                    if (::isalpha(c)) {
                      prev_is_nonalpha = false;
                    } else {
                      prev_is_nonalpha = true;
                    }
                  }
                  return ss.str();
                }))

        .op("aten::center(str self, int width, str fillchar=' ') -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   int64_t width,
                                   std::string fillchar) {
                  if (fillchar.size() != 1) {
                    // TODO: this should be a TypeError
                    throw std::runtime_error(
                        "TypeError: The fill character must be exactly one character long");
                  }
                  if (string.size() >
                      static_cast<std::string::size_type>(width)) {
                    return string;
                  }
                  std::stringstream ss;
                  std::string::size_type full_padding = width - string.size();
                  std::string::size_type l_pad = full_padding / 2;
                  std::string::size_type r_pad = (full_padding + 1) / 2;
                  if (width % 2) {
                    auto tmp = r_pad;
                    r_pad = l_pad;
                    l_pad = tmp;
                  }
                  for (std::string::size_type i = 0; i < l_pad; ++i) {
                    ss << fillchar;
                  }
                  ss << string;
                  for (std::string::size_type i = 0; i < r_pad; ++i) {
                    ss << fillchar;
                  }
                  return ss.str();
                }))

        // Adapted from
        // https://stackoverflow.com/questions/22489073/counting-the-number-of-occurrences-of-a-string-within-a-string
        .op("aten::count(str self, str substr, int start=0, int end=-1) -> int",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  int64_t size = string.size();
                  if (start > size) {
                    return int64_t(0);
                  }
                  if (start < 0) {
                    start = std::max(int64_t(0), int64_t(size + start));
                  }
                  if (end < 0) {
                    end = std::max(int64_t(0), int64_t(size + end + 1));
                  }

                  int64_t occurrences = 0;
                  std::string::size_type pos = start;
                  while ((pos = string.find(substr, pos)) !=
                         std::string::npos) {
                    if (pos < static_cast<std::string::size_type>(end)) {
                      ++occurrences;
                    } else {
                      break;
                    }
                    pos += substr.length();
                  }
                  return occurrences;
                }))

        .op("aten::endswith(str self, str substr, int start=0, int end=-1) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  int64_t size = string.size();
                  if (start < 0) {
                    start = std::max(int64_t(0), int64_t(size + start));
                  }
                  if (end < 0) {
                    end = std::max(int64_t(0), int64_t(size + end + 1));
                  }

                  string = string.substr(start, end - start);

                  auto result = false;
                  if (string.length() >= substr.length()) {
                    result = !string.compare(
                        string.length() - substr.length(),
                        substr.length(),
                        substr);
                  }
                  return result;
                }))

        .op("aten::startswith(str self, str substr, int start=0, int end=-1) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  int64_t size = string.size();
                  if (start < 0) {
                    start = std::max(int64_t(0), int64_t(size + start));
                  }
                  if (end < 0) {
                    end = std::max(int64_t(0), int64_t(size + end + 1));
                  }

                  string = string.substr(start, end - start);

                  auto result = false;
                  if (string.length() >= substr.length()) {
                    result = !string.compare(0, substr.length(), substr);
                  }
                  return result;
                }))

        .op("aten::expandtabs(str self, int tabsize=8) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, int64_t tabsize) {
                  std::stringstream ss;
                  size_t index = 0;
                  for (const auto& c : string) {
                    if (c != '\t') {
                      ss << c;
                      index++;
                    } else {
                      if (tabsize <= 0) {
                        continue;
                      }
                      do {
                        ss << ' ';
                        index++;
                      } while (index % tabsize);
                    }
                  }
                  return ss.str();
                }))

        .op("aten::find(str self, str substr, int start=0, int end=-1) -> int",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  return stringFindImpl(string, substr, start, end);
                }))

        .op("aten::rfind(str self, str substr, int start=0, int end=-1) -> int",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  return stringFindImpl(string, substr, start, end, true);
                }))

        .op("aten::index.str(str self, str substr, int start=0, int end=-1) -> int",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  auto result = stringFindImpl(string, substr, start, end);
                  if (result < 0) {
                    throw std::runtime_error("ValueError: substring not found");
                  }
                  return result;
                }))

        .op("aten::rindex(str self, str substr, int start=0, int end=-1) -> int",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string substr,
                                   int64_t start,
                                   int64_t end) {
                  auto result =
                      stringFindImpl(string, substr, start, end, true);
                  if (result < 0) {
                    throw std::runtime_error("ValueError: substring not found");
                  }
                  return result;
                }))

        .op("aten::isidentifier(str self) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  LOG(WARNING)
                      << "The isidentifier() implementation being used is from Python 2\n";
                  if (string.size() < 1) {
                    return false;
                  }
                  if (::isdigit(string[0])) {
                    return false;
                  }
                  auto result =
                      std::all_of(string.begin(), string.end(), [](char c) {
                        return ::isalnum(c);
                      });
                  return result;
                }))

        .op("aten::istitle(str self) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  auto result = false;

                  bool prev_is_alpha = false;
                  for (char c : string) {
                    if (prev_is_alpha) {
                      if (c != static_cast<char>(::tolower(c))) {
                        result = false;
                        break;
                      }
                    } else {
                      if (c != static_cast<char>(::toupper(c))) {
                        result = false;
                        break;
                      }
                      // Only true if there exists at least one alpha
                      if (::isalpha(c)) {
                        result = true;
                      }
                    }
                    if (::isalpha(c)) {
                      prev_is_alpha = true;
                    } else {
                      prev_is_alpha = false;
                    }
                  }
                  return result;
                }))

        // Can't reuse DEFINE_STRING_IS_OP because "" is printable
        .op("aten::isprintable(str self) -> bool",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string) {
                  auto result =
                      std::all_of(string.begin(), string.end(), [](char c) {
                        return ::isalnum(c) || ::ispunct(c) || c == ' ';
                      });
                  return result;
                }))

        .op("aten::ljust(str self, int width, str fillchar=' ') -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   int64_t width,
                                   std::string fillchar) {
                  if (fillchar.size() != 1) {
                    // TODO: this should be a TypeError
                    throw std::runtime_error(
                        "TypeError: The fill character must be exactly one character long");
                  }
                  auto to_append = std::max(
                      int64_t(0), width - static_cast<int64_t>(string.size()));

                  std::stringstream ss;
                  ss << string;
                  for (auto i = 0; i < to_append; ++i) {
                    ss << fillchar;
                  }

                  return ss.str();
                }))

        .op("aten::rjust(str self, int width, str fillchar=' ') -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   int64_t width,
                                   std::string fillchar) {
                  if (fillchar.size() != 1) {
                    // TODO: this should be a TypeError
                    throw std::runtime_error(
                        "TypeError: The fill character must be exactly one character long");
                  }
                  auto to_append = std::max(
                      int64_t(0), width - static_cast<int64_t>(string.size()));

                  std::stringstream ss;
                  for (auto i = 0; i < to_append; ++i) {
                    ss << fillchar;
                  }
                  ss << string;
                  return ss.str();
                }))

        .op("aten::zfill(str self, int width) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, int64_t width) {
                  auto to_append = std::max(
                      int64_t(0), width - static_cast<int64_t>(string.size()));

                  std::stringstream ss;
                  for (auto i = 0; i < to_append; ++i) {
                    ss << '0';
                  }
                  ss << string;

                  return ss.str();
                }))

        .op("aten::lstrip(str self, str chars=' \\n\\t\\f\\v') -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, std::string chars) {
                  auto index = string.find_first_not_of(chars);
                  if (index != std::string::npos) {
                    string = string.substr(index, string.size());
                  } else {
                    string = "";
                  }
                  return string;
                }))

        .op("aten::rstrip(str self, str chars=' \\n\\t\\f\\v') -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, std::string chars) {
                  auto index = string.find_last_not_of(chars);
                  if (index != std::string::npos) {
                    string = string.substr(0, index + 1);
                  } else {
                    string = "";
                  }
                  return string;
                }))

        .op("aten::strip(str self, str chars=' \\n\\t\\f\\v') -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, std::string chars) {
                  auto rindex = string.find_last_not_of(chars);
                  if (rindex != std::string::npos) {
                    string = string.substr(0, rindex + 1);
                  } else {
                    string = "";
                  }
                  auto lindex = string.find_first_not_of(chars);
                  if (lindex != std::string::npos) {
                    string = string.substr(lindex, string.size());
                  } else {
                    string = "";
                  }
                  return string;
                }))

        .op("aten::replace(str self, str old, str new, int max=-1) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string,
                                   std::string old_str,
                                   std::string new_str,
                                   int64_t max) {
                  int64_t occurrences = 0;
                  std::string::size_type pos = 0;
                  while ((pos = string.find(old_str, pos)) !=
                         std::string::npos) {
                    if (max >= 0 && ++occurrences > max) {
                      break;
                    }
                    string = string.replace(pos, old_str.length(), new_str);
                    pos += new_str.length();
                  }

                  return string;
                }))

        .op("aten::partition(str self, str separator) -> (str, str, str)",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, std::string separator) {
                  auto pos = string.find(separator, 0);
                  if (pos == std::string::npos) {
                    pos = string.size();
                    separator = "";
                  }
                  auto pre_partition = string.substr(0, pos);
                  auto post_partition =
                      string.substr(pos + separator.size(), string.size());

                  return std::make_tuple(
                      pre_partition, separator, post_partition);
                }))

        .op("aten::rpartition(str self, str separator) -> (str, str, str)",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](std::string string, std::string separator) {
                  auto pos = string.find(separator, 0);
                  auto rpos = pos;
                  do {
                    pos = rpos;
                    rpos = string.find(separator, pos + 1);
                  } while (rpos != std::string::npos);

                  if (pos == std::string::npos) {
                    pos = 0;
                    separator = "";
                  }

                  auto pre_partition = string.substr(0, pos);
                  auto post_partition =
                      string.substr(pos + separator.size(), string.size());

                  return std::make_tuple(
                      pre_partition, separator, post_partition);
                }))

        .op("aten::split.str(str self, str separator=' ', int max=-1) -> str[]",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel(
                    [](std::string string, std::string separator, int64_t max) {
                      std::string::size_type prev_pos = 0;
                      std::string::size_type pos = 0;
                      c10::List<std::string> splits;
                      auto count = 0;
                      while ((pos = string.find(separator, pos)) !=
                             std::string::npos) {
                        count++;
                        if (max >= 0 && count > max) {
                          break;
                        } else {
                          splits.emplace_back(
                              string.substr(prev_pos, pos - prev_pos));
                        }
                        pos += separator.size();
                        prev_pos = pos;
                      }
                      splits.emplace_back(
                          string.substr(prev_pos, string.size() - prev_pos));
                      return splits;
                    }))

        .op("aten::rsplit(str self, str separator=' ', int max=-1) -> str[]",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel(
                    [](std::string string, std::string separator, int64_t max) {
                      std::reverse(separator.begin(), separator.end());
                      std::reverse(string.begin(), string.end());

                      std::string::size_type prev_pos = 0;
                      std::string::size_type pos = 0;
                      c10::List<std::string> splits;
                      auto count = 0;
                      while ((pos = string.find(separator, pos)) !=
                             std::string::npos) {
                        count++;
                        if (max >= 0 && count > max) {
                          break;
                        } else {
                          auto substr = string.substr(prev_pos, pos - prev_pos);
                          std::reverse(substr.begin(), substr.end());
                          splits.emplace(splits.begin(), substr);
                        }
                        pos += separator.size();
                        prev_pos = pos;
                      }
                      auto substr =
                          string.substr(prev_pos, string.size() - prev_pos);
                      std::reverse(substr.begin(), substr.end());
                      splits.emplace(splits.begin(), substr);
                      return splits;
                    }))

        .op("aten::join(str self, str[] values) -> str",
            torch::RegisterOperators::options()
                .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
                .catchAllKernel([](const std::string& string,
                                   const c10::List<std::string>& values) {
                  std::stringstream ss;
                  for (auto it = values.begin(); it != values.end(); ++it) {
                    ss << static_cast<std::string>(*it);
                    if (it != values.end() - 1) {
                      ss << string;
                    }
                  }
                  return ss.str();
                }));
} // namespace
} // namespace jit
} // namespace torch
