//==-- llvm/Support/FileCheck.h ---------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// API modified from llvm::FileCheck

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <optional>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace torch::jit::testing {

enum CheckType {
  CHECK,
  CHECK_NEXT,
  CHECK_SAME,
  CHECK_NOT,
  CHECK_COUNT,
  CHECK_DAG,
  CHECK_SOURCE_HIGHLIGHTED,
  CHECK_REGEX,
};

struct Check {
  Check(
      CheckType type,
      std::string str,
      std::optional<size_t> count = std::nullopt)
      : type_(type), count_(count), search_str_(std::move(str)) {}

  Check(
      CheckType type,
      std::string_view str,
      std::optional<size_t> count = std::nullopt)
      : Check(type, std::string(str.begin(), str.end()), count) {}

  CheckType type_;
  std::optional<size_t> count_;
  const std::string search_str_;

  friend std::ostream& operator<<(std::ostream& out, const Check& c);
};

std::ostream& operator<<(std::ostream& out, const Check& c) {
  switch (c.type_) {
    case CHECK:
      out << "CHECK";
      break;
    case CHECK_NEXT:
      out << "CHECK-NEXT";
      break;
    case CHECK_SAME:
      out << "CHECK-SAME";
      break;
    case CHECK_NOT:
      out << "CHECK-NOT";
      break;
    case CHECK_DAG:
      out << "CHECK-DAG";
      break;
    case CHECK_COUNT:
      out << "CHECK-COUNT-" << *c.count_;
      break;
    case CHECK_SOURCE_HIGHLIGHTED:
      out << "CHECK-SOURCE-HIGHLIGHTED";
      break;
    case CHECK_REGEX:
      out << "CHECK-REGEX";
      break;
  }
  out << ": " << c.search_str_;
  return out;
}

namespace {

using VariableMap = std::unordered_map<std::string, std::string>;

struct PatternMatch {
  size_t start;
  size_t end;
  VariableMap captures;
};

bool isValidVariableName(std::string_view name) {
  if (name.empty()) {
    return false;
  }
  size_t start = name[0] == '$' ? 1 : 0;
  if (start == name.size() ||
      !(std::isalpha(static_cast<unsigned char>(name[start])) ||
        name[start] == '_')) {
    return false;
  }
  return std::all_of(
      name.begin() + static_cast<std::string_view::difference_type>(start + 1),
      name.end(),
      [](unsigned char c) { return std::isalnum(c) || c == '_'; });
}

std::string escapeRegex(std::string_view str) {
  std::string escaped;
  escaped.reserve(str.size());
  for (const char c : str) {
    switch (c) {
      case '\\':
      case '^':
      case '$':
      case '.':
      case '|':
      case '?':
      case '*':
      case '+':
      case '(':
      case ')':
      case '[':
      case ']':
      case '{':
      case '}':
        escaped.push_back('\\');
        break;
      default:
        break;
    }
    escaped.push_back(c);
  }
  return escaped;
}

size_t countCapturingGroups(std::string_view regex) {
  size_t count = 0;
  bool escaped = false;
  bool in_character_class = false;
  for (size_t i = 0; i < regex.size(); ++i) {
    const char c = regex[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '[') {
      in_character_class = true;
      continue;
    }
    if (c == ']' && in_character_class) {
      in_character_class = false;
      continue;
    }
    if (c == '(' && !in_character_class &&
        (i + 1 == regex.size() || regex[i + 1] != '?')) {
      ++count;
    }
  }
  return count;
}

std::optional<PatternMatch> findPattern(
    const SourceRange& search_range,
    const std::string& pattern,
    const VariableMap& variables,
    bool allow_definitions = true) {
  const auto& text = search_range.source()->text_str();
  const size_t search_end = std::min(search_range.end(), text.size());
  if (search_range.start() > search_end) {
    return std::nullopt;
  }

  const std::string_view pattern_view(pattern);
  if (pattern_view.find("[[") == std::string_view::npos) {
    const auto pos = text.find(pattern, search_range.start());
    if (pos == std::string::npos || pos + pattern.size() > search_end) {
      return std::nullopt;
    }
    return PatternMatch{pos, pos + pattern.size(), {}};
  }

  std::string regex;
  std::unordered_map<std::string, size_t> local_definitions;
  std::vector<std::pair<std::string, size_t>> captures;
  size_t capturing_groups = 0;
  size_t cursor = 0;
  while (cursor < pattern_view.size()) {
    const auto open = pattern_view.find("[[", cursor);
    if (open == std::string_view::npos) {
      regex += escapeRegex(pattern_view.substr(cursor));
      break;
    }
    regex += escapeRegex(pattern_view.substr(cursor, open - cursor));
    const auto close = pattern_view.find("]]", open + 2);
    if (close == std::string_view::npos) {
      throw std::runtime_error(c10::str(
          "Unterminated FileCheck variable block in pattern: ", pattern));
    }

    const auto block = pattern_view.substr(open + 2, close - open - 2);
    const auto colon = block.find(':');
    const auto name_view = block.substr(0, colon);
    if (!isValidVariableName(name_view)) {
      throw std::runtime_error(
          c10::str("Invalid FileCheck variable name: ", name_view));
    }
    const std::string name(name_view);

    if (colon != std::string_view::npos) {
      if (!allow_definitions) {
        throw std::runtime_error(
            "FileCheck variable definitions are not allowed in CHECK-NOT");
      }
      const auto definition = block.substr(colon + 1);
      const size_t capture_group = capturing_groups + 1;
      regex += c10::str("(", definition, ")");
      capturing_groups += 1 + countCapturingGroups(definition);
      local_definitions[name] = capture_group;
      captures.emplace_back(name, capture_group);
    } else if (const auto local = local_definitions.find(name);
               local != local_definitions.end()) {
      regex += c10::str("(?:\\", local->second, ")");
    } else if (const auto variable = variables.find(name);
               variable != variables.end()) {
      regex += escapeRegex(variable->second);
    } else {
      throw std::runtime_error(
          c10::str("Undefined FileCheck variable: ", name));
    }
    cursor = close + 2;
  }

  const auto target =
      text.substr(search_range.start(), search_end - search_range.start())
          .str();
  std::smatch match;
  if (!std::regex_search(target, match, std::regex(regex))) {
    return std::nullopt;
  }

  VariableMap captured_variables;
  for (const auto& [name, group] : captures) {
    captured_variables[name] = match[group].str();
  }
  const size_t start = search_range.start() + match.position(0);
  return PatternMatch{start, start + match.length(0), captured_variables};
}

PatternMatch assertFindPattern(
    const SourceRange& search_range,
    const Check& check,
    const VariableMap& variables) {
  auto match = findPattern(search_range, check.search_str_, variables);
  if (!match) {
    auto found_range = SourceRange(
        search_range.source(), search_range.start(), search_range.end());
    std::stringstream ss;
    ss << "Expected to find pattern ";
    c10::printQuotedString(ss, check.search_str_);
    ss << " but did not find it\n";
    ss << "Searched string:\n";
    found_range.highlight(ss);
    ss << "From " << check << '\n';
    throw std::runtime_error(std::move(ss).str());
  }
  return std::move(*match);
}

PatternMatch assertFindPattern(
    const std::shared_ptr<Source>& source,
    const Check& check,
    size_t start,
    const VariableMap& variables) {
  return assertFindPattern(
      SourceRange(source, start, source->size()), check, variables);
}

void updateVariables(VariableMap& variables, const PatternMatch& match) {
  for (const auto& [name, value] : match.captures) {
    variables[name] = value;
  }
}

size_t assertFind(
    const SourceRange& search_range,
    const std::string& sub,
    const std::function<void(std::ostream& out)>& extra_msg = nullptr) {
  auto pos = search_range.source()->text_str().find(sub, search_range.start());
  if (pos == std::string::npos || (pos + sub.size()) > search_range.end()) {
    auto found_range =
        SourceRange(search_range.source(), search_range.start(), sub.size());
    std::stringstream ss;
    ss << "Expected to find ";
    c10::printQuotedString(ss, sub);
    ss << " but did not find it" << '\n';
    ss << "Searched string:" << '\n';
    found_range.highlight(ss);
    if (extra_msg) {
      extra_msg(ss);
    }
    throw std::runtime_error(std::move(ss).str());
  }
  return pos;
}

size_t assertFind(
    const SourceRange& search_range,
    const std::string& sub,
    const Check& check) {
  return assertFind(search_range, sub, [&](std::ostream& out) {
    out << "From " << check << '\n';
  });
}

size_t assertFind(
    const std::shared_ptr<Source>& source,
    const std::string& sub,
    size_t start,
    const Check& check) {
  return assertFind(SourceRange(source, start, source->size()), sub, check);
}

size_t assertFindRegex(
    const SourceRange& search_range,
    const std::string& sub,
    const std::function<void(std::ostream& out)>& extra_msg = nullptr) {
  auto pos =
      search_range.source()->text_str().find_regex(sub, search_range.start());

  if (pos == std::string::npos) {
    std::stringstream ss;
    ss << "Expected to find regex ";
    c10::printQuotedString(ss, sub);
    ss << " but did not find it" << '\n';
    ss << "Searched string:" << '\n';
    if (extra_msg) {
      extra_msg(ss);
    }
    throw std::runtime_error(std::move(ss).str());
  }
  return pos;
}

size_t assertFindRegex(
    const SourceRange& search_range,
    const std::string& sub,
    const Check& check) {
  return assertFindRegex(search_range, sub, [&](std::ostream& out) {
    out << "From " << check << '\n';
  });
}

size_t assertFindRegex(
    const std::shared_ptr<Source>& source,
    const std::string& sub,
    size_t start,
    const Check& check) {
  return assertFindRegex(
      SourceRange(source, start, source->size()), sub, check);
}

void assertNotFind(
    const SourceRange& search_range,
    const std::string& sub,
    const Check& check) {
  auto pos = search_range.source()->text_str().find(sub, search_range.start());
  if (pos != std::string::npos && (pos + sub.size()) <= search_range.end()) {
    auto found_range =
        SourceRange(search_range.source(), pos, sub.size() + pos);
    std::stringstream ss;
    ss << "Expected to not find ";
    c10::printQuotedString(ss, sub);
    ss << " but found it\n";
    found_range.highlight(ss);
    ss << "From " << check << '\n';
    throw std::runtime_error(std::move(ss).str());
  }
}

void assertNotFindPattern(
    const SourceRange& search_range,
    const Check& check,
    const VariableMap& variables) {
  const auto match =
      findPattern(search_range, check.search_str_, variables, false);
  if (match) {
    auto found_range =
        SourceRange(search_range.source(), match->start, match->end);
    std::stringstream ss;
    ss << "Expected to not find pattern ";
    c10::printQuotedString(ss, check.search_str_);
    ss << " but found it\n";
    found_range.highlight(ss);
    ss << "From " << check << '\n';
    throw std::runtime_error(std::move(ss).str());
  }
}

} // namespace

struct FileCheckImpl {
  TORCH_API explicit FileCheckImpl() = default;

  TORCH_API void run(const std::string& test_file) {
    has_run = true;

    if (groups.empty() || groups[0].empty()) {
      throw std::runtime_error(
          "No checks have been added to this instance of"
          "Filecheck! Check for bad input.");
    }

    doChecks(std::make_shared<Source>(test_file));
  }

  TORCH_API void run(
      const std::string& checks_file,
      const std::string& test_file) {
    auto source = std::make_shared<Source>(checks_file);
    parseStrings(source);
    run(test_file);
  }

  TORCH_API void addCheck(const Check& check) {
    // consecutive CHECK_DAGs & CHECK_NOTs need to be evaluated as a group
    if (groups.empty() ||
        (check.type_ != CHECK_NOT && check.type_ != CHECK_DAG)) {
      groups.push_back({check});
    } else {
      auto& last_group = groups.back();
      if (last_group.at(0).type_ == check.type_) {
        last_group.push_back(check);
      } else {
        groups.push_back({check});
      }
    }
    checks.push_back(check);
    has_run = false;
  }

  TORCH_API void addCheck(
      CheckType type,
      const std::string& s,
      std::optional<size_t> count = std::nullopt) {
    addCheck(Check(type, s, count));
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool has_run = false;

  friend std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc);

 private:
  bool parseSingleCheck(const std::shared_ptr<Source>& source, size_t* start) {
    const static std::vector<std::pair<CheckType, std::string>> check_pairs = {
        {CHECK, ": "},
        {CHECK_NEXT, "-NEXT: "},
        {CHECK_SAME, "-SAME: "},
        {CHECK_NOT, "-NOT: "},
        {CHECK_DAG, "-DAG: "},
        {CHECK_COUNT, "-COUNT-"}, // needs special parsing
        {CHECK_SOURCE_HIGHLIGHTED, "-SOURCE-HIGHLIGHTED: "},
        {CHECK_REGEX, "-REGEX: "},
    };

    for (const auto& check_pair : check_pairs) {
      const std::string& check_suffix = check_pair.second;
      auto suffix_pos = source->text_str().find(check_suffix, *start);
      if (suffix_pos != *start) {
        continue;
      }
      size_t end_check_string = suffix_pos + check_suffix.size();
      CheckType type = check_pair.first;
      std::optional<size_t> count = std::nullopt;
      auto end_line = source->text_str().find("\n", end_check_string);
      bool exactly = false;
      if (type == CHECK_COUNT) {
        const std::string exact = "EXACTLY-";
        if (source->text_str().find(exact, end_check_string) ==
            end_check_string) {
          exactly = true;
          end_check_string += exact.size();
        }
        size_t end =
            assertFind(SourceRange(source, end_check_string, end_line), ":");
        auto count_view = source->text_str()
                              .substr(end_check_string, end - end_check_string)
                              .str();
        count = std::stoll(std::string(count_view.begin(), count_view.end()));
        end_check_string = end + 2; // add ':' and the space
      }
      auto check = Check(
          type,
          source->text_str()
              .substr(end_check_string, end_line - end_check_string)
              .str(),
          count);
      addCheck(check);
      if (exactly) {
        addCheck(CHECK_NOT, check.search_str_);
      }
      *start = end_line;
      return true;
    }
    return false;
  }

  size_t findNextStart(const std::shared_ptr<Source>& source, size_t prev_end) {
    size_t start = source->text_str().find("#", prev_end);
    if (start == std::string::npos) {
      return start;
    }
    start += 1;
    static constexpr size_t max_whitespace = 6;
    size_t i = 0;
    while (start + i < source->size() && i < max_whitespace) {
      auto c = source->char_at(start + i);
      if (c != ' ' && c != '\t') {
        break;
      }
      i++;
    }
    static const std::string check = "CHECK";
    if (source->text_str().substr(start + i, check.size()) == check) {
      return start + i + check.size();
    } else {
      return findNextStart(source, start + i + 1);
    }
  }

  void parseStrings(const std::shared_ptr<Source>& source) {
    size_t start = 0;
    start = findNextStart(source, 0);
    while (start != std::string::npos) {
      bool found_match = parseSingleCheck(source, &start);
      if (!found_match) {
        std::ostringstream ss;
        ss << "Could not parse check at:\n";
        SourceRange(source, start, start + 1).highlight(ss);
        ss << "Check for bad input.";
        has_run = true;
        throw std::runtime_error(std::move(ss).str());
      }
      start = findNextStart(source, start);
    }
  }

  void doCheckNot(
      const std::vector<Check>& nots,
      const std::shared_ptr<Source>& source,
      const SourceRange& prev,
      const SourceRange& next,
      const VariableMap& variables) {
    auto start = prev.end(); // inclusive
    auto end = next.start(); // exclusive
    if (end < start) {
      return;
    }
    for (const auto& check : nots) {
      AT_ASSERT(check.type_ == CHECK_NOT);
      assertNotFindPattern(SourceRange(source, start, end), check, variables);
    }
  }

  // Checks that source token is highlighted, does not advance search range.
  void doCheckSourceHighlighted(
      const Check& check,
      const std::shared_ptr<Source>& source,
      size_t start_offset) {
    auto construct_error_and_throw = [&](size_t error_start_pos) {
      SourceRange error_range(
          source, error_start_pos, check.search_str_.size());
      std::stringstream ss;
      ss << "Expected to find ";
      c10::printQuotedString(ss, check.search_str_);
      ss << "highlighted but it is not." << '\n';
      error_range.highlight(ss);
      throw std::runtime_error(std::move(ss).str());
    };

    size_t search_start_offset = start_offset;
    bool found_token_at_least_once = false;
    size_t pos = search_start_offset;
    while (pos < source->size()) {
      pos = source->text_str().find(check.search_str_, search_start_offset);
      if (pos == std::string::npos) {
        break;
      }

      found_token_at_least_once = true;

      auto lineno = source->lineno_for_offset(pos);
      auto col = pos - source->offset_for_line(lineno);
      auto highlight_lineno = lineno + 1;

      if (highlight_lineno >= source->num_lines()) {
        construct_error_and_throw(pos);
      }

      auto highlight_start_offset =
          source->offset_for_line(highlight_lineno) + col;
      auto highlight_end_offset = std::min(
          highlight_start_offset + check.search_str_.size(), source->size());

      if (highlight_end_offset >= source->size()) {
        construct_error_and_throw(pos);
      }

      bool found_highlight = true;
      for (const auto posi :
           c10::irange(highlight_start_offset, highlight_end_offset)) {
        if (source->char_at(posi) != '~') {
          found_highlight = false;
        }
      }

      if (found_highlight) {
        assertNotFind(
            SourceRange(
                source, highlight_start_offset - 1, highlight_start_offset),
            "~",
            check);
        assertNotFind(
            SourceRange(source, highlight_end_offset, highlight_end_offset + 1),
            "~",
            check);
        return;
      }

      search_start_offset = pos + 1;
    }

    if (!found_token_at_least_once) {
      // Guaranteed to fail to generate error message.
      assertFind(source, check.search_str_, start_offset, check);
    }

    construct_error_and_throw(start_offset);
  }

  SourceRange matchDagGroup(
      const std::vector<Check>& group,
      const std::shared_ptr<Source>& source,
      const SourceRange& prev,
      VariableMap& variables) {
    size_t group_beg = std::string::npos;
    size_t group_end = 0;

    AT_ASSERT(!groups.empty());
    for (const auto& check : group) {
      AT_ASSERT(check.type_ == group[0].type_);
      auto match = assertFindPattern(source, check, prev.end(), variables);
      updateVariables(variables, match);
      group_beg = std::min(match.start, group_beg);
      group_end = std::max(match.end, group_end);
    }

    return SourceRange(source, group_beg, group_end);
  }

  SourceRange matchGroup(
      const std::vector<Check>& group,
      const std::shared_ptr<Source>& source,
      const SourceRange& prev,
      VariableMap& variables) {
    AT_ASSERT(!group.empty());
    CheckType type = group[0].type_;

    if (type == CHECK_DAG) {
      return matchDagGroup(group, source, prev, variables);
    }
    AT_ASSERT(type != CHECK_NOT);
    AT_ASSERT(group.size() == 1);

    const auto& check = group[0];
    size_t start_range = prev.end();
    size_t end_range = start_range;

    switch (check.type_) {
      case CHECK: {
        auto match = assertFindPattern(source, check, start_range, variables);
        updateVariables(variables, match);
        start_range = match.start;
        end_range = match.end;
      } break;
      case CHECK_SAME: {
        auto match = assertFindPattern(source, check, start_range, variables);
        assertNotFind(
            SourceRange(source, prev.end(), match.start), "\n", check);
        updateVariables(variables, match);
        start_range = match.start;
        end_range = match.end;
      } break;
      case CHECK_NEXT: {
        auto line_end = assertFind(source, "\n", start_range, check);
        auto match = assertFindPattern(source, check, line_end + 1, variables);
        assertNotFind(
            SourceRange(source, line_end + 1, match.start), "\n", check);
        updateVariables(variables, match);
        start_range = match.start;
        end_range = match.end;
      } break;
      case CHECK_COUNT: {
        auto group_start_range = std::string::npos;
        AT_ASSERT(check.count_ && *check.count_ != 0);
        for (size_t i = 0; i < *check.count_; ++i) {
          auto match = assertFindPattern(source, check, start_range, variables);
          updateVariables(variables, match);
          start_range = match.start;
          group_start_range = std::min(match.start, group_start_range);
          end_range = match.end;
          start_range = end_range;
        }
        start_range = group_start_range;
      } break;
      case CHECK_SOURCE_HIGHLIGHTED: {
        doCheckSourceHighlighted(check, source, start_range);
        break;
      }
      case CHECK_REGEX: {
        start_range =
            assertFindRegex(source, check.search_str_, start_range, check);
        end_range = start_range + check.search_str_.size();
        break;
      }
      default:
        TORCH_CHECK(false);
    }

    return SourceRange(source, start_range, end_range);
  }

  void doChecks(const std::shared_ptr<Source>& source) {
    SourceRange prev(source, 0, 0);
    VariableMap variables;
    for (size_t i = 0; i < groups.size(); i++) {
      const auto& curr_group = groups[i];
      CheckType type = curr_group.at(0).type_;
      if (type != CHECK_NOT) {
        prev = matchGroup(curr_group, source, prev, variables);
      } else {
        if (i + 1 < groups.size()) {
          const auto& next_group = groups[i + 1];
          AT_ASSERT(next_group.at(0).type_ != CHECK_NOT);
          auto next_variables = variables;
          SourceRange after_not =
              matchGroup(next_group, source, prev, next_variables);
          doCheckNot(curr_group, source, prev, after_not, variables);
          variables = std::move(next_variables);
          prev = after_not;
          ++i; // already checked the group after
        } else {
          SourceRange end_of_file(
              source, source->size() + 1, source->size() + 1);
          doCheckNot(curr_group, source, prev, end_of_file, variables);
        }
      }
    }
  }

  std::vector<Check> checks;
  std::vector<std::vector<Check>> groups;
};

FileCheck::FileCheck() : fcImpl(new FileCheckImpl()) {}

std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc) {
  out << "FileCheck checks:\n";
  for (const Check& c : fc.checks) {
    out << '\t' << c << '\n';
  }
  return out;
}

FileCheck::~FileCheck() {
  if (!fcImpl->has_run) {
    std::cout << "You have not run this instance of FileCheck!\n";
    std::cout << *fcImpl;
  }
  fcImpl.reset();
}

void FileCheck::run(const std::string& test_file) {
  fcImpl->run(test_file);
}

void FileCheck::run(const Graph& graph) {
  std::stringstream graph_str;
  graph_str << graph;
  fcImpl->run(std::move(graph_str).str());
}

void FileCheck::run(
    const std::string& input_checks_string,
    const std::string& test_string) {
  fcImpl->run(input_checks_string, test_string);
}

void FileCheck::run(
    const std::string& input_checks_string,
    const Graph& graph) {
  std::stringstream graph_str;
  graph_str << graph;
  fcImpl->run(input_checks_string, std::move(graph_str).str());
}

FileCheck* FileCheck::check(const std::string& str) {
  fcImpl->addCheck(CHECK, str);
  return this;
}

FileCheck* FileCheck::check_not(const std::string& str) {
  fcImpl->addCheck(CHECK_NOT, str);
  return this;
}

FileCheck* FileCheck::check_same(const std::string& str) {
  fcImpl->addCheck(CHECK_SAME, str);
  return this;
}

FileCheck* FileCheck::check_next(const std::string& str) {
  fcImpl->addCheck(CHECK_NEXT, str);
  return this;
}

FileCheck* FileCheck::check_count(
    const std::string& str,
    size_t count,
    bool exactly) {
  TORCH_INTERNAL_ASSERT(
      count != 0 || exactly, "Count == 0 && !exactly doesn't do anything");
  if (count) {
    fcImpl->addCheck(CHECK_COUNT, str, count);
  }
  if (exactly) {
    fcImpl->addCheck(CHECK_NOT, str);
  }
  return this;
}

FileCheck* FileCheck::check_dag(const std::string& str) {
  fcImpl->addCheck(CHECK_DAG, str);
  return this;
}

FileCheck* FileCheck::check_source_highlighted(const std::string& str) {
  fcImpl->addCheck(CHECK_SOURCE_HIGHLIGHTED, str);
  return this;
}

FileCheck* FileCheck::check_regex(const std::string& str) {
  fcImpl->addCheck(CHECK_REGEX, str);
  return this;
}

} // namespace torch::jit::testing
