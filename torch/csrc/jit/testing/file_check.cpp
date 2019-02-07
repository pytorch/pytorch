//==-- llvm/Support/FileCheck.h ---------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// modified from llvm::FileCheck

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/source_range.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
namespace testing {

enum CheckType {
  CHECK,
  CHECK_NEXT,
  CHECK_SAME,
  CHECK_NOT,
  CHECK_COUNT,
  CHECK_DAG,
};

struct Check {
  Check(
      CheckType type,
      std::string str,
      c10::optional<size_t> count = c10::nullopt)
      : type_(type), search_str_(std::move(str)) {
    count_ = count;
  };

  void setSourceLocation(SourceRange sl) {
    source_range_ = std::move(sl);
  }

  CheckType type_;
  c10::optional<size_t> count_;
  const std::string search_str_;
  c10::optional<SourceRange> source_range_;
};

// For a basic CHECK, the match is equal to the matching substring
// For a CHECK-DAG group, the begin is first position of all checks,
// and end is the last position of all checks.
struct Match {
  Match(int64_t begin, int64_t end) : begin(begin), end(end){};

  int64_t begin; // inclusive
  int64_t end; // exclusive
};

namespace {
static std::string escapeString(const std::string& input) {
  std::string s = input;
  std::vector<char> search = {'\n', '\t', '\v'};
  std::vector<std::string> replace = {"\\n", "\\t", "\\v"};
  for (size_t i = 0; i < search.size(); i++) {
    for (size_t i = 0; i < search.size(); i++) {
      size_t pos = s.find(search[i]);
      while (pos != std::string::npos) {
        s.replace(pos, 1, replace[i]);
        pos = s.find(search[i], pos + 1);
      }
    }
  }
  return s;
}

size_t assertFind(
    std::shared_ptr<std::string> file,
    const std::string& sub,
    size_t start) {
  auto pos = file->find(sub, start);
  if (pos == std::string::npos) {
    auto range = SourceRange(file, start, sub.size());
    std::stringstream ss;
    ss << "Expected to find '" << escapeString(sub)
       << "' but did not find it\n";
    range.highlight(ss);
    throw std::runtime_error(ss.str());
  }
  return pos;
}

size_t assertFind(
    const std::string& file,
    const std::string& sub,
    size_t start,
    const Check& check) {
  auto pos = file.find(sub, start);
  if (pos == std::string::npos) {
    auto range =
        SourceRange(std::make_shared<std::string>(file), start, sub.size());
    std::stringstream ss;
    ss << "Expected to find '" << escapeString(sub)
       << "' but did not find it\n";
    range.highlight(ss);
    ss << "From the check defined\n";
    check.source_range_->highlight(ss);
    throw std::runtime_error(ss.str());
  }
  return pos;
}

void assertNotFind(
    const std::string& file,
    const std::string& sub,
    const Check& check) {
  auto pos = file.find(sub);
  if (pos != std::string::npos) {
    auto range =
        SourceRange(std::make_shared<std::string>(file), pos, sub.size() + pos);
    std::stringstream ss;
    ss << "Expected to not find '" << escapeString(sub) << "' but found it\n";
    range.highlight(ss);
    ss << "From the check defined\n";
    check.source_range_->highlight(ss);
    throw std::runtime_error(ss.str());
  }
}
} // namespace

struct FileCheckImpl {
 public:
  TORCH_API explicit FileCheckImpl(const std::string& file) {
    check_file = std::make_shared<std::string>(file);
    makeGroups(parseStrings());
  }

  TORCH_API void checkFile(const std::string& test_file) {
    doChecks(test_file);
  }

 private:
  std::vector<Check> parseStrings() {
    std::vector<Check> operands;
    std::vector<std::pair<CheckType, std::string>> check_pairs = {
        {CHECK, ":"},
        {CHECK_NEXT, "-NEXT:"},
        {CHECK_SAME, "-SAME:"},
        {CHECK_NOT, "-NOT:"},
        {CHECK_DAG, "-DAG:"},
        {CHECK_COUNT, "-COUNT-"}, // needs special parsing
    };
    const std::string prefix = "CHECK";

    size_t start = 0;
    start = check_file->find(prefix, start);

    while (start != std::string::npos) {
      for (const auto& check_pair : check_pairs) {
        const std::string& check_suffix = check_pair.second;
        auto suffix_pos = check_file->find(check_suffix, start);
        if (suffix_pos != start + prefix.size()) {
          continue;
        }
        size_t end_check_string = suffix_pos + check_suffix.size();
        CheckType type = check_pair.first;
        c10::optional<size_t> count = c10::nullopt;
        if (type == CHECK_COUNT) {
          size_t end = assertFind(check_file, ":", end_check_string);
          count = std::stoll(
              check_file->substr(end_check_string, end - end_check_string));
          end_check_string = end + 1;
        }
        auto end_line = check_file->find("\n", end_check_string);
        auto check = Check(
            type,
            check_file->substr(end_check_string, end_line - end_check_string),
            count);
        check.setSourceLocation(
            SourceRange(check_file, start, end_check_string));
        operands.push_back(check);
        start = end_line;
        break;
      }
      start = check_file->find(prefix, start);
      ;
    }
    return operands;
  }

  // consecutive CHECK_DAGs & CHECK_NOTs need to be evaluated as a group
  void makeGroups(std::vector<Check> input) {
    if (input.size() == 0) {
      return;
    }
    for (size_t i = 0; i < input.size(); ++i) {
      std::vector<Check> group = {input[i]};
      CheckType type = input[i].type_;
      if (type != CHECK_NOT && type != CHECK_DAG) {
        groups.push_back(group);
        continue;
      }
      while (i + 1 < input.size() && input[i + 1].type_ == type) {
        group.push_back(input[++i]);
      }
      groups.push_back(group);
    }
  }

  void doCheckNot(
      const std::vector<Check>& nots,
      const std::string& file,
      Match prev,
      Match next) {
    auto start = prev.end; // inclusive
    auto end = next.begin; // exclusive
    if (end < start) {
      return;
    }
    const auto& substr = file.substr(start, end - start);
    for (const auto& check : nots) {
      AT_ASSERT(check.type_ == CHECK_NOT);
      assertNotFind(substr, check.search_str_, check);
    }
  }

  Match matchDagGroup(
      const std::vector<Check>& group,
      const std::string& test_file,
      Match prev) {
    size_t group_beg = std::string::npos;
    size_t group_end = 0;

    AT_ASSERT(groups.size() != 0);
    for (const auto& check : group) {
      AT_ASSERT(check.type_ == group[0].type_);
      auto pos = assertFind(test_file, check.search_str_, prev.end, check);
      group_beg = std::min(pos, group_beg);
      group_end = std::max(pos + check.search_str_.size(), group_end);
    }

    return Match(group_beg, group_end);
  }

  Match matchGroup(
      const std::vector<Check>& group,
      const std::string& test_file,
      Match prev) {
    AT_ASSERT(group.size() != 0);
    CheckType type = group[0].type_;

    if (type == CHECK_DAG) {
      return matchDagGroup(group, test_file, prev);
    }
    AT_ASSERT(type != CHECK_NOT);
    AT_ASSERT(group.size() == 1);

    const auto& check = group[0];
    size_t start_range = prev.end;
    size_t end_range = start_range;

    switch (check.type_) {
      case CHECK: {
        start_range =
            assertFind(test_file, check.search_str_, start_range, check);
        end_range = start_range + check.search_str_.size();
      } break;
      case CHECK_SAME: {
        auto pos = assertFind(test_file, check.search_str_, start_range, check);
        assertNotFind(test_file.substr(prev.end, pos), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_NEXT: {
        auto line_end = assertFind(test_file, "\n", start_range, check);
        auto pos =
            assertFind(test_file, check.search_str_, line_end + 1, check);
        assertNotFind(
            test_file.substr(line_end + 1, pos - (line_end + 1)), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_COUNT: {
        auto group_start_range = std::string::npos;
        AT_ASSERT(check.count_ && *check.count_ != 0);
        for (size_t i = 0; i < *check.count_; ++i) {
          start_range =
              assertFind(test_file, check.search_str_, start_range, check);
          group_start_range = std::min(start_range, group_start_range);
          end_range = start_range + check.search_str_.size();
          start_range = end_range;
        }
        start_range = group_start_range;
      } break;
      case CHECK_DAG: {
        AT_ASSERT(false);
      } break;
      case CHECK_NOT: {
        AT_ASSERT(false);
      } break;
    }
    return Match(start_range, end_range);
  }

  void doChecks(const std::string& test_file) {
    Match prev(0, 0);
    for (size_t i = 0; i < groups.size(); i++) {
      const auto& curr_group = groups[i];
      CheckType type = curr_group.at(0).type_;
      if (type != CHECK_NOT) {
        prev = matchGroup(curr_group, test_file, prev);
      } else {
        if (i + 1 < groups.size()) {
          const auto& next_group = groups[i + 1];
          AT_ASSERT(next_group.at(0).type_ != CHECK_NOT);
          Match after_not = matchGroup(next_group, test_file, prev);
          doCheckNot(curr_group, test_file, prev, after_not);
          prev = after_not;
          ++i; // already checked the group after
        } else {
          Match end_of_file(test_file.size() + 1, test_file.size() + 1);
          doCheckNot(curr_group, test_file, prev, end_of_file);
        }
      }
    }
  }

  std::shared_ptr<std::string> check_file;
  std::vector<std::vector<Check>> groups;
};

void FileCheck::checkFile(
    const std::string& check_file,
    const std::string& test_file) {
  FileCheckImpl(check_file).checkFile(test_file);
};

} // namespace testing
} // namespace jit
} // namespace torch
