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
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/source_range.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

void printQuotedString(std::ostream& stmt, const std::string& str);

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
    count_ = std::move(count);
  };

  CheckType type_;
  c10::optional<size_t> count_;
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
  }
  out << ": " << c.search_str_;
  return out;
};

namespace {

size_t assertFind(
    const SourceRange& search_range,
    const std::string& sub,
    std::function<void(std::ostream& out)> extra_msg = nullptr) {
  auto pos = search_range.file_ptr()->find(sub, search_range.start());
  if (pos == std::string::npos || (pos + sub.size()) > search_range.end()) {
    auto found_range =
        SourceRange(search_range.file_ptr(), search_range.start(), sub.size());
    std::stringstream ss;
    ss << "Expected to find ";
    printQuotedString(ss, sub);
    ss << " but did not find it\n";
    found_range.highlight(ss);
    if (extra_msg) {
      extra_msg(ss);
    }
    throw std::runtime_error(ss.str());
  }
  return pos;
}

size_t assertFind(
    const SourceRange& search_range,
    const std::string& sub,
    const Check& check) {
  return assertFind(search_range, sub, [&](std::ostream& out) {
    out << "From " << check << "\n";
  });
}

size_t assertFind(
    const std::shared_ptr<std::string>& file,
    const std::string& sub,
    size_t start,
    const Check& check) {
  return assertFind(SourceRange(file, start, file->size()), sub, check);
}

void assertNotFind(
    const SourceRange& search_range,
    const std::string& sub,
    const Check& check) {
  auto pos = search_range.file_ptr()->find(sub, search_range.start());
  if (pos != std::string::npos && (pos + sub.size()) <= search_range.end()) {
    auto found_range =
        SourceRange(search_range.file_ptr(), pos, sub.size() + pos);
    std::stringstream ss;
    ss << "Expected to not find ";
    printQuotedString(ss, sub);
    ss << " but found it\n";
    found_range.highlight(ss);
    ss << "From " << check << "\n";
    throw std::runtime_error(ss.str());
  }
}

size_t substringCount(
    const std::shared_ptr<std::string>& file,
    const std::string& sub) {
  size_t occurances = 0;
  std::string::size_type pos = 0;
  while ((pos = file->find(sub, pos)) != std::string::npos) {
    ++occurances;
    pos += sub.length();
  }
  return occurances;
}
} // namespace

struct FileCheckImpl {
  TORCH_API explicit FileCheckImpl() = default;

  TORCH_API void run(const std::string& test_file) {
    has_run = true;

    if (groups.size() == 0 || groups[0].size() == 0) {
      throw std::runtime_error(
          "No checks have been added to this instance of"
          "Filecheck! Check for bad input.");
    }

    doChecks(std::make_shared<std::string>(test_file));
  }

  TORCH_API void run(
      const std::string& checks_file,
      const std::string& test_file) {
    auto checks_ptr = std::make_shared<std::string>(checks_file);
    parseStrings(checks_ptr);
    run(test_file);
  }

  TORCH_API void addCheck(Check check) {
    // consecutive CHECK_DAGs & CHECK_NOTs need to be evaluated as a group
    if (groups.size() == 0 ||
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
    has_run = false;
  }

  TORCH_API void addCheck(
      CheckType type,
      const std::string& s,
      c10::optional<size_t> count = c10::nullopt) {
    addCheck(Check(type, s, std::move(count)));
  }

  bool has_run = false;

  friend std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc);

 private:
  bool parseSingleCheck(
      const std::shared_ptr<std::string>& checks_file,
      size_t* start) {
    const static std::vector<std::pair<CheckType, std::string>> check_pairs = {
        {CHECK, ": "},
        {CHECK_NEXT, "-NEXT: "},
        {CHECK_SAME, "-SAME: "},
        {CHECK_NOT, "-NOT: "},
        {CHECK_DAG, "-DAG: "},
        {CHECK_COUNT, "-COUNT-"}, // needs special parsing
    };

    for (const auto& check_pair : check_pairs) {
      const std::string& check_suffix = check_pair.second;
      auto suffix_pos = checks_file->find(check_suffix, *start);
      if (suffix_pos != *start) {
        continue;
      }
      size_t end_check_string = suffix_pos + check_suffix.size();
      CheckType type = check_pair.first;
      c10::optional<size_t> count = c10::nullopt;
      auto end_line = checks_file->find("\n", end_check_string);
      bool exactly = false;
      if (type == CHECK_COUNT) {
        const std::string exact = "EXACTLY-";
        if (checks_file->find(exact, end_check_string) == end_check_string) {
          exactly = true;
          end_check_string += exact.size();
        }
        size_t end = assertFind(
            SourceRange(checks_file, end_check_string, end_line), ":");
        count = std::stoll(
            checks_file->substr(end_check_string, end - end_check_string));
        end_check_string = end + 2; // add ':' and the space
      }
      auto check = Check(
          type,
          checks_file->substr(end_check_string, end_line - end_check_string),
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

  size_t findNextStart(
      const std::shared_ptr<std::string>& checks_file,
      size_t prev_end) {
    size_t start = checks_file->find("#", prev_end);
    if (start == std::string::npos) {
      return start;
    }
    start += 1;
    static constexpr size_t max_whitespace = 6;
    size_t i = 0;
    while (start + i < checks_file->size() && i < max_whitespace) {
      auto c = checks_file->at(start + i);
      if (c != ' ' && c != '\t') {
        break;
      }
      i++;
    }
    static const std::string check = "CHECK";
    if (checks_file->substr(start + i, check.size()) == check) {
      return start + i + check.size();
    } else {
      return findNextStart(checks_file, start + i + 1);
    }
  }

  void parseStrings(const std::shared_ptr<std::string>& checks_file) {
    size_t start = 0;
    start = findNextStart(checks_file, 0);
    while (start != std::string::npos) {
      bool found_match = parseSingleCheck(checks_file, &start);
      if (!found_match) {
        std::ostringstream ss;
        ss << "Could not parse check at:\n";
        SourceRange(checks_file, start, start + 1).highlight(ss);
        ss << "Check for bad input.";
        has_run = true;
        throw std::runtime_error(ss.str());
      }
      start = findNextStart(checks_file, start);
    }
  }

  void doCheckNot(
      const std::vector<Check>& nots,
      const std::shared_ptr<std::string>& file,
      const SourceRange& prev,
      const SourceRange& next) {
    auto start = prev.end(); // inclusive
    auto end = next.start(); // exclusive
    if (end < start) {
      return;
    }
    for (const auto& check : nots) {
      AT_ASSERT(check.type_ == CHECK_NOT);
      assertNotFind(SourceRange(file, start, end), check.search_str_, check);
    }
  }

  SourceRange matchDagGroup(
      const std::vector<Check>& group,
      const std::shared_ptr<std::string>& test_file,
      const SourceRange& prev) {
    size_t group_beg = std::string::npos;
    size_t group_end = 0;

    AT_ASSERT(groups.size() != 0);
    for (const auto& check : group) {
      AT_ASSERT(check.type_ == group[0].type_);
      auto pos = assertFind(test_file, check.search_str_, prev.end(), check);
      group_beg = std::min(pos, group_beg);
      group_end = std::max(pos + check.search_str_.size(), group_end);
    }

    return SourceRange(test_file, group_beg, group_end);
  }

  SourceRange matchGroup(
      const std::vector<Check>& group,
      const std::shared_ptr<std::string>& test_file,
      const SourceRange& prev) {
    AT_ASSERT(group.size() != 0);
    CheckType type = group[0].type_;

    if (type == CHECK_DAG) {
      return matchDagGroup(group, test_file, prev);
    }
    AT_ASSERT(type != CHECK_NOT);
    AT_ASSERT(group.size() == 1);

    const auto& check = group[0];
    size_t start_range = prev.end();
    size_t end_range = start_range;

    switch (check.type_) {
      case CHECK: {
        start_range =
            assertFind(test_file, check.search_str_, start_range, check);
        end_range = start_range + check.search_str_.size();
      } break;
      case CHECK_SAME: {
        auto pos = assertFind(test_file, check.search_str_, start_range, check);
        assertNotFind(SourceRange(test_file, prev.end(), pos), "\n", check);
        start_range = pos;
        end_range = pos + check.search_str_.size();
      } break;
      case CHECK_NEXT: {
        auto line_end = assertFind(test_file, "\n", start_range, check);
        auto pos =
            assertFind(test_file, check.search_str_, line_end + 1, check);
        assertNotFind(SourceRange(test_file, line_end + 1, pos), "\n", check);
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
        AT_ERROR();
      } break;
      case CHECK_NOT: {
        AT_ERROR();
      } break;
    }
    return SourceRange(test_file, start_range, end_range);
  }

  void doChecks(const std::shared_ptr<std::string>& test_file) {
    SourceRange prev(test_file, 0, 0);
    for (size_t i = 0; i < groups.size(); i++) {
      const auto& curr_group = groups[i];
      CheckType type = curr_group.at(0).type_;
      if (type != CHECK_NOT) {
        prev = matchGroup(curr_group, test_file, prev);
      } else {
        if (i + 1 < groups.size()) {
          const auto& next_group = groups[i + 1];
          AT_ASSERT(next_group.at(0).type_ != CHECK_NOT);
          SourceRange after_not = matchGroup(next_group, test_file, prev);
          doCheckNot(curr_group, test_file, prev, after_not);
          prev = after_not;
          ++i; // already checked the group after
        } else {
          SourceRange end_of_file(
              test_file, test_file->size() + 1, test_file->size() + 1);
          doCheckNot(curr_group, test_file, prev, end_of_file);
        }
      }
    }
  }

  std::vector<Check> checks;
  std::vector<std::vector<Check>> groups;
};

FileCheck::FileCheck() : fcImpl(new FileCheckImpl()){};

std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc) {
  out << "FileCheck checks:\n";
  for (const Check& c : fc.checks) {
    out << "\t" << c << "\n";
  }
  return out;
};

FileCheck::~FileCheck() {
  if (!fcImpl->has_run) {
    std::cout << "You have not run this instance of FileCheck!\n";
    std::cout << *fcImpl;
  }
  fcImpl.reset();
};

void FileCheck::run(const std::string& test_file) {
  fcImpl->run(test_file);
};

void FileCheck::run(const Graph& graph) {
  std::stringstream graph_str;
  graph_str << graph;
  fcImpl->run(graph_str.str());
};

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
  fcImpl->run(input_checks_string, graph_str.str());
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
  fcImpl->addCheck(CHECK_COUNT, str, count);
  if (exactly) {
    fcImpl->addCheck(CHECK_NOT, str);
  }
  return this;
}

FileCheck* FileCheck::check_dag(const std::string& str) {
  fcImpl->addCheck(CHECK_DAG, str);
  return this;
}
} // namespace testing
} // namespace jit
} // namespace torch
