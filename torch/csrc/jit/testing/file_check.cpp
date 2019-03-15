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
    const Check& check) {
  auto pos = search_range.file_ptr()->find(sub, search_range.start());
  if (pos == std::string::npos || (pos + sub.size()) > search_range.end()) {
    auto found_range =
        SourceRange(search_range.file_ptr(), search_range.start(), sub.size());
    std::stringstream ss;
    ss << "Expected to find ";
    printQuotedString(ss, sub);
    ss << " but did not find it\n";
    found_range.highlight(ss);
    ss << "From " << check << "\n";
    throw std::runtime_error(ss.str());
  }
  return pos;
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
} // namespace

struct FileCheckImpl {
  TORCH_API explicit FileCheckImpl() = default;

  TORCH_API void run(const std::string& test_file) {
    has_run = true;
    doChecks(std::make_shared<std::string>(test_file));
  }

  TORCH_API void addCheck(
      CheckType type,
      const std::string& s,
      c10::optional<size_t> count = c10::nullopt) {
    Check check(type, s, std::move(count));

    // consecutive CHECK_DAGs & CHECK_NOTs need to be evaluated as a group
    if (groups.size() == 0 || (type != CHECK_NOT && type != CHECK_DAG)) {
      groups.push_back({check});
    } else {
      auto& last_group = groups.back();
      if (last_group.at(0).type_ == type) {
        last_group.push_back(check);
      } else {
        groups.push_back({check});
      }
    }

    has_run = false;
  }

  bool has_run = false;

  friend std::ostream& operator<<(std::ostream& out, const FileCheckImpl& fc);

 private:
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
  std::shared_ptr<std::string> check_file;
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
