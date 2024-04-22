#include <gtest/gtest.h>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

#include <utility>
#include <vector>

using namespace c10d;

TEST(ProcessGroupUCCTest, testTrim) {
  std::vector<std::pair<std::string, std::string>> tests = {
      {" allreduce ", "allreduce"},
      {"\tallgather", "allgather"},
      {"send\n", "send"},
  };
  for (auto entry : tests) {
    ASSERT_EQ(trim(entry.first), entry.second);
  }
}

TEST(ProcessGroupUCCTest, testToLower) {
  std::vector<std::pair<std::string, std::string>> tests = {
      {"AllReduce", "allreduce"},
      {"ALLGATHER", "allgather"},
      {"send", "send"},
  };
  for (auto entry : tests) {
    ASSERT_EQ(tolower(entry.first), entry.second);
  }
}

TEST(ProcessGroupUCCTest, testParseList) {
  std::string input = "\tAllReduce, ALLGATHER, send\n";
  std::vector<std::string> expect{"allreduce", "allgather", "send"};
  ASSERT_EQ(parse_list(input), expect);
}
