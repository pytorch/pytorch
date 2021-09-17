#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <unordered_set>

namespace torch_lazy_tensors {
namespace cpp_test {

class LtcTsTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup() {}

  void ExpectCounterNotChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ExpectCounterChanged(const std::string& counter_regex,
                            const std::unordered_set<std::string>* ignore_set) {
  }

  void ResetCounters() {}

 private:
  void MakeEndSnapshot() {}
};

class AtenLtcTsTensorTestBase : public LtcTsTest {
 protected:
  static void SetUpTestCase() {}
};

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
