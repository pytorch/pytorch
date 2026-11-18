#include <gtest/gtest.h>
#include <torch/torch.h>
#include <limits>
#include <sstream>

bool ends_with(const std::string& str, const std::string& suffix) {
  const auto str_len = str.length();
  const auto suffix_len = suffix.length();
  return str_len < suffix_len ? false : suffix == str.substr(str_len - suffix_len, suffix_len);
}

TEST(MPSPrintTest, PrintFloatMatrix) {
  std::stringstream ss;
  ss << torch::randn({3, 3}, at::device(at::kMPS));
  ASSERT_TRUE (ends_with(ss.str(), "[ MPSFloatType{3,3} ]")) << " got " << ss.str();
}

TEST(MPSPrintTest, PrintHalf4DTensor) {
  std::stringstream ss;
  ss << torch::randn({2, 2, 2, 2}, at::device(at::kMPS).dtype(at::kHalf));
  ASSERT_TRUE (ends_with(ss.str(), "[ MPSHalfType{2,2,2,2} ]")) << " got " << ss.str();
}

TEST(MPSPrintTest, PrintLongMatrix) {
  std::stringstream ss;
  ss << torch::full({2, 2}, std::numeric_limits<int>::max(), at::device(at::kMPS));
  ASSERT_TRUE (ends_with(ss.str(), "[ MPSLongType{2,2} ]")) << " got " << ss.str();
}

TEST(MPSPrintTest, PrintFloatScalar) {
  std::stringstream ss;
  ss << torch::ones({}, at::device(at::kMPS));
  ASSERT_TRUE(ss.str() == "1\n[ MPSFloatType{} ]") << " got " << ss.str();
}
