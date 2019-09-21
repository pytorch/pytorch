#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

namespace F = torch::nn::functional;

using namespace torch::nn;

struct FunctionalTest : torch::test::SeedingFixture {};

TEST_F(FunctionalTest, MaxPool1d) {
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto y = F::max_pool1d(x, MaxPool1dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1 ,2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, MaxPool2d) {
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = F::max_pool2d(x, MaxPool2dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2 ,2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(FunctionalTest, MaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = F::max_pool3d(x, MaxPool3dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool1d) {
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto y = F::avg_pool1d(x, AvgPool1dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, AvgPool2d) {
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = F::avg_pool2d(x, AvgPool2dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = F::avg_pool3d(x, AvgPool3dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, Pad) {
  auto t4d = torch::empty({3, 3, 4, 2});
  std::vector<int64_t> p1d = {1, 1}; // pad last dim by 1 on each side
  auto out1 = F::pad(t4d, PadOptions(p1d).mode("constant").value(0));  // effectively zero padding
  ASSERT_EQ(out1.sizes(), torch::IntArrayRef({3, 3, 4, 4}));

  std::vector<int64_t> p2d = {1, 1, 2, 2}; // pad last dim by (1, 1) and 2nd to last by (2, 2)
  auto out2 = F::pad(t4d, PadOptions(p2d).mode("constant").value(0));
  ASSERT_EQ(out2.sizes(), torch::IntArrayRef({3, 3, 8, 4}));

  t4d = torch::empty({3, 3, 4, 2});
  std::vector<int64_t> p3d = {0, 1, 2, 1, 3, 3}; // pad by (0, 1), (2, 1), and (3, 3)
  auto out3 = F::pad(t4d, PadOptions(p3d).mode("constant").value(0));
  ASSERT_EQ(out3.sizes(), torch::IntArrayRef({3, 9, 7, 3}));
}
