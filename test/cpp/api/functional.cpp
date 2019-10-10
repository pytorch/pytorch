#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

namespace F = torch::nn::functional;

using namespace torch::nn;

struct FunctionalTest : torch::test::SeedingFixture {};

TEST_F(FunctionalTest, MaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::max_pool1d(x, MaxPool1dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, MaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::max_pool2d(x, MaxPool2dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(FunctionalTest, MaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::max_pool3d(x, MaxPool3dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::avg_pool1d(x, AvgPool1dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, AvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::avg_pool2d(x, AvgPool2dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::avg_pool3d(x, AvgPool3dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, CosineSimilarity) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output =
      F::cosine_similarity(input1, input2, CosineSimilarityOptions().dim(1));
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, PairwiseDistance) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output =
      F::pairwise_distance(input1, input2, PairwiseDistanceOptions(1));
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, PDist) {
  {
    auto input = torch::tensor({{-1.0, -5.0, -1.0}, {2.0, 4.0, 6.0}});
    auto output = F::pdist(input);
    auto expected = torch::tensor({11.7898});
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    auto input = torch::tensor({{1.0, -1.0}, {1.0, 3.0}, {3.0, 3.0}});
    auto output = F::pdist(input, 1.5);
    auto expected = torch::tensor({4.0, 4.8945, 2.0});
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(FunctionalTest, AdaptiveMaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_max_pool1d(x, AdaptiveMaxPool1dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
}

TEST_F(FunctionalTest, AdaptiveMaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_max_pool2d(x, AdaptiveMaxPool2dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveMaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_max_pool3d(x, AdaptiveMaxPool3dOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_avg_pool1d(x, AdaptiveAvgPool1dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_avg_pool2d(x, AdaptiveAvgPool2dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_avg_pool3d(x, AdaptiveAvgPool3dOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3, 3}));
}

TEST_F(FunctionalTest, HingeEmbeddingLoss) {
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::kFloat);
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  auto output = F::hinge_embedding_loss(
      input, target, HingeEmbeddingLossOptions().margin(2));
  auto expected = torch::tensor({10}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, AffineGrid) {
  {
    // 2D affine.
    auto theta = torch::arange(1, 13, torch::kDouble)
                     .view(torch::IntArrayRef({2, 2, 3}));
    auto size = torch::IntArrayRef({2, 3, 2, 2}).vec();
    auto align_corners = true;
    auto output = F::affine_grid(theta, size, !align_corners);
    auto expected = torch::tensor(
        {{{{1.50, 1.50}, {2.50, 5.50}}, {{3.50, 6.50}, {4.50, 10.50}}},
         {{{1.50, 1.50}, {8.50, 11.50}}, {{9.50, 12.50}, {16.50, 22.50}}}});
    auto output_aligned = F::affine_grid(theta, size, align_corners);
    auto expected_aligned = torch::tensor(
        {{{{0.0, -3.0}, {2.0, 5.0}}, {{4.0, 7.0}, {6.0, 15.0}}},
         {{{-6.0, -9.0}, {8.0, 11.0}}, {{10.0, 13.0}, {24.0, 33.0}}}});

    ASSERT_TRUE(output.allclose(expected));
    ASSERT_TRUE(output_aligned.allclose(expected_aligned));
  }
  {
    // 3D affine.
    auto theta = torch::arange(1, 13, torch::kDouble)
                     .view(torch::IntArrayRef({1, 3, 4}));
    auto size = torch::IntArrayRef({1, 1, 3, 2, 2}).vec();
    auto align_corners = true;
    auto output = F::affine_grid(theta, size, !align_corners);
    auto expected = torch::tensor(
        {{{{{0.5000, -2.1667, -4.8333}, {1.5000, 2.8333, 4.1667}},
           {{2.5000, 3.8333, 5.1667}, {3.5000, 8.8333, 14.1667}}},
          {{{2.5000, 2.5000, 2.5000}, {3.5000, 7.5000, 11.5000}},
           {{4.5000, 8.5000, 12.5000}, {5.5000, 13.5000, 21.5000}}},
          {{{4.5000, 7.1667, 9.8333}, {5.5000, 12.1667, 18.8333}},
           {{6.5000, 13.1667, 19.8333}, {7.5000, 18.1667, 28.8333}}}}});
    auto output_aligned = F::affine_grid(theta, size, align_corners);
    auto expected_aligned =
        torch::tensor({{{{{-2.0, -10.0, -18.0}, {0.0, 0.0, 0.0}},
                         {{2.0, 2.0, 2.0}, {4.0, 12.0, 20.0}}},
                        {{{1.0, -3.0, -7.0}, {3.0, 7.0, 11.0}},
                         {{5.0, 9.0, 13.0}, {7.0, 19.0, 31.0}}},
                        {{{4.0, 4.0, 4.0}, {6.0, 14.0, 22.0}},
                         {{8.0, 16.0, 24.0}, {10.0, 26.0, 42.0}}}}});

    ASSERT_TRUE(output.allclose(expected, 1e-2));
    ASSERT_TRUE(output_aligned.allclose(expected_aligned));
  }
  {
    auto theta = torch::empty({1, 2, 3}, torch::kDouble);
    auto size = torch::IntArrayRef({1, 1, 2, 2}).vec();
    ASSERT_THROWS_WITH(
        F::affine_grid(torch::empty({2, 2, 3}), {-1, 1, 2, 2}),
        "Expected non-zero, positive output size. Got [-1, 1, 2, 2]");
    ASSERT_THROWS_WITH(
        F::affine_grid(torch::empty({2, 2, 3}, torch::kInt), size),
        "Expected theta to have floating point type, but got int");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta[0], size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [2, 3].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.unsqueeze(0), size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [1, 1, 2, 3].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 2, 1}), size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [1, 4, 3].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 1, 2}), size),
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
        "[1, 1, 2, 2]. Got [1, 2, 6].");
  }
  {
    auto theta = torch::empty({1, 3, 4}, torch::kDouble);
    auto size = torch::IntArrayRef({1, 1, 2, 2, 3}).vec();
    ASSERT_THROWS_WITH(
        F::affine_grid(theta[0], size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [3, 4].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.unsqueeze(0), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 1, 3, 4].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 2, 1}), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 6, 4].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta.repeat({1, 1, 2}), size),
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
        "[1, 1, 2, 2, 3]. Got [1, 3, 8].");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta, {1, 1, 1, 2, 2, 3}),
        "affine_grid only supports 4D and 5D sizes, for 2D and 3D affine "
        "transforms, respectively. Got size [1, 1, 1, 2, 2, 3]");
    ASSERT_THROWS_WITH(
        F::affine_grid(theta, {1, 1}),
        "affine_grid only supports 4D and 5D sizes, for 2D and 3D affine "
        "transforms, respectively. Got size [1, 1]");
  }
}

TEST_F(FunctionalTest, MultiMarginLoss) {
  auto weight = torch::tensor({0.3, 0.3, 0.4}, torch::kFloat);
  auto input = torch::tensor({{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}}, torch::requires_grad());
  auto target = torch::tensor({2, 1, 0}, torch::kLong);
  auto output = F::multi_margin_loss(
    input, target, MultiMarginLossOptions().margin(2).weight(weight));
  auto expected = torch::tensor({0.305556}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, CosineEmbeddingLoss) {
  auto input1 = torch::tensor({{2, 3, 4}, {6, 2, 4}});
  auto input2 = torch::tensor({{2, 3, 5}, {9, 12, 0}});
  auto target = torch::tensor({1, -1});
  auto output = F::cosine_embedding_loss(
      input1, input2, target, CosineEmbeddingLossOptions().margin(0.5));
  auto expected = torch::tensor({0.1004}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-4));
}

TEST_F(FunctionalTest, MaxUnpool1d) {
  auto x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto y = F::max_unpool1d(x, indices, MaxUnpool1dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 9}));

  x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(
      x, indices, MaxUnpool1dOptions(3), c10::IntArrayRef({1, 1, 9}));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 9}));

  x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(x, indices, MaxUnpool1dOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 5}));
}

TEST_F(FunctionalTest, MaxUnpool2d) {
  auto indices = torch::tensor({
  {{{ 6,  8,  9},
    {16, 18, 19},
    {21, 23, 24}}},
  {{{ 6,  8,  9},
    {16, 18, 19},
    {21, 23, 24}}}}, torch::kLong);
  auto x = torch::tensor({
  {{{ 6,  8,  9},
    {16, 18, 19},
    {21, 23, 24}}},
  {{{31, 33, 34},
    {41, 43, 44},
    {46, 48, 49}}}}, torch::requires_grad());
  auto y = F::max_unpool2d(x, indices, MaxUnpool2dOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::tensor(
   {{{{ 0,  0,  0,  0,  0},
      { 0,  6,  0,  8,  9},
      { 0,  0,  0,  0,  0},
      { 0, 16,  0, 18, 19},
      { 0, 21,  0, 23, 24}}},
    {{{ 0,  0,  0,  0,  0},
      { 0, 31,  0, 33, 34},
      { 0,  0,  0,  0,  0},
      { 0, 41,  0, 43, 44},
      { 0, 46,  0, 48, 49}}}} , torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 1, 5, 5}));
}

TEST_F(FunctionalTest, ELU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      auto y_exp = torch::max(torch::zeros_like(x), x) +
                torch::min(torch::zeros_like(x), alpha * (torch::exp(x) - 1.0));
      auto y = F::elu(x, ELUOptions().alpha(alpha).inplace(inplace));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      }
    }
  }
}

TEST_F(FunctionalTest, SELU) {
  {
    const double scale = 1.0507009873554804934193349852946;
    const double alpha = 1.6732632423543772848170429916717;
    for (const auto inplace : {false, true}) {
      auto input = torch::randn({5, 5});
      auto expected = scale *
          (torch::max(torch::zeros_like(input), input) +
           torch::min(
               torch::zeros_like(input), alpha * (torch::exp(input) - 1)));
      auto output = F::selu(input, inplace);

      ASSERT_TRUE(output.allclose(expected));
      if (inplace) {
        ASSERT_TRUE(input.allclose(expected));
      }
    }
  }
  {
    auto input = torch::arange(0, 9, torch::kDouble).view({3, 3});
    auto output = F::selu(input);
    auto expected = F::selu(input, false);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(FunctionalTest, Hardshrink) {
  const auto size = 3;
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = F::hardshrink(x, HardshrinkOptions().lambda(lambda));
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
    auto y_exp = (x.abs() > lambda) * x;
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(FunctionalTest, OneHot) {
  { // Test #1
    auto x = torch::arange(0, 5, torch::kLong);
    auto y = F::one_hot(x % 3);
    auto expected = torch::tensor(
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, torch::kLong);

    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({5, 3}));
  }

  { // Test #2
    auto x = torch::arange(0, 5, torch::kLong);
    auto y = F::one_hot(x % 3, 5);
    auto expected = torch::tensor(
        {{1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0}},
        torch::kLong);

    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({5, 5}));
  }

  { // Test #3
    auto x = torch::arange(0, 6, torch::kLong);
    auto y = F::one_hot(x.view(torch::IntArrayRef({3, 2})) % 3);
    auto expected = torch::tensor(
        {{{1, 0, 0}, {0, 1, 0}},
         {{0, 0, 1}, {1, 0, 0}},
         {{0, 1, 0}, {0, 0, 1}}},
        torch::kLong);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 2, 3}));
  }
}

TEST_F(FunctionalTest, Hardtanh) {
  const auto size = 3;
  for (const auto min_val : {-4.2, -1.0, -0.42, 0.0}) {
    for (const auto max_val : {0.0, 0.42, 1.0, 4.2}) {
      for (const auto inplace : {false, true}) {
        auto x = torch::linspace(-10.0, 10.0, size * size * size);
        x.resize_({size, size, size});
        auto y_exp = (x < min_val) * min_val +
                     ((x >= min_val) * (x <= max_val)) * x +
                     (x > max_val) * max_val;
        auto y = F::hardtanh(x,HardtanhOptions().min_val(min_val)
          .max_val(max_val).inplace(inplace));

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
        ASSERT_TRUE(torch::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
}

TEST_F(FunctionalTest, LeakyReLU) {
  const auto size = 3;
  for (const auto negative_slope : {0.0, 0.42, 1.0}) {
    for (const auto inplace : {false, true}) {
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      auto y_exp = (x < 0) * x * negative_slope + (x >= 0) * x;
      auto y = F::leaky_relu(x, LeakyReLUOptions()
        .negative_slope(negative_slope).inplace(inplace));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      }
    }
  }
}

TEST_F(FunctionalTest, LogSigmoid) {
  const auto size = 3;
  LogSigmoid model;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto y = F::logsigmoid(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
  auto y_exp = torch::log(torch::ones_like(x)/(torch::ones_like(x) + torch::exp(torch::neg(x))));
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(FunctionalTest, Softmax) {
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  auto output = F::softmax(input, /*dim=*/1);
  auto sum = torch::sum(torch::exp(input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::exp(input[i]) / sum[i];
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(FunctionalTest, PReLU) {
  const auto x = torch::rand({42, 24}) * 200 - 100;
  const auto w = torch::rand(24) * 200 - 100;
  const auto y = F::prelu(x, w);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({42, 24}));
  const auto y_exp = (x < 0) * w * x  + (x >= 0) * x;
  ASSERT_TRUE(torch::allclose(y, y_exp));
}
