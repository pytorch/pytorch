#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

namespace F = torch::nn::functional;

using namespace torch::nn;

struct FunctionalTest : torch::test::SeedingFixture {};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Conv1d) {
  auto x = torch::arange(30, torch::dtype(torch::kFloat).requires_grad(true)).reshape({2, 3, 5});
  auto weight = torch::arange(18, torch::dtype(torch::kFloat).requires_grad(true)).reshape({2, 3, 3});
  auto y = F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
  auto expected = torch::tensor({{{ 312.,  348.,  384.},
                                  { 798.,  915., 1032.}},

                                 {{ 852.,  888.,  924.},
                                  {2553., 2670., 2787.}}}, torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv1d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Conv2dEven) {
  auto x = torch::arange(75, torch::dtype(torch::kFloat).requires_grad(true)).reshape({1, 3, 5, 5});
  auto weight = torch::arange(54, torch::dtype(torch::kFloat).requires_grad(true)).reshape({2, 3, 3, 3});
  auto y = F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
  auto expected = torch::tensor({{{{15219., 15570., 15921.},
                                   {16974., 17325., 17676.},
                                   {18729., 19080., 19431.}},

                                  {{37818., 38898., 39978.},
                                   {43218., 44298., 45378.},
                                   {48618., 49698., 50778.}}}}, torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv2d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Conv2dUneven) {
  auto x = torch::arange(60, torch::dtype(torch::kFloat).requires_grad(true)).reshape({1, 3, 5, 4});
  auto weight = torch::arange(36, torch::dtype(torch::kFloat).requires_grad(true)).reshape({2, 3, 3, 2});
  auto y = F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
  auto expected = torch::tensor({{{{ 5289.,  5442.,  5595.},
                                   { 5901.,  6054.,  6207.},
                                   { 6513.,  6666.,  6819.}},

                                  {{13227., 13704., 14181.},
                                   {15135., 15612., 16089.},
                                   {17043., 17520., 17997.}}}}, torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv2d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Conv3d) {
  auto x = torch::arange(375, torch::dtype(torch::kFloat).requires_grad(true)).reshape({1, 3, 5, 5, 5});
  auto weight = torch::arange(162, torch::dtype(torch::kFloat).requires_grad(true)).reshape({2, 3, 3, 3, 3});
  auto y = F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
  auto expected = torch::tensor({{{{{ 700704.,  703944.,  707184.},
                                    { 716904.,  720144.,  723384.},
                                    { 733104.,  736344.,  739584.}},

                                   {{ 781704.,  784944.,  788184.},
                                    { 797904.,  801144.,  804384.},
                                    { 814104.,  817344.,  820584.}},

                                   {{ 862704.,  865944.,  869184.},
                                    { 878904.,  882144.,  885384.},
                                    { 895104.,  898344.,  901584.}}},


                                  {{{1724220., 1734021., 1743822.},
                                    {1773225., 1783026., 1792827.},
                                    {1822230., 1832031., 1841832.}},

                                   {{1969245., 1979046., 1988847.},
                                    {2018250., 2028051., 2037852.},
                                    {2067255., 2077056., 2086857.}},

                                   {{2214270., 2224071., 2233872.},
                                    {2263275., 2273076., 2282877.},
                                    {2312280., 2322081., 2331882.}}}}}, torch::kFloat);
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv3d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::max_pool1d(x, F::MaxPool1dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MaxPool2dBackward) {
  auto input = torch::rand({1, 2, 4, 4}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = F::max_pool2d(input, F::MaxPool2dFuncOptions(2));
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::max_pool3d(x, F::MaxPool3dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::avg_pool3d(x, F::AvgPool3dFuncOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, FractionalMaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::fractional_max_pool2d(x, F::FractionalMaxPool2dFuncOptions(3).output_size(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));

  auto y_with_indices = F::fractional_max_pool2d_with_indices(x, F::FractionalMaxPool2dFuncOptions(3).output_size(2));
  ASSERT_TRUE(torch::equal(y, std::get<0>(y_with_indices)));
  ASSERT_TRUE(torch::allclose(
    std::get<1>(y_with_indices),
    torch::tensor({{{ 0,  2},
                    {10, 12}},
                   {{ 0,  2},
                    {10, 12}}})));
  ASSERT_EQ(std::get<1>(y_with_indices).sizes(), std::vector<int64_t>({2, 2, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, FractionalMaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::fractional_max_pool3d(x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  auto y_with_indices = F::fractional_max_pool3d_with_indices(x, F::FractionalMaxPool3dFuncOptions(3).output_size(2));
  ASSERT_TRUE(torch::equal(y, std::get<0>(y_with_indices)));
  ASSERT_TRUE(torch::allclose(
    std::get<1>(y_with_indices),
    torch::tensor({{{{ 0,  2},
                     {10, 12}},
                    {{50, 52},
                     {60, 62}}},
                   {{{ 0,  2},
                     {10, 12}},
                    {{50, 52},
                     {60, 62}}}})));
  ASSERT_EQ(std::get<1>(y_with_indices).sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LPPool1d) {
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;

  auto x = torch::ones({1, 1, 5});
  auto y = F::lp_pool1d(x, F::LPPool1dFuncOptions(norm_type, kernel_size).stride(stride));
  auto expected = (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) * kernel_size).pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LPPool2d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});

  auto x = torch::ones({1, 2, 5});
  auto y = F::lp_pool2d(x, F::LPPool2dFuncOptions(norm_type, kernel_size).stride(stride));
  auto expected = (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) * (kernel_size[0] * kernel_size[1])).pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, CosineSimilarity) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output =
      F::cosine_similarity(input1, input2, F::CosineSimilarityFuncOptions().dim(1));
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SmoothL1LossDefaultOptions) {
  auto input = torch::tensor({0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      F::smooth_l1_loss(input, target);
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SmoothL1LossBeta) {
  auto input = torch::tensor({0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,bugprone-argument-comment)
      F::smooth_l1_loss(input, target, /*reduction=*/torch::kMean, /*beta=*/0.5);
  auto expected = torch::tensor(1.67, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SmoothL1LossNoReduction) {
  auto input = torch::tensor({0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      // NOLINTNEXTLINE(bugprone-argument-comment)
      F::smooth_l1_loss(input, target, /*reduction=*/torch::kNone);
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, HuberLossDefaultOptions) {
  auto input = torch::tensor({0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output =
      F::huber_loss(input, target);
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, HuberLossDelta) {
  auto input = torch::tensor({0.1, 1.5, 10.0}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto options = F::HuberLossFuncOptions().reduction(torch::kMean).delta(0.5);
  auto output = F::huber_loss(input, target, options);
  auto expected = torch::tensor(1.67 * 0.5, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, HuberLossNoReduction) {
  auto input = torch::tensor({0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto options = F::HuberLossFuncOptions().reduction(torch::kNone);
  auto output = F::huber_loss(input, target, options);
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  auto s = output.sum();
  s.backward();
  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(input.sizes() == input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SoftMarginLossDefaultOptions) {
  auto input = torch::tensor({2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  auto output =
      F::soft_margin_loss(input, target);
  auto expected = torch::tensor({1.3767317}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MultiLabelSoftMarginLossDefaultOptions) {
  auto input = torch::tensor({{0., 2., 2., 0.}, {2., 1., 0., 1.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  auto output =
      F::multilabel_soft_margin_loss(input, target);
  auto expected = torch::tensor({0.7608436}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SoftMarginLossNoReduction) {
  auto input = torch::tensor({2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  auto output =
      F::soft_margin_loss(input, target, torch::kNone);
  auto expected = torch::tensor({2.1269281, 0.01814993, 0.3132617, 3.0485873}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MultiLabelSoftMarginLossWeightedNoReduction) {
  auto input = torch::tensor({{0., 2., 2., 0.}, {2., 1., 0., 1.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  auto weight = torch::tensor({0.1, 0.6, 0.4, 0.8}, torch::kFloat);
  auto options = F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone).weight(weight);
  auto output =
      F::multilabel_soft_margin_loss(input, target, options);
  auto expected = torch::tensor({0.4876902, 0.3321295}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, PairwiseDistance) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output =
      F::pairwise_distance(input1, input2, F::PairwiseDistanceFuncOptions().p(1));
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AdaptiveMaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_max_pool1d(x, F::AdaptiveMaxPool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AdaptiveMaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_max_pool2d(x, F::AdaptiveMaxPool2dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AdaptiveMaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_max_pool3d(x, F::AdaptiveMaxPool3dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AdaptiveAvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AdaptiveAvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AdaptiveAvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_avg_pool3d(x, F::AdaptiveAvgPool3dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, L1Loss) {
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = F::l1_loss(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MSELoss) {
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = F::mse_loss(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BCELoss) {
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = F::binary_cross_entropy(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, KLDivLoss) {
  KLDivLoss loss;
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = F::kl_div(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, HingeEmbeddingLoss) {
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::kFloat);
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  auto output = F::hinge_embedding_loss(
      input, target, F::HingeEmbeddingLossFuncOptions().margin(2));
  auto expected = torch::tensor({10}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, GridSample) {
  auto input = torch::arange(9, torch::kFloat).view(std::vector<int64_t>({1, 1, 3, 3}));
  auto grid = torch::tensor({{
      {{-2., -1.}, {-1., -1.}, {0., -1.}},
      {{-1., 0.}, {0., 0.}, {1., 0.}},
      {{0., 1.}, {1., 1.}, {2., 1.}}
  }}, torch::kFloat);

  // bilinear, zeros, true
  auto options = F::GridSampleFuncOptions()
                    .mode(torch::kBilinear)
                    .padding_mode(torch::kZeros)
                    .align_corners(true);
  auto output = F::grid_sample(input, grid, options);
  auto expected = torch::tensor({{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bilinear, zeros, false
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor({{{{0., 0., 0.5}, {1.5, 4., 2.5}, {3.5, 2., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // default options (bilinear, zeros, false) same result as above
  output = F::grid_sample(input, grid);

  ASSERT_TRUE(output.allclose(expected));

  // nearest, zeros, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kNearest)
                .padding_mode(torch::kZeros)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor({{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 0.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bilinear, border, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor({{{{0., 0., 1.}, {3., 4., 5.}, {7., 8., 8.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));

  // bilinear, reflection, true
  options = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kReflection)
                .align_corners(true);
  output = F::grid_sample(input, grid, options);
  expected = torch::tensor({{{{1., 0., 1.}, {3., 4., 5.}, {7., 8., 7.}}}}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AffineGrid) {
  {
    // 2D affine.
    auto theta = torch::arange(1., 13)
                     .view(std::vector<int64_t>({2, 2, 3}));
    auto size = std::vector<int64_t>({2, 3, 2, 2});
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
    auto theta = torch::arange(1., 13)
                     .view(std::vector<int64_t>({1, 3, 4}));
    auto size = std::vector<int64_t>({1, 1, 3, 2, 2});
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
    auto size = std::vector<int64_t>({1, 1, 2, 2});
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
    auto size = std::vector<int64_t>({1, 1, 2, 2, 3});
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MultiMarginLoss) {
  auto weight = torch::tensor({0.3, 0.3, 0.4}, torch::kFloat);
  auto input = torch::tensor(
    {{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}},
    torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({2, 1, 0}, torch::kLong);
  auto output = F::multi_margin_loss(
    input, target, F::MultiMarginLossFuncOptions().margin(2).weight(weight));
  auto expected = torch::tensor({0.305556}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, CosineEmbeddingLoss) {
  auto input1 = torch::tensor({{2, 3, 4}, {6, 2, 4}});
  auto input2 = torch::tensor({{2, 3, 5}, {9, 12, 0}});
  auto target = torch::tensor({1, -1});
  auto output = F::cosine_embedding_loss(
      input1, input2, target, F::CosineEmbeddingLossFuncOptions().margin(0.5));
  auto expected = torch::tensor({0.1004}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-4));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MultiLabelMarginLossDefaultOptions) {
  auto input = torch::tensor({{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  auto output = F::multilabel_margin_loss(input, target);
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MultiLabelMarginLossNoReduction) {
  auto input = torch::tensor({{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  auto output = F::multilabel_margin_loss(
    input, target, torch::kNone);
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, TripletMarginLoss) {
  auto anchor = torch::tensor({{3., 3.}}, torch::kFloat);
  auto positive = torch::tensor({{2., 2.}}, torch::kFloat);
  auto negative = torch::tensor({{0., 0.}}, torch::kFloat);
  auto output = F::triplet_margin_loss(
      anchor, positive, negative, F::TripletMarginLossFuncOptions().margin(1.0));
  auto expected = torch::tensor({0.}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, TripletMarginWithDistanceLossDefaultParity) {
  // Check that if we use torch::pairwise_distance with the default
  // TripletMarginLoss options as our distance function, the outputs
  // are equal (i.e., equal under defaults).

  std::vector<TripletMarginWithDistanceLossOptions::reduction_t>
      reductions = {torch::kSum, torch::kMean, torch::kNone};
  std::vector<float> margins = {0.5, 1.0, 1.5};
  std::vector<bool> swaps = {true, false};

  for (auto& reduction : reductions) {
    for (auto& margin : margins) {
      for (const auto& swap : swaps) {
        auto anchor =
            torch::randn({100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        auto positive =
            torch::randn({100, 128}, torch::dtype(torch::kFloat).requires_grad(true));
        auto negative =
            torch::randn({100, 128}, torch::dtype(torch::kFloat).requires_grad(true));

        auto basicOptions = F::TripletMarginLossFuncOptions()
                                .reduction(reduction)
                                .margin(margin)
                                .swap(swap);
        auto distanceOptions =
            F::TripletMarginWithDistanceLossFuncOptions()
                .reduction(reduction)
                .margin(margin)
                .swap(swap);
        TripletMarginLoss basicLoss(basicOptions);
        TripletMarginWithDistanceLoss distanceLoss(distanceOptions);

        auto basicOutput =
            F::triplet_margin_loss(anchor, positive, negative, basicOptions);
        auto distanceOutput = F::triplet_margin_with_distance_loss(
            anchor, positive, negative, distanceOptions);

        ASSERT_TRUE(distanceOutput.allclose(basicOutput, 1e-6, 1e-6));

        // handle for torch::kNone reduction
        auto sum = distanceOutput.sum();
        sum.backward();
        ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
        ASSERT_EQ(positive.sizes(), positive.grad().sizes());
        ASSERT_EQ(negative.sizes(), negative.grad().sizes());
      }
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, NLLLoss) {
  auto input = torch::tensor({{-0.1315, -3.1315, -2.5315},
                              {-3.7038, -0.1038, -2.6038},
                              {-2.3422, -1.3422, -0.4422}},
                             torch::kFloat);
  auto target = torch::tensor({1, 0, 2}, torch::kLong);
  auto output = F::nll_loss(
      input, target, F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kMean));
  auto expected = torch::tensor(2.4258, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_TRUE(F::nll_loss(input, target).allclose(expected, 1e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, CrossEntropy) {
  auto input = torch::tensor({{3., 3.}, {2., 2.}}, torch::kFloat);
  auto target = torch::tensor({0, 1}, torch::kLong);
  auto output = F::cross_entropy(
      input, target, F::CrossEntropyFuncOptions().ignore_index(-100).reduction(torch::kMean));
  auto expected = torch::tensor(0.6931, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_TRUE(F::cross_entropy(input, target).allclose(expected, 1e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MaxUnpool1d) {
  auto x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto y = F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(
      x, indices, F::MaxUnpool1dFuncOptions(3).output_size(std::vector<int64_t>({1, 1, 9})));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(
      y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(x, indices, F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(
      torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 5}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
    {46, 48, 49}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = F::max_unpool2d(x, indices, F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));

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
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 1, 5, 5}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MaxUnpool3d) {
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  auto x = torch::tensor({{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(torch::allclose(y, torch::tensor(
   {{{{{ 0,  0,  0},
       { 0,  0,  0},
       { 0,  0,  0}},
      {{ 0,  0,  0},
       { 0,  0,  0},
       { 0,  0,  0}},
      {{ 0,  0,  0},
       { 0,  0,  0},
       { 0,  0, 26}}}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3, 3, 3}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ELU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      auto y_exp = torch::max(torch::zeros_like(x), x) +
                torch::min(torch::zeros_like(x), alpha * (torch::exp(x) - 1.0));
      auto y = F::elu(x, F::ELUFuncOptions().alpha(alpha).inplace(inplace));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      }
    }
  }
  ASSERT_TRUE(F::elu(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
  ASSERT_TRUE(F::selu(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, GLU) {
  int64_t dim = 1;
  auto input = torch::randn({4, 2}, torch::requires_grad());
  auto output = F::glu(input, dim);
  auto input_size = input.sizes()[dim] / 2;
  auto first_half = input.narrow(dim, 0, input_size);
  auto second_half = input.narrow(dim, input_size, input_size);
  auto expected = first_half * torch::sigmoid(second_half);

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_TRUE(F::glu(input).allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, GELU) {
  GELU model;
  const auto x = torch::linspace(-3.0, 3.0, 100);
  const auto y_exp = x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
  const auto y = F::gelu(x);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Hardshrink) {
  const auto size = 3;
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = F::hardshrink(x, F::HardshrinkFuncOptions().lambda(lambda));
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x.abs() > lambda) * x;
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, OneHot) {
  { // Test #1
    auto x = torch::arange(0, 5, torch::kLong);
    auto y = F::one_hot(x % 3);
    auto expected = torch::tensor(
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, torch::kLong);

    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({5, 3}));
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
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({5, 5}));
  }

  { // Test #3
    auto x = torch::arange(0, 6, torch::kLong);
    auto y = F::one_hot(x.view(std::vector<int64_t>({3, 2})) % 3);
    auto expected = torch::tensor(
        {{{1, 0, 0}, {0, 1, 0}},
         {{0, 0, 1}, {1, 0, 0}},
         {{0, 1, 0}, {0, 0, 1}}},
        torch::kLong);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_TRUE(torch::allclose(y, expected));
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({3, 2, 3}));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
        auto y = F::hardtanh(x, F::HardtanhFuncOptions().min_val(min_val)
          .max_val(max_val).inplace(inplace));

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(torch::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  ASSERT_TRUE(F::hardtanh(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LeakyReLU) {
  const auto size = 3;
  for (const auto negative_slope : {0.0, 0.42, 1.0}) {
    for (const auto inplace : {false, true}) {
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      auto y_exp = (x < 0) * x * negative_slope + (x >= 0) * x;
      auto y = F::leaky_relu(x, F::LeakyReLUFuncOptions()
        .negative_slope(negative_slope).inplace(inplace));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      }
    }
  }
  ASSERT_TRUE(F::leaky_relu(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LogSigmoid) {
  const auto size = 3;
  LogSigmoid model;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto y = F::logsigmoid(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = torch::log(torch::ones_like(x)/(torch::ones_like(x) + torch::exp(torch::neg(x))));
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, GumbelSoftmax) {
  // Test 1: No-options
  {
    auto logits = torch::randn({5});
    int expected_count = 1;
    auto y_draw = F::gumbel_softmax(logits);

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  // Test 2: 1D shape, 0 and -1 dim
  for(const auto dim: {0, -1}) {
    auto logits = torch::randn({5});
    int expected_count = 1;
    auto y_draw = F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(dim));

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  { // Test 3: 2D shape, 1 dim
    auto logits = torch::randn({5, 4});
    int expected_count = 5;
    auto y_draw = F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(1));

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  // Test 4: 3D shape, 1 and -1 dim
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int dims[] = {1, -1};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
  int expected[] = {5*3, 5*4};
  for(auto i=0; i<2; i++) {
    auto logits = torch::randn({5, 4, 3});
    int expected_count = expected[i];
    auto y_draw = F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(dims[i]));

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Shape unchanged
    ASSERT_TRUE(y_draw.sizes() == logits.sizes());
    // One choice per draw
    ASSERT_TRUE(torch::allclose(y_draw.sum(), torch::tensor(expected_count, torch::kFloat)));
  }

  { // Test 5: Straight through
    int num_draws = 100;
    auto logits = torch::tensor({{0.2, 0.8, 0.1}});
    logits = logits.reshape({1, 3});
    logits.requires_grad();
    auto probs = logits.softmax(-1);

    auto counts = torch::zeros_like(logits);
    torch::Tensor y_draw;
    for (auto i=0; i<num_draws; i++) {
        y_draw = F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true));
        counts += y_draw;
    }

    // All values positive
    ASSERT_GE(y_draw.min().item<int>(), 0);
    // Each experiment should result in 1 draw
    ASSERT_EQ(counts.sum().item<int>(), num_draws);

    // Check results are asymptotically as expected
    auto expected = probs * num_draws;
    // ~z is approximately N(0,1) for unbiased count
    auto z = (counts - expected) / (expected * (1 - probs)).sqrt();
    // A (lazy) approximate 99% two-sided test:
    // occurs with prob alpha~>=0.01 if unbiased
    ASSERT_LT(z.abs().max().item<float>(), 2.58);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Softmax) {
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto output = F::softmax(input, /*dim=*/1);
  auto sum = torch::sum(torch::exp(input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::exp(input[i]) / sum[i];
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Softmin) {
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto output = F::softmin(input, /*dim=*/1);
  auto sum = torch::sum(torch::exp(-input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::exp(-input[i]) / sum[i];
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LogSoftmax) {
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  // NOLINTNEXTLINE(bugprone-argument-comment)
  auto output = F::log_softmax(input, /*dim=*/1);
  auto sum = torch::sum(torch::exp(input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::log(torch::exp(input[i]) / sum[i]);
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, PReLU) {
  const auto x = torch::rand({42, 24}) * 200 - 100;
  const auto w = torch::rand(24) * 200 - 100;
  const auto y = F::prelu(x, w);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({42, 24}));
  const auto y_exp = (x < 0) * w * x  + (x >= 0) * x;
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LayerNorm) {
  const auto input = torch::randn({2, 2});
  auto y = F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
  auto y_exp = torch::layer_norm(input, {2, 2}, torch::Tensor(), torch::Tensor(), 2e-5);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, GroupNorm) {
  const auto input = torch::randn({2, 2});
  auto y = F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
  auto y_exp = torch::group_norm(input, 2, torch::Tensor(), torch::Tensor(), 2e-5);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, LocalResponseNorm) {
  const auto x = torch::arange(100, 118).resize_({3, 3, 2});
  const auto y = F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 2}));
  const auto y_exp = torch::tensor(
    {{{73.7788, 74.1462},
      {60.1942, 60.3302},
      {60.4609, 60.5865}},
    {{75.8729, 76.2011},
      {60.9331, 61.0390},
      {61.1403, 61.2370}},
    {{77.7387, 78.0303},
      {61.5011, 61.5807},
      {61.6563, 61.7279}}},
    torch::kFloat
  );
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Linear) {
  {
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});
    const auto w = torch::arange(200., 206).resize_({3, 2});
    const auto b = torch::arange(300., 303);
    const auto y = F::linear(x, w, b);
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 3}));
    const auto y_exp = torch::tensor(
      {{{40601, 41004, 41407},
        {41403, 41814, 42225},
        {42205, 42624, 43043}},
      {{43007, 43434, 43861},
        {43809, 44244, 44679},
        {44611, 45054, 45497}},
      {{45413, 45864, 46315},
        {46215, 46674, 47133},
        {47017, 47484, 47951}}},
      torch::kFloat
    );
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
  {
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});
    const auto w = torch::arange(200., 206).resize_({3, 2});
    const auto y = F::linear(x, w);
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 3}));
    const auto y_exp = torch::tensor(
      {{{40301, 40703, 41105},
        {41103, 41513, 41923},
        {41905, 42323, 42741}},
       {{42707, 43133, 43559},
        {43509, 43943, 44377},
        {44311, 44753, 45195}},
       {{45113, 45563, 46013},
        {45915, 46373, 46831},
        {46717, 47183, 47649}}},
      torch::kFloat
    );
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Embedding) {
  const auto input = torch::tensor({{1,2,4,5}, {4,3,2,9}}, torch::kLong);
  auto weight = torch::empty({10, 3});
  torch::nn::init::normal_(weight);
  auto y = F::embedding(input, weight);
  auto y_exp = torch::embedding(weight, input.contiguous(), -1, false, false);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, EmbeddingBag) {
  const auto input = torch::tensor({1,2,4,5,4,3,2,9}, torch::kLong);
  auto offsets = torch::tensor({0,4}, torch::kLong);
  auto weight = torch::empty({10, 3});
  torch::nn::init::normal_(weight);
  auto y = F::embedding_bag(input, weight, F::EmbeddingBagFuncOptions().mode(torch::kSum).offsets(offsets).padding_idx(4));
  auto y_exp = std::get<0>(torch::embedding_bag(weight, input, offsets, false, 0, false, torch::Tensor(), false, 4));
  ASSERT_TRUE(torch::allclose(y, y_exp));

  // no options test
  const auto input_ = torch::tensor({{1,2,4,5}, {4,3,2,9}}, torch::kLong);
  auto offsets_ = torch::arange(0, input_.numel(), input_.size(1), torch::TensorOptions().dtype(torch::kLong).device(input.device()));
  y = F::embedding_bag(input_, weight);
  y_exp = std::get<0>(torch::embedding_bag(weight, input_.reshape(-1), offsets_, false, 1, false, torch::Tensor()));
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Bilinear) {
  auto input1 = torch::tensor({{1, 2, 3}, {7, 6, 5}});
  auto input2 = torch::tensor({{7, 4}, {8 ,9}});
  auto weight = torch::tensor({{{2, 3}, {9, 7}, {8, 6}}});
  auto bias = torch::tensor({1});

  auto y_with_bias = F::bilinear(input1, input2, weight, bias);
  ASSERT_EQ(y_with_bias.ndimension(), 2);
  ASSERT_EQ(y_with_bias.sizes(), torch::IntArrayRef({2, 1}));
  auto y_with_bias_exp = torch::tensor({{449}, {1702}}).reshape({2, 1});
  ASSERT_TRUE(torch::allclose(y_with_bias, y_with_bias_exp, 1e-4, 1e-7));

  auto y_no_bias = F::bilinear(input1, input2, weight);
  ASSERT_EQ(y_no_bias.ndimension(), 2);
  ASSERT_EQ(y_no_bias.sizes(), torch::IntArrayRef({2, 1}));
  auto y_no_bias_exp = torch::tensor({{448, 1701}}).reshape({2, 1});
  ASSERT_TRUE(torch::allclose(y_no_bias, y_no_bias_exp, 1e-4, 1e-7));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Normalize) {
  const auto expected = torch::tensor(
    {{{0.00000000, 0.10000000, 0.2000, 0.30000000, 0.40000000},
      {0.14285715, 0.17142858, 0.2000, 0.22857143, 0.25714287}}}, torch::requires_grad().dtype(torch::kFloat));
  { // Test #1
    auto input = torch::tensor({{{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}}, torch::dtype(torch::kFloat).requires_grad(true));
    auto norm = F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));

    // reduce to scalar to call .backward()
    torch::Tensor s = norm.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(input.grad().numel(), 10);
    ASSERT_TRUE(torch::allclose(norm, expected));
  }

  { // Test #2 Check variations of optional arguments
    auto input = torch::tensor({{{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}}, torch::dtype(torch::kFloat));
    auto output = torch::randn({1,2,5}, torch::dtype(torch::kFloat));
    // non-null output argument
    F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1).out(output));
    // default options
    F::normalize(input);

    ASSERT_TRUE(torch::allclose(output, expected));
  }

  { // Test #3 Base case of scalar tensor
    auto input = torch::randn({}, torch::requires_grad());
    torch::Tensor norm = F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
    norm.backward();

    ASSERT_EQ(input.grad().numel(), 1);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ReLU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    auto y_exp = (x < 0) * 0 + (x >= 0) * x;
    auto y = F::relu(x, F::ReLUFuncOptions().inplace(inplace));

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    ASSERT_TRUE(torch::allclose(y, y_exp));
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }

    // NOLINTNEXTLINE(bugprone-argument-comment)
    y = F::relu(x, /*inplace=*/inplace);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    ASSERT_TRUE(torch::allclose(y, y_exp));
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }
  }
  ASSERT_TRUE(F::relu(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ReLUDefaultOptions) {
  const auto size = 3;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto y_exp = (x < 0) * 0 + (x >= 0) * x;
  auto y = F::relu(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ReLU6) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    auto y_exp = (x < 0) * 0 + ((x >= 0) * (x <= 6)) * x + (x > 6) * 6;
    auto y = F::relu6(x, F::ReLU6FuncOptions().inplace(inplace));

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    ASSERT_TRUE(torch::allclose(y, y_exp));
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }

    // NOLINTNEXTLINE(bugprone-argument-comment)
    y = F::relu6(x, /*inplace=*/inplace);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    ASSERT_TRUE(torch::allclose(y, y_exp));
    if (inplace) {
      ASSERT_TRUE(torch::allclose(x, y_exp));
    }
  }
  ASSERT_TRUE(F::relu6(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ReLU6DefaultOptions) {
  const auto size = 3;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto y_exp = (x < 0) * 0 + ((x >= 0) * (x <= 6)) * x + (x > 6) * 6;
  auto y = F::relu6(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, RReLU) {
  const auto size = 3;
  for (const auto lower : {0.01, 0.1, 0.2}) {
    for (const auto upper : {0.3, 0.4, 0.5}) {
      for (const auto inplace : {false, true}) {
        auto x = torch::linspace(-10.0, 10.0, size * size * size);
        x.resize_({size, size, size});
        auto x_copy = x.clone();
        auto y = F::rrelu(x, F::RReLUFuncOptions().lower(lower)
          .upper(upper).inplace(inplace));
        auto z = ((x_copy >= 0) * (x_copy == y) +
          (x_copy < 0) * (y >= x_copy * upper) * (y <= lower * x_copy)) * 1.0;

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(torch::allclose(z, torch::ones_like(z)));
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y));
        }
      }
    }
  }
  ASSERT_TRUE(F::rrelu(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, RReLUDefaultOptions) {
  const auto size = 3;
  const auto lower = 1.0 / 8.0;
  const auto upper = 1.0 / 3.0;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto x_copy = x.clone();
  auto y = F::rrelu(x);
  auto z = ((x_copy >= 0) * (x_copy == y) +
    (x_copy < 0) * (y >= x_copy * upper) * (y <= lower * x_copy)) * 1.0;

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  ASSERT_TRUE(torch::allclose(z, torch::ones_like(z)));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, CELU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.42, 1.0, 4.2, 42.42}) {
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      auto y_exp = torch::max(torch::zeros_like(x), x) +
        torch::min(torch::zeros_like(x), alpha * (torch::exp(x / alpha) - 1.0));
      auto y = F::celu(x, F::CELUFuncOptions().alpha(alpha).inplace(inplace));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(x, y_exp));
      }
    }
  }
  ASSERT_TRUE(F::celu(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, CELUDefaultOptions) {
  const auto size = 3;
  const auto alpha = 1.0;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size});
  auto y_exp = torch::max(torch::zeros_like(x), x) +
    torch::min(torch::zeros_like(x), alpha * (torch::exp(x / alpha) - 1.0));
  auto y = F::celu(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, PixelShuffle) {
  auto x = torch::tensor(
    {{{{-17, 19}, {-1, 2}},
      {{7, 14}, {-3, 1}},
      {{0, -2}, {-12, 14}},
      {{-15, 0}, {-3, 9}}}}, torch::kFloat);
  auto y_exp = torch::tensor(
    {{{{-17, 7, 19, 14},
       {0, -15, -2, 0},
       {-1, -3, 2, 1},
       {-12, -3, 14, 9}}}}, torch::kFloat);
  auto y = F::pixel_shuffle(x, 2);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 4, 4}));
  ASSERT_TRUE(y.allclose(y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, PixelUnshuffle) {
  auto x = torch::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      torch::kFloat);
  auto y_exp = torch::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      torch::kFloat);
  auto y = F::pixel_unshuffle(x, 2);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 4, 2, 2}));
  ASSERT_TRUE(y.allclose(y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Softplus) {
  const auto size = 3;
  for (const auto beta : {0.5, 1.0, 2.0}) {
    for (const auto threshold : {1.0, 3.0, 5.0}) {
      auto x = torch::linspace(-3.0, 3.0, 61);
      x.resize_({size, size, size});
      auto y_exp =
        (x <= threshold) * torch::log(1 + torch::exp(x * beta)) / beta +
        (x > threshold) * x;
      auto y = F::softplus(x,
        F::SoftplusFuncOptions().beta(beta).threshold(threshold));

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SoftplusDefaultOptions) {
  const auto size = 3;
  const auto beta = 1.0;
  const auto threshold = 20.0;
  auto x = torch::linspace(-3.0, 3.0, 61);
  x.resize_({size, size, size});
  auto y_exp =
    (x <= threshold) * torch::log(1 + torch::exp(x * beta)) / beta +
    (x > threshold) * x;
  auto y = F::softplus(x);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Fold) {
  auto input = torch::ones({1, 3 * 2 * 2, 2}, torch::kDouble);
  auto output = F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
  auto expected = torch::tensor(
      {{{{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
        {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
        {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}}}},
      torch::kDouble);

  ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 3, 2}));
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Unfold) {
  auto input = torch::arange(0, 12, torch::kDouble).view({1, 2, 2, 3});
  auto output = F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
  auto expected = torch::tensor(
      {{{0.0, 0.0, 0.0, 4.0},
        {0.0, 0.0, 3.0, 5.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 10.0},
        {0.0, 0.0, 9.0, 11.0},
        {0.0, 7.0, 0.0, 0.0},
        {6.0, 8.0, 0.0, 0.0}}},
      torch::kDouble);

  ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 8, 4}));
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Softshrink) {
  const auto size = 3;
  for (const auto lambda : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    // NOLINTNEXTLINE(bugprone-argument-comment)
    auto y = F::softshrink(x, /*lambda=*/lambda);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, SoftshrinkDefaultOptions) {
  const auto size = 3;
  const auto lambda = 0.5;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  auto y = F::softshrink(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Softsign) {
  auto x = torch::randn(100) * 10;
  auto y_exp = x / (1 + x.abs());
  auto y = F::softsign(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Mish) {
  auto x = torch::randn(100) * 10;
  auto y_exp = x * x.exp().log1p().tanh();
  auto y = F::mish(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Tanhshrink) {
  auto x = torch::randn(100) * 10;
  auto y_exp = x - x.tanh();
  auto y = F::tanhshrink(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Threshold) {
  const auto size = 3;
  for (const auto threshold : {0.5, 1.0, 2.0}) {
    for (const auto value : {0.5, 1.0, 2.0}) {
      for (const auto inplace : {false, true}) {
        auto x = torch::linspace(-3.0, 3.0, 61);
        x.resize_({size, size, size});
        auto y_exp = (x <= threshold) * value + (x > threshold) * x;
        auto y = F::threshold(x,
          F::ThresholdFuncOptions(threshold, value).inplace(inplace));

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(torch::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(torch::allclose(x, y_exp));
        }
      }
    }
  }
  ASSERT_TRUE(F::threshold(torch::tensor(1.), F::ThresholdFuncOptions(0.5, 0.5)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BatchNorm1d) {
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  auto input = torch::randn({2, 5});
  auto mean = torch::randn(5);
  auto variance = torch::rand(5);
  auto weight = torch::ones({num_features});
  auto bias = torch::zeros({num_features});
  auto output = F::batch_norm(
    input, mean, variance,
    F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(momentum).eps(eps).training(false));
  auto expected = (input - mean) / torch::sqrt(variance + eps);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BatchNorm1dDefaultOptions) {
  auto input = torch::randn({2, 5});
  auto mean = torch::randn(5);
  auto variance = torch::rand(5);
  auto output = F::batch_norm(input, mean, variance);
  auto expected = (input - mean) / torch::sqrt(variance + 1e-5);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BatchNorm2d) {
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  auto input = torch::randn({2, num_features, 4, 4});
  auto mean = torch::randn(num_features);
  auto variance = torch::rand(num_features);
  auto weight = torch::ones({num_features});
  auto bias = torch::zeros({num_features});
  auto output = F::batch_norm(
    input, mean, variance,
    F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(momentum).eps(eps).training(false));
  auto expected = torch::transpose((torch::transpose(input, 1, 3) - mean) / torch::sqrt(variance + eps), 1, 3);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BatchNorm2dDefaultOptions) {
  int num_features = 5;
  double eps = 1e-05;

  auto input = torch::randn({2, num_features, 4, 4});
  auto mean = torch::randn(num_features);
  auto variance = torch::rand(num_features);
  auto output = F::batch_norm(input, mean, variance);
  auto expected = torch::transpose((torch::transpose(input, 1, 3) - mean) / torch::sqrt(variance + eps), 1, 3);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BatchNorm3d) {
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  auto input = torch::randn({2, num_features, 2, 2, 2});
  auto mean = torch::randn(num_features);
  auto variance = torch::rand(num_features);
  auto weight = torch::ones({num_features});
  auto bias = torch::zeros({num_features});
  auto output = F::batch_norm(
    input, mean, variance,
    F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(momentum).eps(eps).training(false));
  auto expected = torch::transpose((torch::transpose(input, 1, 4) - mean) / torch::sqrt(variance + eps), 1, 4);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BatchNorm3dDefaultOptions) {
  int num_features = 5;
  double eps = 1e-05;

  auto input = torch::randn({2, num_features, 2, 2, 2});
  auto mean = torch::randn(num_features);
  auto variance = torch::rand(num_features);
  auto output = F::batch_norm(input, mean, variance);
  auto expected = torch::transpose((torch::transpose(input, 1, 4) - mean) / torch::sqrt(variance + eps), 1, 4);
  ASSERT_TRUE(output.allclose(expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, InstanceNorm1d) {
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  auto input = torch::arange(40.).view({2, 5, 4});
  auto mean = torch::arange(5.);
  auto variance = torch::arange(5.);
  auto weight = torch::arange((double)num_features);
  auto bias = torch::arange((double)num_features);
  auto output = F::instance_norm(
    input,
    F::InstanceNormFuncOptions()
      .running_mean(mean)
      .running_var(variance)
      .weight(weight)
      .bias(bias)
      .momentum(momentum)
      .eps(eps));
  auto expected = torch::tensor({{{ 0.0000,  0.0000,  0.0000,  0.0000},
                                  {-0.3416,  0.5528,  1.4472,  2.3416},
                                  {-0.6833,  1.1056,  2.8944,  4.6833},
                                  {-1.0249,  1.6584,  4.3416,  7.0249},
                                  {-1.3665,  2.2112,  5.7888,  9.3665}},
                                 {{ 0.0000,  0.0000,  0.0000,  0.0000},
                                  {-0.3416,  0.5528,  1.4472,  2.3416},
                                  {-0.6833,  1.1056,  2.8944,  4.6833},
                                  {-1.0249,  1.6584,  4.3416,  7.0249},
                                  {-1.3665,  2.2112,  5.7888,  9.3665}}});
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, InstanceNorm1dDefaultOptions) {
  auto input = torch::arange(40.).view({2, 5, 4});
  auto output = F::instance_norm(input);
  auto expected = torch::tensor({{{-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416}},
                                 {{-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416},
                                  {-1.3416, -0.4472,  0.4472,  1.3416}}});
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, InstanceNorm2d) {
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  auto input = torch::arange(2. * num_features * 2 * 2).view({2, num_features, 2, 2});
  auto mean = torch::arange((double)num_features);
  auto variance = torch::arange((double)num_features);
  auto weight = torch::arange((double)num_features);
  auto bias = torch::arange((double)num_features);
  auto output = F::instance_norm(
    input,
    F::InstanceNormFuncOptions()
      .running_mean(mean)
      .running_var(variance)
      .weight(weight)
      .bias(bias)
      .momentum(momentum)
      .eps(eps));
  auto expected = torch::tensor({{{{ 0.0000,  0.0000},
                                   { 0.0000,  0.0000}},
                                  {{-0.3416,  0.5528},
                                   { 1.4472,  2.3416}},
                                  {{-0.6833,  1.1056},
                                   { 2.8944,  4.6833}},
                                  {{-1.0249,  1.6584},
                                   { 4.3416,  7.0249}},
                                  {{-1.3665,  2.2112},
                                   { 5.7888,  9.3665}}},
                                 {{{ 0.0000,  0.0000},
                                   { 0.0000,  0.0000}},
                                  {{-0.3416,  0.5528},
                                   { 1.4472,  2.3416}},
                                  {{-0.6833,  1.1056},
                                   { 2.8944,  4.6833}},
                                  {{-1.0249,  1.6584},
                                   { 4.3416,  7.0249}},
                                  {{-1.3665,  2.2112},
                                   { 5.7888,  9.3665}}}});
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, InstanceNorm2dDefaultOptions) {
  int num_features = 5;
  double eps = 1e-05;

  auto input = torch::arange(2. * num_features * 2 * 2).view({2, num_features, 2, 2});
  auto output = F::instance_norm(input);
  auto expected = torch::tensor({{{{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}}},
                                 {{{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}},
                                  {{-1.3416, -0.4472},
                                   { 0.4472,  1.3416}}}});
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, InstanceNorm3d) {
  int num_features = 5;
  double eps = 1e-05;
  double momentum = 0.1;

  auto input = torch::arange(2. * num_features * 2 * 2 * 2).view({2, num_features, 2, 2, 2});
  auto mean = torch::arange((double)num_features);
  auto variance = torch::arange((double)num_features);
  auto weight = torch::arange((double)num_features);
  auto bias = torch::arange((double)num_features);
  auto output = F::instance_norm(
    input,
    F::InstanceNormFuncOptions()
      .running_mean(mean)
      .running_var(variance)
      .weight(weight)
      .bias(bias)
      .momentum(momentum)
      .eps(eps));
  auto expected = torch::tensor({{{{{ 0.0000,  0.0000},
                                    { 0.0000,  0.0000}},
                                   {{ 0.0000,  0.0000},
                                    { 0.0000,  0.0000}}},
                                  {{{-0.5275, -0.0911},
                                    { 0.3453,  0.7818}},
                                   {{ 1.2182,  1.6547},
                                    { 2.0911,  2.5275}}},
                                  {{{-1.0550, -0.1822},
                                    { 0.6907,  1.5636}},
                                   {{ 2.4364,  3.3093},
                                    { 4.1822,  5.0550}}},
                                  {{{-1.5826, -0.2733},
                                    { 1.0360,  2.3453}},
                                   {{ 3.6547,  4.9640},
                                    { 6.2733,  7.5826}}},
                                  {{{-2.1101, -0.3644},
                                    { 1.3814,  3.1271}},
                                   {{ 4.8729,  6.6186},
                                    { 8.3644, 10.1101}}}},
                                 {{{{ 0.0000,  0.0000},
                                    { 0.0000,  0.0000}},
                                   {{ 0.0000,  0.0000},
                                    { 0.0000,  0.0000}}},
                                  {{{-0.5275, -0.0911},
                                    { 0.3453,  0.7818}},
                                   {{ 1.2182,  1.6547},
                                    { 2.0911,  2.5275}}},
                                  {{{-1.0550, -0.1822},
                                    { 0.6907,  1.5636}},
                                   {{ 2.4364,  3.3093},
                                    { 4.1822,  5.0550}}},
                                  {{{-1.5826, -0.2733},
                                    { 1.0360,  2.3453}},
                                   {{ 3.6547,  4.9640},
                                    { 6.2733,  7.5826}}},
                                  {{{-2.1101, -0.3644},
                                    { 1.3814,  3.1271}},
                                   {{ 4.8729,  6.6186},
                                    { 8.3644, 10.1101}}}}});
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, InstanceNorm3dDefaultOptions) {
  int num_features = 5;
  double eps = 1e-05;

  auto input = torch::arange(2. * num_features * 2 * 2 * 2).view({2, num_features, 2, 2, 2});
  auto output = F::instance_norm(input);
  auto expected = torch::tensor({{{{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}}},
                                 {{{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}},
                                  {{{-1.5275, -1.0911},
                                    {-0.6547, -0.2182}},
                                   {{ 0.2182,  0.6547},
                                    { 1.0911,  1.5275}}}}});
  ASSERT_TRUE(output.allclose(expected, 2e-04));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Interpolate) {
  {
    // 1D interpolation
    auto input = torch::ones({1, 1, 2});
    auto options = F::InterpolateFuncOptions()
                       .size(std::vector<int64_t>({4}))
                       .mode(torch::kNearest);
    auto output = F::interpolate(input, options);
    auto expected = torch::ones({1, 1, 4});

    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // 2D interpolation
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        auto input = torch::ones({1, 1, 2, 2});
        auto options = F::InterpolateFuncOptions()
                           .scale_factor(std::vector<double>({scale_factor, scale_factor}))
                           .mode(torch::kBilinear)
                           .align_corners(align_corners);
        auto output = F::interpolate(input, options);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = torch::ones({1, 1, expected_size, expected_size});

        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    // 3D interpolation
    for (const auto align_corners : {true, false}) {
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        auto input = torch::ones({1, 1, 2, 2, 2});
        auto options =
            F::InterpolateFuncOptions()
                .scale_factor(std::vector<double>({scale_factor, scale_factor, scale_factor}))
                .mode(torch::kTrilinear)
                .align_corners(align_corners);
        auto output = F::interpolate(input, options);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected =
            torch::ones({1, 1, expected_size, expected_size, expected_size});

        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    auto input = torch::randn({3, 2, 2});
    ASSERT_THROWS_WITH(
        F::interpolate(input[0], F::InterpolateFuncOptions().size(std::vector<int64_t>({4, 4}))),
        "Input Error: Only 3D, 4D and 5D input Tensors supported (got 2D) "
        "for the modes: nearest | linear | bilinear | bicubic | trilinear (got kNearest)");
    ASSERT_THROWS_WITH(
        F::interpolate(
            torch::reshape(input, {1, 1, 1, 3, 2, 2}),
            F::InterpolateFuncOptions().size(std::vector<int64_t>({1, 1, 1, 3, 4, 4}))),
        "Input Error: Only 3D, 4D and 5D input Tensors supported (got 6D) "
        "for the modes: nearest | linear | bilinear | bicubic | trilinear (got kNearest)");
    ASSERT_THROWS_WITH(
        F::interpolate(input, F::InterpolateFuncOptions()),
        "either size or scale_factor should be defined");
    ASSERT_THROWS_WITH(
        F::interpolate(
            input,
            F::InterpolateFuncOptions().size(std::vector<int64_t>({3, 4, 4})).scale_factor(std::vector<double>({0.5}))),
        "only one of size or scale_factor should be defined");
    ASSERT_THROWS_WITH(
        F::interpolate(input, F::InterpolateFuncOptions().scale_factor(std::vector<double>({3, 2}))),
        "scale_factor shape must match input shape. "
        "Input is 1D, scale_factor size is [3, 2]");
    ASSERT_THROWS_WITH(
        F::interpolate(
            input,
            F::InterpolateFuncOptions()
                .mode(torch::kNearest)
                .align_corners(true)),
        "align_corners option can only be set with the "
        "interpolating modes: linear | bilinear | bicubic | trilinear");
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Pad) {
  {
    auto input = torch::arange(6, torch::kDouble).reshape({1, 2, 3});
    auto output = F::pad(input, F::PadFuncOptions({1, 2}).mode(torch::kCircular));
    auto expected = torch::tensor({{{2., 0., 1., 2., 0., 1.},
                                    {5., 3., 4., 5., 3., 4.}}}, torch::kDouble);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 2, 6}));
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
  {
    auto input = torch::arange(9, torch::kDouble).reshape({1, 1, 3, 3});
    auto output = F::pad(input, F::PadFuncOptions({3, 3, 3, 1}).mode(torch::kCircular));
    auto expected = torch::tensor(
       {{{{0., 1., 2., 0., 1., 2., 0., 1., 2.},
          {3., 4., 5., 3., 4., 5., 3., 4., 5.},
          {6., 7., 8., 6., 7., 8., 6., 7., 8.},
          {0., 1., 2., 0., 1., 2., 0., 1., 2.},
          {3., 4., 5., 3., 4., 5., 3., 4., 5.},
          {6., 7., 8., 6., 7., 8., 6., 7., 8.},
          {0., 1., 2., 0., 1., 2., 0., 1., 2.}}}}, torch::kDouble);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 7, 9}));
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
  {
    auto input = torch::arange(12, torch::kDouble).reshape({1, 1, 2, 2, 3});
    auto output = F::pad(input, F::PadFuncOptions({3, 3, 2, 1, 2, 2}).mode(torch::kCircular));
    auto expected = torch::tensor(
       {{{{{ 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.},
           { 3.,  4.,  5.,  3.,  4.,  5.,  3.,  4.,  5.},
           { 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.},
           { 3.,  4.,  5.,  3.,  4.,  5.,  3.,  4.,  5.},
           { 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.}},

          {{ 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.},
           { 9., 10., 11.,  9., 10., 11.,  9., 10., 11.},
           { 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.},
           { 9., 10., 11.,  9., 10., 11.,  9., 10., 11.},
           { 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.}},

          {{ 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.},
           { 3.,  4.,  5.,  3.,  4.,  5.,  3.,  4.,  5.},
           { 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.},
           { 3.,  4.,  5.,  3.,  4.,  5.,  3.,  4.,  5.},
           { 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.}},

          {{ 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.},
           { 9., 10., 11.,  9., 10., 11.,  9., 10., 11.},
           { 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.},
           { 9., 10., 11.,  9., 10., 11.,  9., 10., 11.},
           { 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.}},

          {{ 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.},
           { 3.,  4.,  5.,  3.,  4.,  5.,  3.,  4.,  5.},
           { 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.},
           { 3.,  4.,  5.,  3.,  4.,  5.,  3.,  4.,  5.},
           { 0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.}},

          {{ 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.},
           { 9., 10., 11.,  9., 10., 11.,  9., 10., 11.},
           { 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.},
           { 9., 10., 11.,  9., 10., 11.,  9., 10., 11.},
           { 6.,  7.,  8.,  6.,  7.,  8.,  6.,  7.,  8.}}}}}, torch::kDouble);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 6, 5, 9}));
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
  {
    auto input = torch::arange(16, torch::kDouble).reshape({2, 2, 2, 2});
    auto output = F::pad(input, F::PadFuncOptions({1, 1, 1, 1}).mode(torch::kReflect));
    auto expected = torch::tensor(
       {{{{ 3.,  2.,  3.,  2.},
          { 1.,  0.,  1.,  0.},
          { 3.,  2.,  3.,  2.},
          { 1.,  0.,  1.,  0.}},

         {{ 7.,  6.,  7.,  6.},
          { 5.,  4.,  5.,  4.},
          { 7.,  6.,  7.,  6.},
          { 5.,  4.,  5.,  4.}}},

        {{{11., 10., 11., 10.},
          { 9.,  8.,  9.,  8.},
          {11., 10., 11., 10.},
          { 9.,  8.,  9.,  8.}},

         {{15., 14., 15., 14.},
          {13., 12., 13., 12.},
          {15., 14., 15., 14.},
          {13., 12., 13., 12.}}}}, torch::kDouble);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({2, 2, 4, 4}));
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
  {
    auto input = torch::arange(12, torch::kDouble).reshape({1, 1, 2, 2, 3});
    auto output = F::pad(input, F::PadFuncOptions({1, 2, 2, 1, 1, 2}).mode(torch::kReplicate));
    auto expected = torch::tensor(
       {{{{{ 0.,  0.,  1.,  2.,  2.,  2.},
           { 0.,  0.,  1.,  2.,  2.,  2.},
           { 0.,  0.,  1.,  2.,  2.,  2.},
           { 3.,  3.,  4.,  5.,  5.,  5.},
           { 3.,  3.,  4.,  5.,  5.,  5.}},

          {{ 0.,  0.,  1.,  2.,  2.,  2.},
           { 0.,  0.,  1.,  2.,  2.,  2.},
           { 0.,  0.,  1.,  2.,  2.,  2.},
           { 3.,  3.,  4.,  5.,  5.,  5.},
           { 3.,  3.,  4.,  5.,  5.,  5.}},

          {{ 6.,  6.,  7.,  8.,  8.,  8.},
           { 6.,  6.,  7.,  8.,  8.,  8.},
           { 6.,  6.,  7.,  8.,  8.,  8.},
           { 9.,  9., 10., 11., 11., 11.},
           { 9.,  9., 10., 11., 11., 11.}},

          {{ 6.,  6.,  7.,  8.,  8.,  8.},
           { 6.,  6.,  7.,  8.,  8.,  8.},
           { 6.,  6.,  7.,  8.,  8.,  8.},
           { 9.,  9., 10., 11., 11., 11.},
           { 9.,  9., 10., 11., 11., 11.}},

          {{ 6.,  6.,  7.,  8.,  8.,  8.},
           { 6.,  6.,  7.,  8.,  8.,  8.},
           { 6.,  6.,  7.,  8.,  8.,  8.},
           { 9.,  9., 10., 11., 11., 11.},
           { 9.,  9., 10., 11., 11., 11.}}}}}, torch::kDouble);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 5, 5, 6}));
    ASSERT_TRUE(output.allclose(expected, 1e-04));
  }
  {
    auto input = torch::ones({1, 1, 1, 1}, torch::kDouble);
    auto output = F::pad(input, F::PadFuncOptions({1, 1}).mode(torch::kConstant).value(0));
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 1, 3}));
    auto expected = torch::tensor({{{{0., 1., 0.}}}}, torch::kDouble);
  }
  {
    auto input = torch::ones({1, 1, 1, 1}, torch::kDouble);
    auto output = F::pad(input, F::PadFuncOptions({1, 1}));
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 1, 3}));
    auto expected = torch::tensor({{{{0., 1., 0.}}}}, torch::kDouble);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, CTCLoss) {
  { // test CTCLoss typechecks
    const auto target_lengths = torch::tensor({30, 25, 20});
    const auto input_lengths = torch::tensor({50, 50, 50});
    const auto targets =
      torch::randint(1, 15, {target_lengths.sum().item<int>()}, torch::kInt);
    const auto log_probs = torch::randn({50, 3, 15}, torch::kFloat).log_softmax(2);

    const auto _input_lengths = input_lengths.to(torch::kFloat);
    ASSERT_THROWS_WITH(
      F::ctc_loss(
        log_probs, targets, _input_lengths, target_lengths),
        "input_lengths must be integral");

    const auto target_lengths_ = target_lengths.to(torch::kFloat);
    ASSERT_THROWS_WITH(
      F::ctc_loss(
        log_probs, targets, input_lengths, target_lengths_),
        "target_lengths must be integral");
  }
  { // test CTCLoss length checks
    const auto target_lengths = torch::tensor({30, 25, 20});
    const auto input_lengths = torch::tensor({50, 50, 50});
    const auto targets = torch::randint(1, 15, {3, 29}, torch::kInt);
    const auto log_probs = torch::randn({50, 3, 15}, torch::kFloat)
      .log_softmax(2);
    ASSERT_THROWS_WITH(
      F::ctc_loss(
        log_probs, targets, input_lengths, target_lengths),
        "Expected tensor to have size at least 30 at dimension 1");
  }
  { // test CTCLoss empty target
    {
      const auto target_lengths = torch::tensor({0, 0, 0});
      const auto input_lengths = torch::tensor({50, 50, 50});
      const auto targets =
        torch::randint(1, 15, at::IntArrayRef({0}), torch::kLong);
      const auto log_probs =
        torch::randn({50, 3, 15}, torch::kDouble).log_softmax(2);
      const auto loss = F::ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        F::CTCLossFuncOptions().reduction(torch::kNone));
      ASSERT_TRUE(loss.ge(0).all().item<bool>());
      ASSERT_TRUE(torch::allclose(
        -log_probs.sum(0).slice(1, 0, 1).view_as(loss), loss));
    }
    {
      const auto target_lengths = torch::tensor({0, 9, 0});
      const auto input_lengths = torch::tensor({50, 50, 50});
      const auto targets = torch::randint(1, 15, {9}, torch::kLong);
      const auto log_probs =
        torch::randn({50, 3, 15}, torch::kDouble).log_softmax(2);
      const auto loss = F::ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        F::CTCLossFuncOptions().reduction(torch::kNone));
      ASSERT_TRUE(loss.ge(0).all().item<bool>());
      ASSERT_TRUE(torch::allclose(
          -log_probs.sum(0)
            .index_select(0, torch::tensor({0, 2}, torch::kLong))
            .slice(1, 0, 1).view({2}),
          loss.index_select(0, torch::tensor({0, 2}, torch::kLong))
      ));
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, PoissonNLLLoss) {
  const auto input = torch::tensor({0.5, 1.5, 2.5});
  const auto target = torch::tensor({1., 2., 3.});
  const auto component_wise_loss = torch::exp(input) - target * input;
  ASSERT_TRUE(torch::allclose(torch::mean(component_wise_loss),
    F::poisson_nll_loss(input, target)));
  ASSERT_TRUE(torch::allclose(component_wise_loss,
    F::poisson_nll_loss(input, target,
    F::PoissonNLLLossFuncOptions().reduction(torch::kNone))));
  ASSERT_TRUE(torch::allclose(torch::sum(component_wise_loss),
    F::poisson_nll_loss(input, target,
    F::PoissonNLLLossFuncOptions().reduction(torch::kSum))));
  ASSERT_TRUE(torch::allclose(torch::mean(component_wise_loss),
    F::poisson_nll_loss(input, target,
    F::PoissonNLLLossFuncOptions().reduction(torch::kMean))));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, MarginRankingLoss) {
  {
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    const auto target = torch::randn(15).sign();
    ASSERT_TRUE(torch::allclose(
      F::margin_ranking_loss(input1, input2, target),
      (-target * (input1 - input2)).clamp(0).mean()
    ));
  }
  {
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    const auto target = torch::randn(15).sign();
    const auto margin = 0.5;
    ASSERT_TRUE(torch::allclose(
      F::margin_ranking_loss(input1, input2, target,
        F::MarginRankingLossFuncOptions().margin(0.5).reduction(torch::kSum)
      ),
      (-target * (input1 - input2) + margin).clamp(0).sum()
    ));
  }
  {
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    const auto target = torch::randn(15).sign();
    const auto margin = 0.5;
    ASSERT_TRUE(torch::allclose(
      F::margin_ranking_loss(input1, input2, target,
        F::MarginRankingLossFuncOptions().margin(0.5).reduction(torch::kMean)
      ),
      (-target * (input1 - input2) + margin).clamp(0).mean()
    ));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ConvTranspose1d) {
  auto x = torch::arange(20.).view({2, 2, 5});
  auto weight = torch::arange(18.).view({2, 3, 3});
  auto y = F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
  auto expected = torch::tensor({{{  45.,  104.,  179.,  212.,  245.,  188.,  107.},
                                  {  60.,  140.,  242.,  293.,  344.,  260.,  146.},
                                  {  75.,  176.,  305.,  374.,  443.,  332.,  185.}},
                                 {{ 135.,  304.,  509.,  542.,  575.,  428.,  237.},
                                  { 210.,  460.,  752.,  803.,  854.,  620.,  336.},
                                  { 285.,  616.,  995., 1064., 1133.,  812.,  435.}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv_transpose1d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ConvTranspose2dEven) {
  auto x = torch::arange(50.).view({1, 2, 5, 5});
  auto weight = torch::arange(54.).view({2, 3, 3, 3});
  auto y = F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
  auto expected = torch::tensor({{{{  675.,  1402.,  2183.,  2270.,  2357.,  1634.,   849.},
                                   { 1560.,  3240.,  5044.,  5236.,  5428.,  3760.,  1952.},
                                   { 2685.,  5574.,  8673.,  8988.,  9303.,  6438.,  3339.},
                                   { 3180.,  6594., 10248., 10563., 10878.,  7518.,  3894.},
                                   { 3675.,  7614., 11823., 12138., 12453.,  8598.,  4449.},
                                   { 2820.,  5832.,  9040.,  9268.,  9496.,  6544.,  3380.},
                                   { 1605.,  3314.,  5129.,  5252.,  5375.,  3698.,  1907.}},
                                  {{  900.,  1870.,  2912.,  3053.,  3194.,  2210.,  1146.},
                                   { 2100.,  4356.,  6772.,  7072.,  7372.,  5092.,  2636.},
                                   { 3630.,  7518., 11670., 12147., 12624.,  8706.,  4500.},
                                   { 4395.,  9078., 14055., 14532., 15009., 10326.,  5325.},
                                   { 5160., 10638., 16440., 16917., 17394., 11946.,  6150.},
                                   { 3900.,  8028., 12388., 12724., 13060.,  8956.,  4604.},
                                   { 2190.,  4502.,  6938.,  7115.,  7292.,  4994.,  2564.}},
                                  {{ 1125.,  2338.,  3641.,  3836.,  4031.,  2786.,  1443.},
                                   { 2640.,  5472.,  8500.,  8908.,  9316.,  6424.,  3320.},
                                   { 4575.,  9462., 14667., 15306., 15945., 10974.,  5661.},
                                   { 5610., 11562., 17862., 18501., 19140., 13134.,  6756.},
                                   { 6645., 13662., 21057., 21696., 22335., 15294.,  7851.},
                                   { 4980., 10224., 15736., 16180., 16624., 11368.,  5828.},
                                   { 2775.,  5690.,  8747.,  8978.,  9209.,  6290.,  3221.}}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv_transpose2d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ConvTranspose2dUneven) {
  auto x = torch::arange(40.).view({1, 2, 5, 4});
  auto weight = torch::arange(36.).view({2, 3, 3, 2});
  auto y = F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
  auto expected = torch::tensor({{{{ 360.,  758.,  796.,  834.,  440.},
                                   { 832., 1752., 1836., 1920., 1012.},
                                   {1432., 3014., 3152., 3290., 1732.},
                                   {1696., 3566., 3704., 3842., 2020.},
                                   {1960., 4118., 4256., 4394., 2308.},
                                   {1504., 3152., 3252., 3352., 1756.},
                                   { 856., 1790., 1844., 1898.,  992.}},
                                  {{ 480., 1010., 1072., 1134.,  596.},
                                   {1120., 2352., 2484., 2616., 1372.},
                                   {1936., 4058., 4268., 4478., 2344.},
                                   {2344., 4898., 5108., 5318., 2776.},
                                   {2752., 5738., 5948., 6158., 3208.},
                                   {2080., 4328., 4476., 4624., 2404.},
                                   {1168., 2426., 2504., 2582., 1340.}},
                                  {{ 600., 1262., 1348., 1434.,  752.},
                                   {1408., 2952., 3132., 3312., 1732.},
                                   {2440., 5102., 5384., 5666., 2956.},
                                   {2992., 6230., 6512., 6794., 3532.},
                                   {3544., 7358., 7640., 7922., 4108.},
                                   {2656., 5504., 5700., 5896., 3052.},
                                   {1480., 3062., 3164., 3266., 1688.}}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv_transpose2d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, ConvTranspose3d) {
  auto x = torch::arange(16.).view({1, 2, 2, 2, 2});
  auto weight = torch::arange(32.).view({2, 2, 2, 2, 2});
  auto y = F::conv_transpose3d(x, weight, F::ConvTranspose3dFuncOptions().stride(1));
  auto expected = torch::tensor({{{{{ 128.,  280.,  154.},
                                    { 304.,  664.,  364.},
                                    { 184.,  400.,  218.}},
                                   {{ 352.,  768.,  420.},
                                    { 832., 1808.,  984.},
                                    { 496., 1072.,  580.}},
                                   {{ 256.,  552.,  298.},
                                    { 592., 1272.,  684.},
                                    { 344.,  736.,  394.}}},
                                  {{{ 192.,  424.,  234.},
                                    { 464., 1016.,  556.},
                                    { 280.,  608.,  330.}},
                                   {{ 544., 1184.,  644.},
                                    {1280., 2768., 1496.},
                                    { 752., 1616.,  868.}},
                                   {{ 384.,  824.,  442.},
                                    { 880., 1880., 1004.},
                                    { 504., 1072.,  570.}}}}});
  ASSERT_TRUE(torch::allclose(y, expected));

  auto y_no_options = F::conv_transpose3d(x, weight);
  ASSERT_TRUE(torch::allclose(y_no_options, expected));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AlphaDropout) {
  auto input = torch::randn(5000);
  auto input_mean = input.mean();
  auto input_std = input.std();

  for (const auto rate : {0.2, 0.5, 0.8}) {
    for (const auto inplace : {false, true}) {
      auto input_ = input.clone();
      auto output = F::alpha_dropout(input_, F::AlphaDropoutFuncOptions().p(rate).training(false).inplace(inplace));
      ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
      ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(input_, output));
      }
    }
  }
  auto output = F::detail::alpha_dropout(input, 0.5, false, false);
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
  ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, FeatureAlphaDropout) {
  auto input = torch::randn(5000);
  auto input_mean = input.mean();
  auto input_std = input.std();

  for (const auto rate : {0.2, 0.5, 0.8}) {
    for (const auto inplace : {false, true}) {
      auto input_ = input.clone();
      auto output = F::feature_alpha_dropout(input_, F::FeatureAlphaDropoutFuncOptions().p(rate).training(false).inplace(inplace));
      ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
      ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
      if (inplace) {
        ASSERT_TRUE(torch::allclose(input_, output));
      }
    }
  }
  auto output = F::feature_alpha_dropout(input);
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.1));
  ASSERT_TRUE(torch::allclose(input_std, output.std(), 0.1));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Dropout) {
  auto input = torch::randn(5000);
  auto input_mean = input.mean();
  auto input_std = input.std();

  for (const auto rate : {0.2, 0.5, 0.8}) {
    auto output = F::dropout(input, F::DropoutFuncOptions().p(rate));
    ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
    ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  }
  auto output = F::dropout(input);
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  ASSERT_TRUE(F::dropout(torch::tensor(1.)).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Dropout2d) {
  auto input = torch::randn({50, 100});
  auto input_mean = input.mean();
  auto input_std = input.std();

  for (const auto rate : {0.2, 0.5, 0.8}) {
    auto output = F::dropout2d(input, F::Dropout2dFuncOptions().p(rate));
    ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
    ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  }
  auto output = F::dropout2d(input);
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  ASSERT_TRUE(F::dropout2d(torch::randn({50, 100})).defined());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, Dropout3d) {
  auto input = torch::randn({50, 10, 10});
  auto input_mean = input.mean();
  auto input_std = input.std();

  for (const auto rate : {0.2, 0.5, 0.8}) {
    auto output = F::dropout3d(input, F::Dropout3dFuncOptions().p(rate));
    ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
    ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  }
  auto output = F::dropout3d(input);
  ASSERT_TRUE(torch::allclose(input_mean, output.mean(), 0.01, 0.05));
  ASSERT_TRUE((input_std <= output.std()).all().item<bool>());
  ASSERT_TRUE(F::dropout3d(torch::randn({50, 100})).defined());
}

template<c10::ScalarType S, typename T>
void test_isfinite(const at::Device& device) {
  const std::vector<T> values = {
    std::numeric_limits<T>::lowest(),
    0, 1, 42,
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max()
  };
  for (const auto value : values) {
    const auto x = torch::full({3, 3}, value, torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::isfinite(x).all().template item<bool>());
  }
  if (std::numeric_limits<T>::has_infinity) {
    const auto inf = std::numeric_limits<T>::infinity();
    const auto x = torch::tensor({
      -inf,
      std::numeric_limits<T>::lowest(),
      static_cast<T>(0),
      static_cast<T>(1),
      static_cast<T>(42),
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(),
      inf
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(
      // torch::allclose does not support comparing torch::kBool
      torch::isfinite(x).toType(torch::kInt),
      torch::tensor(
        {false, true, true, true, true, true, true, false},
        torch::TensorOptions().device(device)
      ).toType(torch::kInt)
    ));
  }
  if (std::numeric_limits<T>::has_quiet_NaN) {
    const auto x = torch::tensor({
      std::numeric_limits<T>::quiet_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isfinite(x).all().template item<bool>());
  }
  if (std::numeric_limits<T>::has_signaling_NaN) {
    const auto x = torch::tensor({
      std::numeric_limits<T>::signaling_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isfinite(x).all().template item<bool>());
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, isfinite) {
  const at::Device device("cpu");
  test_isfinite<torch::kUInt8, uint8_t>(device);
  test_isfinite<torch::kInt8, int8_t>(device);
  test_isfinite<torch::kInt16, int16_t>(device);
  test_isfinite<torch::kInt32, int32_t>(device);
  test_isfinite<torch::kInt64, int64_t>(device);
  test_isfinite<torch::kFloat32, float>(device);
  test_isfinite<torch::kFloat64, double>(device);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, isfinite_CUDA) {
  const at::Device device("cuda");
  test_isfinite<torch::kUInt8, uint8_t>(device);
  test_isfinite<torch::kInt8, int8_t>(device);
  test_isfinite<torch::kInt16, int16_t>(device);
  test_isfinite<torch::kInt32, int32_t>(device);
  test_isfinite<torch::kInt64, int64_t>(device);
  test_isfinite<torch::kFloat32, float>(device);
  test_isfinite<torch::kFloat64, double>(device);
  test_isfinite<torch::kFloat16, c10::Half>(device);
}

template<c10::ScalarType S, typename T>
void test_isinf(const at::Device& device) {
  const std::vector<T> values = {
    std::numeric_limits<T>::lowest(),
    0, 1, 42,
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max()
  };
  for (const auto value : values) {
    const auto x = torch::full({3, 3}, value, torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isinf(x).all().template item<bool>());
  }
  if (std::numeric_limits<T>::has_infinity) {
    const auto inf = std::numeric_limits<T>::infinity();
    const auto x = torch::tensor({
      -inf,
      std::numeric_limits<T>::lowest(),
      static_cast<T>(0),
      static_cast<T>(1),
      static_cast<T>(42),
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(),
      inf
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(
      // torch::allclose does not support comparing torch::kBool
      torch::isinf(x).toType(torch::kInt),
      torch::tensor(
        {true, false, false, false, false, false, false, true},
        torch::TensorOptions().device(device)
      ).toType(torch::kInt)
    ));
  }
  if (std::numeric_limits<T>::has_quiet_NaN) {
    const auto x = torch::tensor({
      std::numeric_limits<T>::quiet_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isinf(x).all().template item<bool>());
  }
  if (std::numeric_limits<T>::has_signaling_NaN) {
    const auto x = torch::tensor({
      std::numeric_limits<T>::signaling_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_FALSE(torch::isinf(x).all().template item<bool>());
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, isinf) {
  const at::Device device("cpu");
  test_isinf<torch::kUInt8, uint8_t>(device);
  test_isinf<torch::kInt8, int8_t>(device);
  test_isinf<torch::kInt16, int16_t>(device);
  test_isinf<torch::kInt32, int32_t>(device);
  test_isinf<torch::kInt64, int64_t>(device);
  test_isinf<torch::kFloat32, float>(device);
  test_isinf<torch::kFloat64, double>(device);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, isinf_CUDA) {
  const at::Device device("cuda");
  test_isinf<torch::kUInt8, uint8_t>(device);
  test_isinf<torch::kInt8, int8_t>(device);
  test_isinf<torch::kInt16, int16_t>(device);
  test_isinf<torch::kInt32, int32_t>(device);
  test_isinf<torch::kInt64, int64_t>(device);
  test_isinf<torch::kFloat32, float>(device);
  test_isinf<torch::kFloat64, double>(device);
  test_isinf<torch::kFloat16, c10::Half>(device);
}

template<c10::ScalarType S, typename T>
void test_allclose(const at::Device& device) {
  const std::vector<T> values = {
    std::numeric_limits<T>::lowest(),
    0, 1, 42,
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max()
  };
  for (const auto value : values) {
    const auto x = torch::full({1}, value, torch::TensorOptions().dtype(S).device(device));
    const auto y = torch::full({1}, value, torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(x, x));
    ASSERT_TRUE(torch::allclose(x, y));
    ASSERT_TRUE(torch::allclose(y, x));
    ASSERT_FALSE(torch::allclose(1.1 * x + 0.1, 1.0 * x));
    ASSERT_TRUE(torch::allclose(0.99 * x + 0.1, 1.0 * x, 1.1, 0.1));
  }
  if (std::numeric_limits<T>::has_infinity) {
    const auto inf = std::numeric_limits<T>::infinity();
    const auto x = torch::tensor({-inf, inf},
      torch::TensorOptions().dtype(S).device(device));
    const auto y = torch::tensor({-inf, inf},
      torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(x, x));
    ASSERT_TRUE(torch::allclose(x, y));
    ASSERT_TRUE(torch::allclose(y, x));
  }
  if (std::numeric_limits<T>::has_quiet_NaN) {
    const auto x = torch::tensor({
      std::numeric_limits<T>::quiet_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    const auto y = torch::tensor({
      std::numeric_limits<T>::quiet_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(x, x, 1.0, 0.0, /*equal_nan=*/true));
    ASSERT_TRUE(torch::allclose(x, y, 1.0, 0.0, /*equal_nan=*/true));
    ASSERT_TRUE(torch::allclose(y, x, 1.0, 0.0, /*equal_nan=*/true));
  }
  if (std::numeric_limits<T>::has_signaling_NaN) {
    const auto x = torch::tensor({
      std::numeric_limits<T>::signaling_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    const auto y = torch::tensor({
      std::numeric_limits<T>::signaling_NaN()
    }, torch::TensorOptions().dtype(S).device(device));
    ASSERT_TRUE(torch::allclose(x, x, 1.0, 0.0, /*equal_nan=*/true));
    ASSERT_TRUE(torch::allclose(x, y, 1.0, 0.0, /*equal_nan=*/true));
    ASSERT_TRUE(torch::allclose(y, x, 1.0, 0.0, /*equal_nan=*/true));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AllClose) {
  const at::Device device("cpu");
  test_allclose<torch::kUInt8, uint8_t>(device);
  test_allclose<torch::kInt8, int8_t>(device);
  test_allclose<torch::kInt16, int16_t>(device);
  test_allclose<torch::kInt32, int32_t>(device);
  test_allclose<torch::kInt64, int64_t>(device);
  test_allclose<torch::kFloat32, float>(device);
  test_allclose<torch::kFloat64, double>(device);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, AllClose_CUDA) {
  const at::Device device("cuda");
  test_allclose<torch::kUInt8, uint8_t>(device);
  test_allclose<torch::kInt8, int8_t>(device);
  test_allclose<torch::kInt16, int16_t>(device);
  test_allclose<torch::kInt32, int32_t>(device);
  test_allclose<torch::kInt64, int64_t>(device);
  test_allclose<torch::kFloat32, float>(device);
  test_allclose<torch::kFloat64, double>(device);
  test_allclose<torch::kFloat16, c10::Half>(device);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(FunctionalTest, BCEWithLogitsLoss) {
  { // test BCE with logits raises if target and input are different size
    {
      const auto target = torch::rand(5);
      const auto input = torch::rand({5, 1});
      ASSERT_THROWS_WITH(
        F::binary_cross_entropy_with_logits(input, target),
        "must be the same as input size"
      );
    }
    {
      const auto target = torch::rand({5, 1});
      const auto input = torch::rand(5);
      ASSERT_THROWS_WITH(
        F::binary_cross_entropy_with_logits(input, target),
        "must be the same as input size"
      );
    }
  }
  { // test BCE with logits gives same result as sigmoid and bce loss
    auto sigmoid = Sigmoid();

    auto target = torch::rand({64, 4});
    auto output = torch::rand({64, 4}) - 0.5;

    ASSERT_TRUE(torch::allclose(
      F::binary_cross_entropy_with_logits(output, target),
      F::binary_cross_entropy(sigmoid(output), target)
    ));

    auto weight = torch::rand(4);
    ASSERT_TRUE(torch::allclose(
      F::binary_cross_entropy_with_logits(output, target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)
      ),
      F::binary_cross_entropy(sigmoid(output), target,
        F::BinaryCrossEntropyFuncOptions().weight(weight)
      )
    ));

    target = torch::zeros({4, 1}, torch::kFloat);
    output = torch::empty({4, 1}, torch::kFloat).fill_(-100);

    ASSERT_TRUE(torch::allclose(
      F::binary_cross_entropy_with_logits(output, target),
      F::binary_cross_entropy(sigmoid(output), target)
    ));

    ASSERT_TRUE(torch::allclose(
      F::binary_cross_entropy_with_logits(output, target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone)
      ),
      F::binary_cross_entropy(sigmoid(output), target,
        F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone)
      )
    ));

    weight = torch::rand({1}, torch::kFloat);
    ASSERT_TRUE(torch::allclose(
      F::binary_cross_entropy_with_logits(output, target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)
      ),
      F::binary_cross_entropy(sigmoid(output), target,
        F::BinaryCrossEntropyFuncOptions().weight(weight)
      )
    ));
  }
  { // test BCE with logits has correct grad at zero
    const auto output = torch::zeros({3, 1}, torch::requires_grad());
    const auto target = torch::zeros({3, 1});
    F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum)
    ).backward();
    const auto expected_grad = torch::empty({3, 1}).fill_(0.5);
    ASSERT_TRUE(torch::allclose(output.grad(), expected_grad));
  }
  { // test BCE with logits broadcasts weights
    const auto target = torch::rand({16, 4});
    const auto output = torch::rand({16, 4}) - 0.5;

    auto weight = torch::rand(4);
    auto out1 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)
    );

    weight = weight.expand({16, 4}).contiguous();
    auto out2 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)
    );

    ASSERT_TRUE(torch::allclose(out1, out2));

    weight = torch::rand({16, 1});
    out1 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)
    );

    weight = weight.expand({16, 4}).contiguous();
    out2 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().weight(weight)
    );

    ASSERT_TRUE(torch::allclose(out1, out2));
  }
  { // test BCE with logits ones in pos weights are the same as none
    const auto target = torch::rand({64, 4});
    const auto output = torch::rand({64, 4}) - 0.5;
    const auto pos_weight = torch::ones({64, 4});

    ASSERT_TRUE(torch::allclose(
      F::binary_cross_entropy_with_logits(output, target),
      F::binary_cross_entropy_with_logits(output, target,
        F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight)
      )
    ));
  }
  { // test BCE with logits broadcasts pos weights
    const auto target = torch::rand({64, 4});
    const auto output = torch::rand({64, 4}) - 0.5;
    const auto pos_weight = torch::rand(4);
    const auto out1 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight)
    );

    const auto pos_weight1 = pos_weight.expand({1, 4});
    const auto out2 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight)
    );

    const auto pos_weight2 = pos_weight.expand({64, 4});
    const auto out3 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight)
    );

    ASSERT_TRUE(torch::allclose(out1, out2));
    ASSERT_TRUE(torch::allclose(out1, out3));
  }
  { // test BCE with logits with pos weight has correct grad at zero
    const auto output = torch::zeros({3, 1}, torch::requires_grad());
    const auto target = torch::zeros({3, 1});
    const auto pos_weight = torch::ones({3, 1});
    F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight).reduction(torch::kSum)
    ).backward();
    const auto expected_grad = torch::empty({3, 1}).fill_(0.5);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const auto grad = output.grad();
    ASSERT_TRUE(torch::allclose(grad, expected_grad));
  }
  { // test BCE with logits stability
    const auto output = torch::tensor({0., -120.});
    const auto target = torch::tensor({0., 1.});
    const auto pos_weight = torch::tensor({1., 1.});

    const auto out1 = F::binary_cross_entropy_with_logits(output, target);
    ASSERT_TRUE(torch::isfinite(out1).all().item<bool>());

    const auto out2 = F::binary_cross_entropy_with_logits(output, target,
      F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight)
    );
    ASSERT_TRUE(torch::isfinite(out2).all().item<bool>());
  }
}
