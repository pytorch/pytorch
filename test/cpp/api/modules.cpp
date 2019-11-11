#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

class TestModel : public torch::nn::Module {
 public:
  TestModel()
      : l1(register_module("l1", Linear(10, 3))),
        l2(register_module("l2", Linear(3, 5))),
        l3(register_module("l3", Linear(5, 100))) {}

  Linear l1, l2, l3;
};

class NestedModel : public torch::nn::Module {
 public:
  NestedModel()
      : param_(register_parameter("param", torch::empty({3, 2, 21}))),
        l1(register_module("l1", Linear(5, 20))),
        t(register_module("test", std::make_shared<TestModel>())) {}

  torch::Tensor param_;
  Linear l1;
  std::shared_ptr<TestModel> t;
};

struct ModulesTest : torch::test::SeedingFixture {};

TEST_F(ModulesTest, Conv1d) {
  Conv1d model(Conv1dOptions(3, 2, 3).stride(2));
  auto x = torch::randn({2, 3, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(s.ndimension(), 0);
  for (auto i = 0; i < 3; i++) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3);
}

TEST_F(ModulesTest, Conv2dEven) {
  Conv2d model(Conv2dOptions(3, 2, 3).stride(2));
  auto x = torch::randn({2, 3, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(s.ndimension(), 0);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 3);
}

TEST_F(ModulesTest, Conv2dUneven) {
  Conv2d model(Conv2dOptions(3, 2, {3, 2}).stride({2, 2}));
  auto x = torch::randn({2, 3, 5, 4}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(s.ndimension(), 0);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, Conv3d) {
  Conv3d model(Conv3dOptions(3, 2, 3).stride(2));
  auto x = torch::randn({2, 3, 5, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 5);
  ASSERT_EQ(s.ndimension(), 0);
  for (auto i = 0; i < 5; i++) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_TRUE(model->weight.grad().numel() == 3 * 2 * 3 * 3 * 3);
}

TEST_F(ModulesTest, MaxPool1d) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1 ,2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool1dReturnIndices) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1 ,2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));

  ASSERT_TRUE(torch::allclose(indices, torch::tensor({{{0, 2}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool2dEven) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2 ,2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dUneven) {
  MaxPool2d model(MaxPool2dOptions({3, 2}).stride({2, 2}));
  auto x = torch::ones({2, 5, 4}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dReturnIndices) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2 ,2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
  ASSERT_TRUE(torch::allclose(
    indices,
    torch::tensor({{{ 0,  2},
                    {10, 12}},
                   {{ 0,  2},
                    {10, 12}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3d) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3dReturnIndices) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  ASSERT_TRUE(torch::allclose(
    indices,
    torch::tensor({{{{ 0,  2},
                     {10, 12}},
                    {{50, 52},
                     {60, 62}}},
                   {{{ 0,  2},
                     {10, 12}},
                    {{50, 52},
                     {60, 62}}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool1d) {
  AvgPool1d model(AvgPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, AvgPool2dEven) {
  AvgPool2d model(AvgPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool2dUneven) {
  AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
  auto x = torch::ones({2, 5, 4}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool3d) {
  AvgPool3d model(AvgPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, LPPool1d) {
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;

  LPPool1d model(LPPool1dOptions(norm_type, kernel_size).stride(stride));
  auto x = torch::ones({1, 1, 5});
  auto y = model(x);
  auto expected = (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) * kernel_size).pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(ModulesTest, LPPool2d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});

  LPPool2d model(LPPool2dOptions(norm_type, kernel_size).stride(stride));
  auto x = torch::ones({1, 2, 5});
  auto y = model(x);
  auto expected = (torch::pow(torch::tensor({{{1, 1}}}, torch::kFloat), norm_type) * (kernel_size[0] * kernel_size[1])).pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, expected));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(ModulesTest, Identity) {
  Identity identity;
  auto input = torch::tensor({{1, 3, 4}, {2, 3, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = identity->forward(input);
  auto expected = torch::tensor({{1, 3, 4}, {2, 3, 4}}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, Flatten) {
  Flatten flatten;
  auto input = torch::tensor({{1, 3, 4}, {2, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = flatten->forward(input);
  auto expected = torch::tensor({{1, 3, 4}, {2, 5, 6}}, torch::kFloat);
  auto s = output.sum();

  s.backward();
  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));

  // Testing with optional arguments start_dim and end_dim
  Flatten flatten_optional_dims(FlattenOptions().start_dim(2).end_dim(3));
  input = torch::tensor({
    {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
    {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}
   }, torch::dtype(torch::kFloat).requires_grad(true)); // Tensor with sizes (2, 2, 2, 2)

  output = flatten_optional_dims->forward(input);
  expected = torch::tensor({
    {{1, 2, 3, 4}, {5, 6, 7, 8}},
    {{9, 10, 11, 12}, {13, 14, 15, 16}}
   }, torch::kFloat); // Tensor with sizes (2, 2, 4)

  s = output.sum();
  s.backward();
  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, AdaptiveMaxPool1d) {
  AdaptiveMaxPool1d model(3);
  auto x = torch::tensor({{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool1dReturnIndices) {
  AdaptiveMaxPool1d model(3);
  auto x = torch::tensor({{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({{{1, 3, 4}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dEven) {
  AdaptiveMaxPool2d model(3);
  auto x = torch::arange(0, 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{6, 8, 9},
     {16, 18, 19},
     {21, 23, 24}},
    {{31, 33, 34},
     {41, 43, 44},
     {46, 48, 49}},
  }, torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dUneven) {
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  auto x = torch::arange(0, 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{5, 7},
     {13, 15},
     {17, 19}},
    {{25, 27},
     {33, 35},
     {37, 39}},
  }, torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesEven) {
  AdaptiveMaxPool2d model(3);
  auto x = torch::arange(0, 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{6, 8, 9},
     {16, 18, 19},
     {21, 23, 24}},
    {{31, 33, 34},
     {41, 43, 44},
     {46, 48, 49}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({
    {{6, 8, 9},
     {16, 18, 19},
     {21, 23, 24}},
    {{6, 8, 9},
     {16, 18, 19},
     {21, 23, 24}},
  }, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesUneven) {
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  auto x = torch::arange(0, 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{5, 7},
     {13, 15},
     {17, 19}},
    {{25, 27},
     {33, 35},
     {37, 39}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({
    {{5, 7},
     {13, 15},
     {17, 19}},
    {{5, 7},
     {13, 15},
     {17, 19}},
  }, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3d) {
  AdaptiveMaxPool3d model(3);
  auto x = torch::arange(0, 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{21, 22, 23},
     {25, 26, 27},
     {29, 30, 31}},
    {{37, 38, 39},
     {41, 42, 43},
     {45, 46, 47}},
    {{53, 54, 55},
     {57, 58, 59},
     {61, 62, 63}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3dReturnIndices) {
  AdaptiveMaxPool3d model(3);
  auto x = torch::arange(0, 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{21, 22, 23},
     {25, 26, 27},
     {29, 30, 31}},
    {{37, 38, 39},
     {41, 42, 43},
     {45, 46, 47}},
    {{53, 54, 55},
     {57, 58, 59},
     {61, 62, 63}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({
    {{21, 22, 23},
     {25, 26, 27},
     {29, 30, 31}},
    {{37, 38, 39},
     {41, 42, 43},
     {45, 46, 47}},
    {{53, 54, 55},
     {57, 58, 59},
     {61, 62, 63}},
  }, torch::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool1d) {
  AdaptiveAvgPool1d model(3);
  auto x = torch::tensor({{{1, 2, 3, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{1.5, 3.0, 4.5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool2dEven) {
  AdaptiveAvgPool2d model(3);
  auto x = torch::arange(0, 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{ 3.0,  4.5,  6.0},
     {10.5, 12.0, 13.5},
     {18.0, 19.5, 21.0}},
    {{28.0, 29.5, 31.0},
     {35.5, 37.0, 38.5},
     {43.0, 44.5, 46.0}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool2dUneven) {
  AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
  auto x = torch::arange(0, 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{2.5, 4.5},
     {8.5, 10.5},
     {14.5, 16.5}},
    {{22.5, 24.5},
     {28.5, 30.5},
     {34.5, 36.5}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveAvgPool3d) {
  AdaptiveAvgPool3d model(3);
  auto x = torch::arange(0, 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({
    {{10.5, 11.5, 12.5},
     {14.5, 15.5, 16.5},
     {18.5, 19.5, 20.5}},
    {{26.5, 27.5, 28.5},
     {30.5, 31.5, 32.5},
     {34.5, 35.5, 36.5}},
    {{42.5, 43.5, 44.5},
     {46.5, 47.5, 48.5},
     {50.5, 51.5, 52.5}},
  }, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool1d) {
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool1d{3};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y,
    torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat).requires_grad(true));
  model = MaxUnpool1d{MaxUnpool1dOptions(3).stride(2).padding(1)};
  y = model->forward(x, indices, std::vector<int64_t>({1, 1, 5}));

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y,
    torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 5}));
}

TEST_F(ModulesTest, MaxPool1d_MaxUnpool1d) {
  MaxPool1d pool {MaxPool1dOptions(2).stride(2)};
  MaxUnpool1d unpool {MaxUnpool1dOptions(2).stride(2)};
  auto input = torch::tensor({{{1, 2, 3, 4, 5, 6, 7, 8}}}, torch::kFloat);
  torch::Tensor output, indices;
  std::tie(output, indices) = pool->forward_with_indices(input);
  ASSERT_TRUE(torch::allclose(
    unpool(output, indices),
    torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}} , torch::kFloat)));

  // Example showcasing the use of output_size
  input = torch::tensor({{{1, 2, 3, 4, 5, 6, 7, 8, 9}}}, torch::kFloat);
  std::tie(output, indices) = pool->forward_with_indices(input);
  ASSERT_TRUE(torch::allclose(
    unpool(output, indices, input.sizes()),
    torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8, 0}}} , torch::kFloat)));
  ASSERT_TRUE(torch::allclose(
    unpool(output, indices),
    torch::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}} , torch::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool2d) {
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
  auto model = MaxUnpool2d{MaxUnpool2dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices);

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

TEST_F(ModulesTest, MaxPool2d_MaxUnpool2d) {
  MaxPool2d pool {MaxPool2dOptions(2).stride(2)};
  MaxUnpool2d unpool {MaxUnpool2dOptions(2).stride(2)};
  auto input = torch::tensor({{{{ 1,  2,  3,  4},
                                { 5,  6,  7,  8},
                                { 9, 10, 11, 12},
                                {13, 14, 15, 16}}}}, torch::kFloat);
  torch::Tensor output, indices;
  std::tie(output, indices) = pool->forward_with_indices(input);
  ASSERT_TRUE(torch::allclose(
    unpool(output, indices),
    torch::tensor({{{{ 0,  0, 0,  0},
                     { 0,  6, 0,  8},
                     { 0,  0, 0,  0},
                     { 0, 14, 0, 16}}}} , torch::kFloat)));

  ASSERT_TRUE(torch::allclose(
    unpool(output, indices, std::vector<int64_t>{1, 1, 5, 5}),
    torch::tensor({{{{ 0, 0, 0,  0, 0},
                     { 6, 0, 8,  0, 0},
                     { 0, 0, 0, 14, 0},
                     { 16, 0, 0, 0, 0},
                     { 0, 0, 0,  0, 0}}}}, torch::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool3d) {
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  auto x = torch::tensor({{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool3d{3};
  auto y = model->forward(x, indices);

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

TEST_F(ModulesTest, MaxUnpool3dOutputSize) {
  auto indices = torch::tensor(
    {{{{{21, 23},
        {29, 31}},
       {{53, 55},
        {61, 63}}}}}, torch::kLong);
    auto x = torch::tensor(
    {{{{{21, 23},
        {29, 31}},
       {{53, 55},
        {61, 63}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto model = MaxUnpool3d{MaxUnpool3dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices, std::vector<int64_t>({1, 1, 4, 4, 4}));

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(torch::allclose(y, torch::tensor(
   {{{{{ 0,  0,  0,  0},
       { 0,  0,  0,  0},
       { 0,  0,  0,  0},
       { 0,  0,  0,  0}},
      {{ 0,  0,  0,  0},
       { 0, 21,  0, 23},
       { 0,  0,  0,  0},
       { 0, 29,  0, 31}},
      {{ 0,  0,  0,  0},
       { 0,  0,  0,  0},
       { 0,  0,  0,  0},
       { 0,  0,  0,  0}},
      {{ 0,  0,  0,  0},
       { 0, 53,  0, 55},
       { 0,  0,  0,  0},
       { 0, 61,  0, 63}}}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 4, 4, 4}));
}

TEST_F(ModulesTest, MaxPool3d_MaxUnpool3d) {
  MaxPool3d pool {MaxPool3dOptions(3).stride(2)};
  MaxUnpool3d unpool {MaxUnpool3dOptions(3).stride(2)};
  auto input = torch::randn({20, 16, 51, 33, 15});
  torch::Tensor output, indices;
  std::tie(output, indices) = pool->forward_with_indices(input);
  auto unpooled_output = unpool(output, indices);
  ASSERT_EQ(unpooled_output.sizes(), std::vector<int64_t>({20, 16, 51, 33, 15}));
}

TEST_F(ModulesTest, Linear) {
  {
    Linear model(5, 2);
    auto x = torch::randn({10, 5}, torch::requires_grad());
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.size(0), 10);
    ASSERT_EQ(y.size(1), 2);

    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    auto y_exp = torch::addmm(model->bias, x, model->weight.t());
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
  {
    Linear model(LinearOptions(5, 2).bias(false));
    auto x = torch::randn({10, 5}, torch::requires_grad());
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.size(0), 10);
    ASSERT_EQ(y.size(1), 2);

    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    auto y_exp = torch::mm(x, model->weight.t());
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, LocalResponseNorm) {
  {
    LocalResponseNorm model(LocalResponseNormOptions(2));
    const auto x = torch::arange(100, 136, torch::requires_grad()).reshape({2, 3, 3, 2});
    auto y = model(x);
    const auto y_exp = torch::tensor(
      {{{{73.7788, 74.1462},
          {74.5031, 74.8572},
          {75.2010, 75.5420}},

         {{61.6057, 61.7227},
          {61.8347, 61.9418},
          {62.0441, 62.1418}},

         {{62.2349, 62.3235},
          {62.4077, 62.4877},
          {62.5635, 62.6353}}},

        {{{79.3915, 79.6491},
          {79.8978, 80.1446},
          {80.3827, 80.6190}},

         {{63.0317, 63.0742},
          {63.1135, 63.1496},
          {63.1826, 63.2126}},

         {{63.2396, 63.2637},
          {63.2850, 63.3036},
          {63.3195, 63.3328}}}},
      torch::kFloat
    );
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 4);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.sizes(), x.sizes());
    ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
  }
}

TEST_F(ModulesTest, LayerNorm) {
  LayerNorm model(LayerNormOptions({2, 2}).eps(2e-5));
  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto y = model(x);
  auto y_exp = torch::layer_norm(x, {2, 2}, model->weight, model->bias, 2e-5);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  for (auto i = 0; i < 2; i++) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 2 * 2);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Bilinear) {
  Bilinear model(5, 3, 2);
  auto x1 = torch::randn({10, 5}, torch::requires_grad());
  auto x2 = torch::randn({10, 3}, torch::requires_grad());
  auto y = model(x1, x2);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5 * 3);
}

TEST_F(ModulesTest, Fold) {
  {
    Fold model(FoldOptions({3, 2}, {2, 2}));
    auto input = torch::ones({1, 3 * 2 * 2, 2}, torch::requires_grad());
    auto output = model(input);
    auto expected = torch::tensor(
        {{{{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
          {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
          {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}}}},
        torch::kFloat);
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 3, 2}));
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    Fold model(FoldOptions({8, 8}, {3, 3}));
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 3, 16, 16})),
        "Input Error: Only 3D input Tensors are supported (got 4D)");
  }
}

TEST_F(ModulesTest, Unfold) {
  {
    Unfold model(UnfoldOptions({2, 2}).padding(1).stride(2));
    auto input = torch::arange(2, 14, torch::requires_grad()).view({1, 2, 2, 3});
    auto output = model(input);
    auto expected = torch::tensor(
        {{{0.0, 0.0, 0.0, 6.0},
          {0.0, 0.0, 5.0, 7.0},
          {0.0, 3.0, 0.0, 0.0},
          {2.0, 4.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 12.0},
          {0.0, 0.0, 11.0, 13.0},
          {0.0, 9.0, 0.0, 0.0},
          {8.0, 10.0, 0.0, 0.0}}},
        torch::kFloat);
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 8, 4}));
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    Unfold model(UnfoldOptions({2, 4}));
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 5, 2})),
        "Input Error: Only 4D input Tensors are supported (got 3D)");
  }
  {
    // calculated output shape is too small
    Unfold model(UnfoldOptions({2, 3}));
    ASSERT_THROWS_WITH(
        model(torch::randn({1, 2, 2, 2})),
        "Given input with spatial size (2, 2), kernel_size=(2, 3), "
        "dilation=(1, 1), padding=(0, 0), calculated shape of the array of "
        "sliding blocks as (1, 0), which is too small (non-positive).");
  }
}

TEST_F(ModulesTest, SimpleContainer) {
  auto model = std::make_shared<SimpleContainer>();
  auto l1 = model->add(Linear(10, 3), "l1");
  auto l2 = model->add(Linear(3, 5), "l2");
  auto l3 = model->add(Linear(5, 100), "l3");

  auto x = torch::randn({1000, 10}, torch::requires_grad());
  x = l1(x).clamp_min(0);
  x = l2(x).clamp_min(0);
  x = l3(x).clamp_min(0);

  x.backward(torch::ones_like(x));
  ASSERT_EQ(x.ndimension(), 2);
  ASSERT_EQ(x.size(0), 1000);
  ASSERT_EQ(x.size(1), 100);
  ASSERT_EQ(x.min().item<float>(), 0);
}

TEST_F(ModulesTest, EmbeddingBasic) {
  const int64_t dict_size = 10;
  Embedding model(dict_size, 2);
  ASSERT_TRUE(model->named_parameters().contains("weight"));
  ASSERT_EQ(model->weight.ndimension(), 2);
  ASSERT_EQ(model->weight.size(0), dict_size);
  ASSERT_EQ(model->weight.size(1), 2);

  // Cannot get gradients to change indices (input) - only for embedding
  // params
  auto x = torch::full({10}, dict_size - 1, torch::kInt64);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * dict_size);
}

TEST_F(ModulesTest, EmbeddingList) {
  Embedding model(6, 4);
  auto x = torch::full({2, 3}, 5, torch::kInt64);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.size(0), 2);
  ASSERT_EQ(y.size(1), 3);
  ASSERT_EQ(y.size(2), 4);
}

TEST_F(ModulesTest, EmbeddingFromPretrained) {
  auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  Embedding embedding = torch::nn::Embedding::from_pretrained(weight);
  auto input = torch::tensor({1}, torch::kLong);
  ASSERT_TRUE(torch::allclose(embedding(input), torch::tensor({4.0000, 5.1000, 6.3000})));
}

TEST_F(ModulesTest, EmbeddingBagFromPretrained) {
  auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  EmbeddingBag embeddingbag = torch::nn::EmbeddingBag::from_pretrained(weight);
  auto input = torch::zeros({{1, 2}}, torch::kLong);
  input[0] = torch::tensor({1, 0});
  ASSERT_TRUE(torch::allclose(embeddingbag(input), torch::tensor({2.5000, 3.7000, 4.6500})));
}

TEST_F(ModulesTest, Dropout) {
  Dropout dropout(0.5);
  torch::Tensor x = torch::ones(100, torch::requires_grad());
  torch::Tensor y = dropout(x);

  y.backward(torch::ones_like(y));
  ASSERT_EQ(y.ndimension(), 1);
  ASSERT_EQ(y.size(0), 100);
  ASSERT_LT(y.sum().item<float>(), 130); // Probably
  ASSERT_GT(y.sum().item<float>(), 70); // Probably

  dropout->eval();
  y = dropout(x);
  ASSERT_EQ(y.sum().item<float>(), 100);
}

TEST_F(ModulesTest, Parameters) {
  auto model = std::make_shared<NestedModel>();
  auto parameters = model->named_parameters();
  ASSERT_EQ(parameters["param"].size(0), 3);
  ASSERT_EQ(parameters["param"].size(1), 2);
  ASSERT_EQ(parameters["param"].size(2), 21);
  ASSERT_EQ(parameters["l1.bias"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(1), 5);
  ASSERT_EQ(parameters["test.l1.bias"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(1), 10);
  ASSERT_EQ(parameters["test.l2.bias"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(1), 3);
  ASSERT_EQ(parameters["test.l3.bias"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(1), 5);
}

TEST_F(ModulesTest, FunctionalCallsSuppliedFunction) {
  bool was_called = false;
  auto functional = Functional([&was_called](torch::Tensor input) {
    was_called = true;
    return input;
  });
  auto output = functional(torch::ones(5, torch::requires_grad()));
  ASSERT_TRUE(was_called);
  ASSERT_TRUE(output.equal(torch::ones(5, torch::requires_grad())));

  was_called = false;
  // Use the call operator overload here.
  output = functional(torch::ones(5, torch::requires_grad()));
  ASSERT_TRUE(was_called);
  ASSERT_TRUE(output.equal(torch::ones(5, torch::requires_grad())));
}

TEST_F(ModulesTest, FunctionalWithTorchFunction) {
  auto functional = Functional(torch::relu);
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 1);
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 1);
  ASSERT_EQ(functional(torch::ones({}) * -1).item<float>(), 0);
}

TEST_F(ModulesTest, FunctionalArgumentBinding) {
  auto functional =
      Functional(torch::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1);
  ASSERT_EQ(functional(torch::ones({})).item<float>(), 0);
}

TEST_F(ModulesTest, BatchNormStateful) {
  BatchNorm bn(5);

  // Is stateful by default.
  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  // Is affine by default.
  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}
TEST_F(ModulesTest, BatchNormStateless) {
  BatchNorm bn(BatchNormOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());

  ASSERT_THROWS_WITH(
      bn(torch::ones({2, 5})),
      "Calling BatchNorm::forward is only permitted "
      "when the 'track_running_stats' option is true (was false). "
      "Use BatchNorm::pure_forward instead.");
}

TEST_F(ModulesTest, BatchNormPureForward) {
  BatchNorm bn(BatchNormOptions(5).affine(false));
  bn->eval();

  // Want to make sure we use the supplied values in `pure_forward` even if
  // we are stateful.
  auto input = torch::randn({2, 5});
  auto mean = torch::randn(5);
  auto variance = torch::rand(5);
  auto output = bn->pure_forward(input, mean, variance);
  auto expected = (input - mean) / torch::sqrt(variance + bn->options.eps());
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(ModulesTest, BatchNormLegacyWarning) {
  std::stringstream buffer;
  torch::test::CerrRedirect cerr_redirect(buffer.rdbuf());

  BatchNorm bn(5);

  ASSERT_EQ(
    count_substr_occurrences(
      buffer.str(),
      "torch::nn::BatchNorm module is deprecated"
    ),
  1);
}

TEST_F(ModulesTest, BatchNorm1dStateful) {
  BatchNorm1d bn(BatchNorm1dOptions(5));

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm1dStateless) {
  BatchNorm1d bn(BatchNorm1dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm1d) {
  BatchNorm1d bn(BatchNorm1dOptions(5));
  bn->eval();

  auto input = torch::randn({2, 5}, torch::requires_grad());
  auto output = bn->forward(input);
  auto s = output.sum();
  s.backward();
  
  ASSERT_EQ(input.sizes(), input.grad().sizes());
  ASSERT_TRUE(input.grad().allclose(torch::ones({2, 5})));
}

TEST_F(ModulesTest, BatchNorm2dStateful) {
  BatchNorm2d bn(BatchNorm2dOptions(5));

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm2dStateless) {
  BatchNorm2d bn(BatchNorm2dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm2d) {
  BatchNorm2d bn(BatchNorm2dOptions(5));
  bn->eval();

  auto input = torch::randn({2, 5, 4, 4}, torch::requires_grad());
  auto output = bn->forward(input);
  auto s = output.sum();
  s.backward();
  
  ASSERT_EQ(input.sizes(), input.grad().sizes());
  ASSERT_TRUE(input.grad().allclose(torch::ones({2, 5, 4, 4})));
}

TEST_F(ModulesTest, BatchNorm3dStateful) {
  BatchNorm3d bn(BatchNorm3dOptions(5));

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm3dStateless) {
  BatchNorm3d bn(BatchNorm3dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm3d) {
  BatchNorm3d bn(BatchNorm3dOptions(5));
  bn->eval();

  auto input = torch::randn({2, 5, 4, 4, 4}, torch::requires_grad());
  auto output = bn->forward(input);
  auto s = output.sum();
  s.backward();
  
  ASSERT_EQ(input.sizes(), input.grad().sizes());
  ASSERT_TRUE(input.grad().allclose(torch::ones({2, 5, 4, 4, 4})));
}

TEST_F(ModulesTest, Linear_CUDA) {
  Linear model(5, 2);
  model->to(torch::kCUDA);
  auto x =
      torch::randn({10, 5}, torch::device(torch::kCUDA).requires_grad(true));
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5);
}

TEST_F(ModulesTest, Linear2_CUDA) {
  Linear model(5, 2);
  model->to(torch::kCUDA);
  model->to(torch::kCPU);
  auto x = torch::randn({10, 5}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5);
}

TEST_F(ModulesTest, L1Loss) {
  L1Loss loss;
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = loss->forward(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), std::vector<int64_t>());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MSELoss) {
  MSELoss loss;
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = loss->forward(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, BCELoss) {
  BCELoss loss;
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = loss->forward(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, KLDivLoss) {
  KLDivLoss loss;
  auto input = torch::randn({5,6}, torch::requires_grad());
  auto target = torch::empty({5,6}).random_(2);
  auto output = loss->forward(torch::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HingeEmbeddingLoss) {
  HingeEmbeddingLoss loss(HingeEmbeddingLossOptions().margin(2));
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({10}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiMarginLoss) {
  auto weight = torch::tensor({0.3, 0.3, 0.4}, torch::kFloat);
  MultiMarginLoss loss(MultiMarginLossOptions().margin(2).weight(weight));
  auto input = torch::tensor({{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({2, 1, 0}, torch::kLong);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({0.305556}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, CosineEmbeddingLoss) {
  CosineEmbeddingLoss cos(CosineEmbeddingLossOptions().margin(0.5));
  auto input1 = torch::tensor({{2, 3, 4}, {6, 2, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto input2 = torch::tensor({{2, 3, 5}, {9, 12, 0}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({1, -1});
  auto output = cos(input1, input2, target);
  auto expected = torch::tensor({0.1004}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-4));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
  ASSERT_EQ(input2.sizes(), input2.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossDefaultOptions) {
  SmoothL1Loss loss;
  auto input = torch::tensor({0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output = loss(input, target);
  auto expected = torch::tensor(0.0233335, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelMarginLossDefaultOptions) {
  MultiLabelMarginLoss loss;
  auto input = torch::tensor({{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossNoReduction) {
  SmoothL1Loss loss(/*reduction=*/torch::Reduction::None);
  auto input = torch::tensor({0.1, 1.2, 4.7}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({0., 1., 5.}, torch::kFloat);
  auto output = loss(input, target);
  auto expected = torch::tensor({0.005, 0.02, 0.045}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelMarginLossNoReduction) {
  MultiLabelMarginLoss loss(torch::kNone);
  auto input = torch::tensor({{0.1, 0.2, 0.4, 0.8}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{3, 0, -1, 1}}, torch::kLong);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({0.8500}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, TripletMarginLoss) {
  TripletMarginLoss loss(TripletMarginLossOptions().margin(1.0));
  auto anchor = torch::tensor({{3., 3.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto positive = torch::tensor({{2., 2.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto negative = torch::tensor({{0., 0.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = loss->forward(anchor, positive, negative);
  auto expected = torch::tensor({0.}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
}

TEST_F(ModulesTest, CosineSimilarity) {
  CosineSimilarity cos(CosineSimilarityOptions().dim(1));
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = cos->forward(input1, input2);
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}

TEST_F(ModulesTest, SoftMarginLossDefaultOptions) {
  SoftMarginLoss loss;
  auto input = torch::tensor({2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({1.3767317}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelSoftMarginLossDefaultOptions) {
  MultiLabelSoftMarginLoss loss;
  auto input = torch::tensor({{0., 2., 2., 0.}, {2., 1., 0., 1.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({0.7608436}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SoftMarginLossNoReduction) {
  SoftMarginLoss loss(torch::kNone);
  auto input = torch::tensor({2., 4., 1., 3.}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({-1., 1., 1., -1.}, torch::kFloat);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({2.1269281, 0.01814993, 0.3132617, 3.0485873}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelSoftMarginLossWeightedNoReduction) {
  auto input = torch::tensor({{0., 2., 2., 0.}, {2., 1., 0., 1.}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto target = torch::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, torch::kFloat);
  auto weight = torch::tensor({0.1, 0.6, 0.4, 0.8}, torch::kFloat);
  auto options = MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight);
  MultiLabelSoftMarginLoss loss = MultiLabelSoftMarginLoss(options);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({0.4876902, 0.3321295}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, PairwiseDistance) {
  PairwiseDistance dist(PairwiseDistanceOptions().p(1));
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::dtype(torch::kFloat).requires_grad(true));
  auto output = dist->forward(input1, input2);
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}

TEST_F(ModulesTest, ELU) {
  const auto size = 3;
  for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    ELU model {ELUOptions().alpha(alpha)};
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = torch::max(torch::zeros_like(x), x) +
                 torch::min(torch::zeros_like(x), alpha * (torch::exp(x) - 1.0));
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, SELU) {
  SELU model;
  auto input = torch::randn({5, 5}, torch::requires_grad());
  auto output = model->forward(input);
  const double scale = 1.0507009873554804934193349852946;
  const double alpha = 1.6732632423543772848170429916717;
  auto zero = torch::zeros_like(input);
  auto expected = scale *
      (torch::max(zero, input) +
       torch::min(zero, alpha * (torch::exp(input) - 1)));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(ModulesTest, Hardshrink) {
  const auto size = 3;
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    Hardshrink model {HardshrinkOptions().lambda(lambda)};
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x.abs() > lambda) * x;
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, Hardtanh) {
  const auto size = 3;
  for (const auto min_val : {-4.2, -1.0, -0.42, 0.0}) {
    for (const auto max_val : {0.42, 1.0, 4.2}) {
      Hardtanh model {HardtanhOptions().min_val(min_val).max_val(max_val)};
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size}).set_requires_grad(true);
      auto y = model(x);
      torch::Tensor s = y.sum();

      s.backward();
      ASSERT_EQ(s.ndimension(), 0);

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      auto y_exp = (x < min_val) * min_val +
                   ((x >= min_val) * (x <= max_val)) * x +
                   (x > max_val) * max_val;
      ASSERT_TRUE(torch::allclose(y, y_exp));
    }
  }
}

TEST_F(ModulesTest, HardtanhMinValGEMaxVal) {
  ASSERT_THROWS_WITH(Hardtanh{HardtanhOptions().min_val(0.42).max_val(0.42)},
                     "max_val must be greater than min_val");
  ASSERT_THROWS_WITH(Hardtanh{HardtanhOptions().min_val(0.42).max_val(-0.42)},
                     "max_val must be greater than min_val");

  Hardtanh ht {HardtanhOptions().min_val(-0.42).max_val(0.42)};
  ht->options.min_val(0.42);
  ASSERT_THROWS_WITH(ht->reset(), "max_val must be greater than min_val");
  ht->options.max_val(-0.42);
  ASSERT_THROWS_WITH(ht->reset(), "max_val must be greater than min_val");
}

TEST_F(ModulesTest, LeakyReLU) {
  const auto size = 3;
  for (const auto negative_slope : {0.0, 0.42, 1.0}) {
    LeakyReLU model {LeakyReLUOptions().negative_slope(negative_slope)};
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x < 0) * x * negative_slope + (x >= 0) * x;
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, LogSigmoid) {
  const auto size = 3;
  LogSigmoid model;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = torch::log(torch::ones_like(x)/(torch::ones_like(x) + torch::exp(torch::neg(x))));
  ASSERT_TRUE(torch::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(ModulesTest, Softmax) {
  Softmax m(/*dim=*/1);
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  auto output = m(input);
  auto sum = torch::sum(torch::exp(input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::exp(input[i]) / sum[i];
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, Softmin) {
  Softmin m(/*dim=*/1);
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  auto output = m(input);
  auto sum = torch::sum(torch::exp(-input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::exp(-input[i]) / sum[i];
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, LogSoftmax) {
  LogSoftmax m(/*dim=*/1);
  auto input = torch::arange(10, torch::kFloat).reshape({2, 5});
  auto output = m(input);
  auto sum = torch::sum(torch::exp(input), 1);

  for (int i = 0; i < 2; i++) {
    auto expected = torch::log(torch::exp(input[i]) / sum[i]);
    ASSERT_TRUE(torch::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, Softmax2d) {
  Softmax2d m;
  auto input = torch::arange(24, torch::kFloat).reshape({1, 2, 3, 4});
  auto output = m(input);
  auto sum = torch::sum(torch::exp(input), 1);

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 4; l++) {
          auto expected = torch::exp(input[i][j][k][l]) / sum[i][k][l];
          ASSERT_TRUE(torch::allclose(output[i][j][k][l], expected));
        }
      }
    }
  }
}

TEST_F(ModulesTest, PReLU) {
  const auto num_parameters = 42;
  const auto init = 0.42;

  PReLU model {PReLUOptions().num_parameters(num_parameters).init(init)};

  ASSERT_EQ(model->weight.sizes(), std::vector<int64_t>({num_parameters}));
  ASSERT_TRUE(torch::allclose(model->weight,
              torch::full(num_parameters, init)));

  const auto x = torch::rand({100, num_parameters}) * 200 - 100;
  const auto y = model(x);
  const auto s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), x.ndimension());
  ASSERT_EQ(y.sizes(), x.sizes());
  const auto y_exp = (x < 0) * model->weight * x  + (x >= 0) * x;
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, ReLU) {
  const auto size = 3;
  ReLU model;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = (x < 0) * 0 + (x >= 0) * x;
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, ReLU6) {
  const auto size = 3;
  ReLU6 model;
  auto x = torch::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = (x < 0) * 0 + ((x >= 0) * (x <= 6)) * x + (x > 6) * 6;
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, RReLU) {
  const auto size = 3;
  for (const auto lower : {0.01, 0.1, 0.2}) {
    for (const auto upper : {0.3, 0.4, 0.5}) {
      RReLU model {RReLUOptions().lower(lower).upper(upper)};
      auto x = torch::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size}).set_requires_grad(true);
      auto y = model(x);
      torch::Tensor s = y.sum();

      s.backward();
      ASSERT_EQ(s.ndimension(), 0);

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      auto z = ((x >= 0) * (x == y) +
        (x < 0) * (y >= x * upper) * (y <= lower * x)) * 1.0;
      ASSERT_TRUE(torch::allclose(z, torch::ones_like(z)));
    }
  }
}

TEST_F(ModulesTest, CELU) {
  const auto size = 3;
  for (const auto alpha : {0.42, 1.0, 4.2, 42.42}) {
    CELU model {CELUOptions().alpha(alpha)};
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = torch::max(torch::zeros_like(x), x) +
        torch::min(torch::zeros_like(x), alpha * (torch::exp(x / alpha) - 1.0));
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, GELU) {
  GELU model;
  const auto x = torch::linspace(-3.0, 3.0, 100);
  const auto y_exp = x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
  const auto y = model(x);
  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Sigmoid) {
  Sigmoid model;
  auto x = torch::randn(100) * 10;
  auto y_exp = 1 / (1 + torch::exp(-x));
  auto y = model(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, PixelShuffle) {
  PixelShuffle module(/*upscale_factor=*/2);
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
  auto y = module(x);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 4, 4}));
  ASSERT_TRUE(y.allclose(y_exp));
}

TEST_F(ModulesTest, Softplus) {
  const auto size = 3;
  for (const auto beta : {0.5, 1.0, 2.0}) {
    for (const auto threshold : {1.0, 3.0, 5.0}) {
      Softplus model {SoftplusOptions().beta(beta).threshold(threshold)};
      auto x = torch::linspace(-3.0, 3.0, 61);
      x.resize_({size, size, size});
      auto y_exp =
        (x <= threshold) * torch::log(1 + torch::exp(x * beta)) / beta +
        (x > threshold) * x;
      auto y = model(x);

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(torch::allclose(y, y_exp));
    }
  }
}

TEST_F(ModulesTest, Softshrink) {
  const auto size = 3;
  for (const auto lambda : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    Softshrink model {/*lambda=*/lambda};
    auto x = torch::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    torch::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, Softsign) {
  Softsign model;
  auto x = torch::randn(100) * 10;
  auto y_exp = x / (1 + x.abs());
  auto y = model(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Tanh) {
  Tanh model;
  auto x = torch::randn(100) * 10;
  auto y_exp = (x.exp() - (-x).exp()) / (x.exp() + (-x).exp());
  auto y = model(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Tanhshrink) {
  Tanhshrink model;
  auto x = torch::randn(100) * 10;
  auto y_exp = x - x.tanh();
  auto y = model(x);

  ASSERT_TRUE(torch::allclose(y, y_exp));
}

TEST_F(ModulesTest, Threshold) {
  const auto size = 3;
  for (const auto threshold : {0.5, 1.0, 2.0}) {
    for (const auto value : {0.5, 1.0, 2.0}) {
      for (const auto inplace : {false, true}) {
        Threshold model {ThresholdOptions(threshold, value).inplace(inplace)};
        auto x = torch::linspace(-3.0, 3.0, 61);
        x.resize_({size, size, size});
        auto y_exp = (x <= threshold) * value + (x > threshold) * x;
        auto y = model(x);

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(torch::allclose(y, y_exp));
      }
    }
  }
}

TEST_F(ModulesTest, Upsampling1D) {
  {
    Upsample model(UpsampleOptions()
                       .size({4})
                       .mode(torch::kNearest));
    auto input = torch::ones({1, 1, 2}, torch::requires_grad());
    auto output = model->forward(input);
    auto expected = torch::ones({1, 1, 4});
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(UpsampleOptions()
                           .scale_factor({scale_factor})
                           .mode(torch::kLinear)
                           .align_corners(align_corners));
        auto input = torch::ones({1, 1, 2}, torch::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = torch::ones({1, 1, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    // linear (1D) upsampling spatial invariance
    Upsample model(UpsampleOptions()
                       .scale_factor({3})
                       .mode(torch::kLinear)
                       .align_corners(false));
    auto input = torch::zeros({1, 1, 9});
    input.narrow(2, 0, 4).normal_();
    auto output = model->forward(input);
    auto expected = model->forward(input.narrow(2, 0, 5));

    ASSERT_TRUE(torch::allclose(output.narrow(2, 0, 15), expected));
  }
}

TEST_F(ModulesTest, Upsampling2D) {
  {
    Upsample model(UpsampleOptions()
                       .size({4, 4})
                       .mode(torch::kNearest));
    auto input = torch::ones({1, 1, 2, 2}, torch::requires_grad());
    auto output = model->forward(input);
    auto expected = torch::ones({1, 1, 4, 4});
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(UpsampleOptions()
                           .scale_factor({scale_factor, scale_factor})
                           .mode(torch::kBilinear)
                           .align_corners(align_corners));
        auto input = torch::ones({1, 1, 2, 2}, torch::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = torch::ones({1, 1, expected_size, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(UpsampleOptions()
                           .scale_factor({scale_factor, scale_factor})
                           .mode(torch::kBicubic)
                           .align_corners(align_corners));
        auto input = torch::ones({1, 1, 2, 2}, torch::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = torch::ones({1, 1, expected_size, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
}

TEST_F(ModulesTest, Upsampling3D) {
  {
    Upsample model(UpsampleOptions()
                       .size({4, 4, 4})
                       .mode(torch::kNearest));
    auto input = torch::ones({1, 1, 2, 2, 2}, torch::requires_grad());
    auto output = model->forward(input);
    auto expected = torch::ones({1, 1, 4, 4, 4});
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(
            UpsampleOptions()
                .scale_factor({scale_factor, scale_factor, scale_factor})
                .mode(torch::kTrilinear)
                .align_corners(align_corners));
        auto input = torch::ones({1, 1, 2, 2, 2}, torch::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected =
            torch::ones({1, 1, expected_size, expected_size, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
}

TEST_F(ModulesTest, CTCLoss) {
  CTCLoss loss {CTCLossOptions().reduction(torch::kNone)};
  const auto target_lengths = torch::tensor({0, 0, 0});
  const auto input_lengths = torch::tensor({50, 50, 50});
  const auto targets =
    torch::randint(1, 15, at::IntArrayRef({0}), torch::kLong);
  const auto log_probs =
    torch::randn({50, 3, 15}, torch::kDouble).log_softmax(2);
  const auto output =
    loss->forward(log_probs, targets, input_lengths, target_lengths);
  ASSERT_TRUE(output.ge(0).all().item<bool>());
  ASSERT_TRUE(torch::allclose(
    -log_probs.sum(0).slice(1, 0, 1).view_as(output), output));
}

TEST_F(ModulesTest, PoissonNLLLoss) {
  const auto input = torch::tensor({0.5, 1.5, 2.5});
  const auto target = torch::tensor({1., 2., 3.});
  const auto component_wise_loss = torch::exp(input) - target * input;
  {
    PoissonNLLLoss loss {PoissonNLLLossOptions().reduction(torch::kNone)};
    ASSERT_TRUE(torch::allclose(
      component_wise_loss,
      loss->forward(input, target)
    ));
  }
  {
    PoissonNLLLoss loss {PoissonNLLLossOptions().reduction(torch::kSum)};
    ASSERT_TRUE(torch::allclose(
      torch::sum(component_wise_loss),
      loss->forward(input, target)
    ));
  }
  {
    PoissonNLLLoss loss {PoissonNLLLossOptions().reduction(torch::kMean)};
    ASSERT_TRUE(torch::allclose(
      torch::mean(component_wise_loss),
      loss->forward(input, target)
    ));
  }
}

TEST_F(ModulesTest, MarginRankingLoss) {
  {
    MarginRankingLoss loss;
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    const auto target = torch::randn(15).sign();
    ASSERT_TRUE(torch::allclose(
      loss->forward(input1, input2, target),
      (-target * (input1 - input2)).clamp(0).mean()
    ));
  }
  {
    MarginRankingLoss loss {MarginRankingLossOptions().margin(0.5).reduction(torch::kSum)};
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    const auto target = torch::randn(15).sign();
    const auto margin = 0.5;
    ASSERT_TRUE(torch::allclose(
      loss->forward(input1, input2, target),
      (-target * (input1 - input2) + margin).clamp(0).sum()
    ));
  }
  {
    MarginRankingLoss loss {MarginRankingLossOptions().margin(0.5).reduction(torch::kMean)};
    const auto input1 = torch::randn(15) * 10;
    const auto input2 = torch::randn(15) * 10;
    const auto target = torch::randn(15).sign();
    const auto margin = 0.5;
    ASSERT_TRUE(torch::allclose(
      loss->forward(input1, input2, target),
      (-target * (input1 - input2) + margin).clamp(0).mean()
    ));
  }
}

TEST_F(ModulesTest, PrettyPrintIdentity) {
  ASSERT_EQ(c10::str(Identity()), "torch::nn::Identity()");
}

TEST_F(ModulesTest, PrettyPrintFlatten) {
  ASSERT_EQ(c10::str(Flatten()), "torch::nn::Flatten()");
  ASSERT_EQ(c10::str(Flatten(FlattenOptions().start_dim(2).end_dim(4))), "torch::nn::Flatten()");
}

TEST_F(ModulesTest, ReflectionPad1d) {
  {
    ReflectionPad1d m(ReflectionPad1dOptions(2));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor({{{2., 1., 0., 1., 2., 3., 2., 1.},
                                    {6., 5., 4., 5., 6., 7., 6., 5.}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReflectionPad1d m(ReflectionPad1dOptions({3, 1}));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor({{{3., 2., 1., 0., 1., 2., 3., 2.},
                                    {7., 6., 5., 4., 5., 6., 7., 6.}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReflectionPad2d) {
  {
    ReflectionPad2d m(ReflectionPad2dOptions(2));
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{{8., 7., 6., 7., 8., 7., 6.},
                                     {5., 4., 3., 4., 5., 4., 3.},
                                     {2., 1., 0., 1., 2., 1., 0.},
                                     {5., 4., 3., 4., 5., 4., 3.},
                                     {8., 7., 6., 7., 8., 7., 6.},
                                     {5., 4., 3., 4., 5., 4., 3.},
                                     {2., 1., 0., 1., 2., 1., 0.}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReflectionPad2d m(ReflectionPad2dOptions({1, 1, 2, 0}));
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{{7., 6., 7., 8., 7.},
                                     {4., 3., 4., 5., 4.},
                                     {1., 0., 1., 2., 1.},
                                     {4., 3., 4., 5., 4.},
                                     {7., 6., 7., 8., 7.}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad1d) {
  {
    ReplicationPad1d m(ReplicationPad1dOptions(2));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor({{{0., 0., 0., 1., 2., 3., 3., 3.},
                                    {4., 4., 4., 5., 6., 7., 7., 7.}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReplicationPad1d m(ReplicationPad1dOptions({3, 1}));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor({{{0., 0., 0., 0., 1., 2., 3., 3.},
                                    {4., 4., 4., 4., 5., 6., 7., 7.}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad2d) {
  {
    ReplicationPad2d m(ReplicationPad2dOptions(2));
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{{0., 0., 0., 1., 2., 2., 2.},
                                     {0., 0., 0., 1., 2., 2., 2.},
                                     {0., 0., 0., 1., 2., 2., 2.},
                                     {3., 3., 3., 4., 5., 5., 5.},
                                     {6., 6., 6., 7., 8., 8., 8.},
                                     {6., 6., 6., 7., 8., 8., 8.},
                                     {6., 6., 6., 7., 8., 8., 8.}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReplicationPad2d m(ReplicationPad2dOptions({1, 1, 2, 0}));
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{{0., 0., 1., 2., 2.},
                                     {0., 0., 1., 2., 2.},
                                     {0., 0., 1., 2., 2.},
                                     {3., 3., 4., 5., 5.},
                                     {6., 6., 7., 8., 8.}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad3d) {
  {
    ReplicationPad3d m(ReplicationPad3dOptions(1));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = torch::tensor({{{{{0., 0., 1., 1.},
                                      {0., 0., 1., 1.},
                                      {2., 2., 3., 3.},
                                      {2., 2., 3., 3.}},
                                     {{0., 0., 1., 1.},
                                      {0., 0., 1., 1.},
                                      {2., 2., 3., 3.},
                                      {2., 2., 3., 3.}},
                                     {{4., 4., 5., 5.},
                                      {4., 4., 5., 5.},
                                      {6., 6., 7., 7.},
                                      {6., 6., 7., 7.}},
                                     {{4., 4., 5., 5.},
                                      {4., 4., 5., 5.},
                                      {6., 6., 7., 7.},
                                      {6., 6., 7., 7.}}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReplicationPad3d m(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = torch::tensor({{{{{0., 0., 1., 1., 1.},
                                      {0., 0., 1., 1., 1.},
                                      {2., 2., 3., 3., 3.},
                                      {2., 2., 3., 3., 3.},
                                      {2., 2., 3., 3., 3.}},
                                     {{0., 0., 1., 1., 1.},
                                      {0., 0., 1., 1., 1.},
                                      {2., 2., 3., 3., 3.},
                                      {2., 2., 3., 3., 3.},
                                      {2., 2., 3., 3., 3.}},
                                     {{4., 4., 5., 5., 5.},
                                      {4., 4., 5., 5., 5.},
                                      {6., 6., 7., 7., 7.},
                                      {6., 6., 7., 7., 7.},
                                      {6., 6., 7., 7., 7.}},
                                     {{4., 4., 5., 5., 5.},
                                      {4., 4., 5., 5., 5.},
                                      {6., 6., 7., 7., 7.},
                                      {6., 6., 7., 7., 7.},
                                      {6., 6., 7., 7., 7.}},
                                     {{4., 4., 5., 5., 5.},
                                      {4., 4., 5., 5., 5.},
                                      {6., 6., 7., 7., 7.},
                                      {6., 6., 7., 7., 7.},
                                      {6., 6., 7., 7., 7.}}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ZeroPad2d) {
  {
    ZeroPad2d m(ZeroPad2dOptions(2));
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{{0., 0., 0., 0., 0., 0., 0.},
                                     {0., 0., 0., 0., 0., 0., 0.},
                                     {0., 0., 0., 1., 2., 0., 0.},
                                     {0., 0., 3., 4., 5., 0., 0.},
                                     {0., 0., 6., 7., 8., 0., 0.},
                                     {0., 0., 0., 0., 0., 0., 0.},
                                     {0., 0., 0., 0., 0., 0., 0.}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ZeroPad2d m(ZeroPad2dOptions({1, 1, 2, 0}));
    auto input = torch::arange(9, torch::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{{0., 0., 0., 0., 0.},
                                     {0., 0., 0., 0., 0.},
                                     {0., 0., 1., 2., 0.},
                                     {0., 3., 4., 5., 0.},
                                     {0., 6., 7., 8., 0.}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad1d) {
  {
    ConstantPad1d m(ConstantPad1dOptions(2, 3.5));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = torch::tensor({{{3.5000, 3.5000, 0.0000, 1.0000, 2.0000, 3.0000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 4.0000, 5.0000, 6.0000, 7.0000, 3.5000, 3.5000}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ConstantPad1d m(ConstantPad1dOptions({3, 1}, 3.5));
    auto input = torch::arange(6, torch::kFloat).reshape({1, 2, 3});
    auto output = m(input);
    auto expected = torch::tensor({{{3.5000, 3.5000, 3.5000, 0.0000, 1.0000, 2.0000, 3.5000},
                                    {3.5000, 3.5000, 3.5000, 3.0000, 4.0000, 5.0000, 3.5000}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad2d) {
  {
    ConstantPad2d m(ConstantPad2dOptions(2, 3.5));
    auto input = torch::arange(4, torch::kFloat).reshape({1, 2, 2});
    auto output = m(input);
    auto expected = torch::tensor({{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 0.0000, 1.0000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 2.0000, 3.0000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ConstantPad2d m(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
    auto input = torch::arange(4, torch::kFloat).reshape({1, 2, 2});
    auto output = m(input);
    auto expected = torch::tensor({{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                    {3.5000, 3.5000, 3.5000, 0.0000, 1.0000},
                                    {3.5000, 3.5000, 3.5000, 2.0000, 3.0000},
                                    {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad3d) {
  {
    ConstantPad3d m(ConstantPad3dOptions(1, 3.5));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = torch::tensor({{{{{3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 0.0000, 1.0000, 3.5000},
                                      {3.5000, 2.0000, 3.0000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 4.0000, 5.0000, 3.5000},
                                      {3.5000, 6.0000, 7.0000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000}}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ConstantPad3d m(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
    auto input = torch::arange(8, torch::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = torch::tensor({{{{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 0.0000, 1.0000, 3.5000, 3.5000},
                                      {3.5000, 2.0000, 3.0000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 4.0000, 5.0000, 3.5000, 3.5000},
                                      {3.5000, 6.0000, 7.0000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
                                     {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
                                      {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}}}}, torch::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, CrossMapLRN2d) {
  /// size 3, default options
  auto input = torch::arange(9, torch::kFloat32).view({1, 1, 3, 3}).requires_grad_(true);
  auto expected = torch::tensor({{{{0.00000000, 0.99997497, 1.99980010},
                                   {2.99932500, 3.99840070, 4.99687700},
                                   {5.99460600, 6.99143740, 7.98722360}}}}, torch::kFloat32);
  auto grad_expected = torch::tensor({{{{1.00000000, 0.99992496, 0.99970007},
                                        {0.99932520, 0.99880093, 0.99812720},
                                        {0.99730474, 0.99633380, 0.99521490}}}}, torch::kFloat32);
  auto crossmaplrn2d = CrossMapLRN2d(3);
  auto output = crossmaplrn2d(input);
  output.sum().backward();
  
  ASSERT_TRUE(input.grad().allclose(grad_expected));
  ASSERT_TRUE(output.allclose(expected));
  
  /// size change
  crossmaplrn2d = CrossMapLRN2d(CrossMapLRN2dOptions(4).alpha(1e-4).beta(0.75).k(1));
  output = crossmaplrn2d(input);
  expected = torch::tensor({{{{0.00000000, 0.99998120, 1.99985000},
                              {2.99949400, 3.99880050, 4.99765800},
                              {5.99595300, 6.99357600, 7.99041300}}}}, torch::kFloat32);
  ASSERT_TRUE(output.allclose(expected));

  /// alpha change
  crossmaplrn2d = CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-3).beta(0.75).k(1));
  output = crossmaplrn2d(input);
  expected = torch::tensor({{{{0.00000000, 0.99975010, 1.99800230},
                              {2.99326750, 3.98407440, 4.96897600},
                              {5.94656100, 6.91545720, 7.87434340}}}}, torch::kFloat32);
  ASSERT_TRUE(output.allclose(expected));

  /// beta change
  crossmaplrn2d = CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-4).beta(0.95).k(1));
  output = crossmaplrn2d(input);
  expected = torch::tensor({{{{0.00000000, 0.99996830, 1.99974680},
                              {2.99914500, 3.99797440, 4.99604460},
                              {5.99316840, 6.98915600, 7.98382000}}}}, torch::kFloat32);
  ASSERT_TRUE(output.allclose(expected));

  /// k change
  crossmaplrn2d = CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-4).beta(0.75).k(2));
  output = crossmaplrn2d(input);
  expected = torch::tensor({{{{0.00000000, 0.59459610, 1.18914770},
                              {1.78361000, 2.37793870, 2.97208900},
                              {3.56601700, 4.15967700, 4.75302650}}}}, torch::kFloat32);
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(ModulesTest, PrettyPrintLinear) {
  ASSERT_EQ(
      c10::str(Linear(3, 4)), "torch::nn::Linear(in_features=3, out_features=4, bias=true)");
}

TEST_F(ModulesTest, PrettyPrintBilinear) {
  ASSERT_EQ(
      c10::str(Bilinear(3, 2, 4)), "torch::nn::Bilinear(in1_features=3, in2_features=2, out_features=4, bias=true)");
  ASSERT_EQ(
      c10::str(Bilinear(BilinearOptions(3, 2, 4).bias(false))), "torch::nn::Bilinear(in1_features=3, in2_features=2, out_features=4, bias=false)");
}

TEST_F(ModulesTest, PrettyPrintConv) {
  ASSERT_EQ(
      c10::str(Conv1d(3, 4, 5)),
      "torch::nn::Conv1d(input_channels=3, output_channels=4, kernel_size=5, stride=1)");
  ASSERT_EQ(
      c10::str(Conv2d(3, 4, 5)),
      "torch::nn::Conv2d(input_channels=3, output_channels=4, kernel_size=[5, 5], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(Conv2d(Conv2dOptions(3, 4, 5).stride(2))),
      "torch::nn::Conv2d(input_channels=3, output_channels=4, kernel_size=[5, 5], stride=[2, 2])");

  const auto options =
      Conv2dOptions(3, 4, std::vector<int64_t>{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(Conv2d(options)),
      "torch::nn::Conv2d(input_channels=3, output_channels=4, kernel_size=[5, 6], stride=[1, 2])");
}

TEST_F(ModulesTest, PrettyPrintUpsample) {
  ASSERT_EQ(
      c10::str(Upsample(UpsampleOptions().size({2, 4, 4}))),
      "torch::nn::Upsample(size=[2, 4, 4], mode=kNearest)");
  ASSERT_EQ(
      c10::str(Upsample(UpsampleOptions().scale_factor({0.5, 1.5}).mode(torch::kBilinear))),
      "torch::nn::Upsample(scale_factor=[0.5, 1.5], mode=kBilinear)");
}

TEST_F(ModulesTest, PrettyPrintFold) {
  ASSERT_EQ(
      c10::str(Fold(FoldOptions({2, 2}, {5, 5}))),
      "torch::nn::Fold(output_size=[2, 2], kernel_size=[5, 5], dilation=[1, 1], padding=[0, 0], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(Fold(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2, 1}).stride(2))),
      "torch::nn::Fold(output_size=[8, 8], kernel_size=[3, 3], dilation=[2, 2], padding=[2, 1], stride=[2, 2])");
}

TEST_F(ModulesTest, PrettyPrintUnfold) {
  ASSERT_EQ(
      c10::str(Unfold(torch::IntArrayRef({2, 4}))),
      "torch::nn::Unfold(kernel_size=[2, 4], dilation=[1, 1], padding=[0, 0], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(Unfold(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2))),
      "torch::nn::Unfold(kernel_size=[2, 4], dilation=[2, 2], padding=[2, 1], stride=[2, 2])");
}

TEST_F(ModulesTest, PrettyPrintMaxPool) {
  ASSERT_EQ(
      c10::str(MaxPool1d(5)),
      "torch::nn::MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool2d(5)),
      "torch::nn::MaxPool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool2d(MaxPool2dOptions(5).stride(2))),
      "torch::nn::MaxPool2d(kernel_size=[5, 5], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool3d(5)),
      "torch::nn::MaxPool3d(kernel_size=[5, 5, 5], stride=[5, 5, 5], padding=[0, 0, 0], dilation=[1, 1, 1], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool3d(MaxPool3dOptions(5).stride(2))),
      "torch::nn::MaxPool3d(kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=[0, 0, 0], dilation=[1, 1, 1], ceil_mode=false)");

  const auto options =
      MaxPool2dOptions(std::vector<int64_t>{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(MaxPool2d(options)),
      "torch::nn::MaxPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
}

TEST_F(ModulesTest, PrettyPrintAvgPool) {
  ASSERT_EQ(
      c10::str(AvgPool1d(5)),
      "torch::nn::AvgPool1d(kernel_size=5, stride=5, padding=0)");
  ASSERT_EQ(
      c10::str(AvgPool2d(5)),
      "torch::nn::AvgPool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(AvgPool2d(AvgPool2dOptions(5).stride(2))),
      "torch::nn::AvgPool2d(kernel_size=[5, 5], stride=[2, 2], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(AvgPool3d(5)),
      "torch::nn::AvgPool3d(kernel_size=[5, 5, 5], stride=[5, 5, 5], padding=[0, 0, 0])");
  ASSERT_EQ(
      c10::str(AvgPool3d(AvgPool3dOptions(5).stride(2))),
      "torch::nn::AvgPool3d(kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=[0, 0, 0])");

  const auto options =
      AvgPool2dOptions(std::vector<int64_t>{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(AvgPool2d(options)),
      "torch::nn::AvgPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0])");
}

TEST_F(ModulesTest, PrettyPrintLPPool) {
  ASSERT_EQ(
      c10::str(LPPool1d(2, 5)),
      "torch::nn::LPPool1d(norm_type=2, kernel_size=5, stride=5, ceil_mode=false)");
  ASSERT_EQ(
      c10::str(LPPool1d(LPPool1dOptions(1, 2).stride(5).ceil_mode(true))),
      "torch::nn::LPPool1d(norm_type=1, kernel_size=2, stride=5, ceil_mode=true)");
  ASSERT_EQ(
      c10::str(LPPool2d(2, std::vector<int64_t>({1, 2}))),
      "torch::nn::LPPool2d(norm_type=2, kernel_size=[1, 2], stride=[1, 2], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(LPPool2d(LPPool2dOptions(1, std::vector<int64_t>({3, 4})).stride({5, 6}).ceil_mode(true))),
      "torch::nn::LPPool2d(norm_type=1, kernel_size=[3, 4], stride=[5, 6], ceil_mode=true)");
}

TEST_F(ModulesTest, PrettyPrintAdaptiveMaxPool) {
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool1d(5)),
      "torch::nn::AdaptiveMaxPool1d(output_size=5)");

  const auto options = AdaptiveMaxPool1dOptions(3);
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool1d(options)),
      "torch::nn::AdaptiveMaxPool1d(output_size=3)");

  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(5)),
      "torch::nn::AdaptiveMaxPool2d(output_size=[5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(std::vector<int64_t>{5, 6})),
      "torch::nn::AdaptiveMaxPool2d(output_size=[5, 6])");

  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(5)),
      "torch::nn::AdaptiveMaxPool3d(output_size=[5, 5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(std::vector<int64_t>{5, 6, 7})),
      "torch::nn::AdaptiveMaxPool3d(output_size=[5, 6, 7])");
}

TEST_F(ModulesTest, PrettyPrintAdaptiveAvgPool) {
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool1d(5)),
      "torch::nn::AdaptiveAvgPool1d(output_size=5)");

  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(5)),
      "torch::nn::AdaptiveAvgPool2d(output_size=[5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(std::vector<int64_t>{5, 6})),
      "torch::nn::AdaptiveAvgPool2d(output_size=[5, 6])");

  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(5)),
      "torch::nn::AdaptiveAvgPool3d(output_size=[5, 5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(std::vector<int64_t>{5, 6, 7})),
      "torch::nn::AdaptiveAvgPool3d(output_size=[5, 6, 7])");
}

TEST_F(ModulesTest, PrettyPrintMaxUnpool) {
  ASSERT_EQ(
      c10::str(MaxUnpool1d(5)),
      "torch::nn::MaxUnpool1d(kernel_size=5, stride=5, padding=0)");
  ASSERT_EQ(
      c10::str(MaxUnpool1d(MaxUnpool1dOptions(5).stride(3).padding(1))),
      "torch::nn::MaxUnpool1d(kernel_size=5, stride=3, padding=1)");

  ASSERT_EQ(
      c10::str(MaxUnpool2d(5)),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(MaxUnpool2d(std::vector<int64_t>{5, 6})),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[5, 6], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(MaxUnpool2d(MaxUnpool2dOptions(std::vector<int64_t>{5, 6}).stride({3, 4}).padding({1, 2}))),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[3, 4], padding=[1, 2])");
}

TEST_F(ModulesTest, PrettyPrintDropout) {
  ASSERT_EQ(c10::str(Dropout(0.5)), "torch::nn::Dropout(rate=0.5)");
  ASSERT_EQ(
      c10::str(FeatureDropout(0.5)), "torch::nn::FeatureDropout(rate=0.5)");
}

TEST_F(ModulesTest, PrettyPrintFunctional) {
  ASSERT_EQ(c10::str(Functional(torch::relu)), "torch::nn::Functional()");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm) {
  ASSERT_EQ(
      c10::str(BatchNorm(
          BatchNormOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(
              true))),
      "torch::nn::BatchNorm(num_features=4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm1d) {
  ASSERT_EQ(
      c10::str(BatchNorm1d(
          BatchNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false)
          .track_running_stats(true))),
      "torch::nn::BatchNorm1d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm2d) {
  ASSERT_EQ(
      c10::str(BatchNorm2d(
          BatchNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false)
          .track_running_stats(true))),
      "torch::nn::BatchNorm2d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm3d) {
  ASSERT_EQ(
      c10::str(BatchNorm3d(
          BatchNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false)
          .track_running_stats(true))),
      "torch::nn::BatchNorm3d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintLayerNorm) {
  ASSERT_EQ(
    c10::str(LayerNorm(LayerNormOptions({2, 2}))),
      "torch::nn::LayerNorm([2, 2], eps=1e-05, elementwise_affine=true)");
      ASSERT_EQ(
        c10::str(LayerNorm(LayerNormOptions({2, 2}).elementwise_affine(false).eps(2e-5))),
          "torch::nn::LayerNorm([2, 2], eps=2e-05, elementwise_affine=false)");
}

TEST_F(ModulesTest, PrettyPrintLocalResponseNorm) {
  ASSERT_EQ(
    c10::str(LocalResponseNorm(LocalResponseNormOptions(2))),
      "torch::nn::LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=1)");
  ASSERT_EQ(
    c10::str(LocalResponseNorm(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.))),
      "torch::nn::LocalResponseNorm(2, alpha=0.0002, beta=0.85, k=2)");
}

TEST_F(ModulesTest, PrettyPrintEmbedding) {
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2))),
      "torch::nn::Embedding(num_embeddings=10, embedding_dim=2)");
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2).padding_idx(3).max_norm(2))),
      "torch::nn::Embedding(num_embeddings=10, embedding_dim=2, padding_idx=3, max_norm=2)");
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true))),
      "torch::nn::Embedding(num_embeddings=10, embedding_dim=2, padding_idx=3, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true)");
}

TEST_F(ModulesTest, PrettyPrintEmbeddingBag) {
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2).max_norm(2))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum))),
      "torch::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true, mode=kSum)");
}

TEST_F(ModulesTest, PrettyPrintL1Loss) {
  ASSERT_EQ(
      c10::str(L1Loss()),
      "torch::nn::L1Loss()");
}
TEST_F(ModulesTest, PrettyPrintKLDivLoss) {
  ASSERT_EQ(
      c10::str(KLDivLoss()),
      "torch::nn::KLDivLoss()");
}
TEST_F(ModulesTest, PrettyPrintMSELoss) {
  ASSERT_EQ(
      c10::str(MSELoss()),
      "torch::nn::MSELoss()");
}
TEST_F(ModulesTest, PrettyPrintBCELoss) {
  ASSERT_EQ(
      c10::str(BCELoss()),
      "torch::nn::BCELoss()");
}
TEST_F(ModulesTest, PrettyPrintHingeEmbeddingLoss) {
  ASSERT_EQ(
      c10::str(HingeEmbeddingLoss(HingeEmbeddingLossOptions().margin(4))),
      "torch::nn::HingeEmbeddingLoss(margin=4)");
}

TEST_F(ModulesTest, PrettyPrintCosineEmbeddingLoss) {
  ASSERT_EQ(
      c10::str(CosineEmbeddingLoss(CosineEmbeddingLossOptions().margin(0.25))),
      "torch::nn::CosineEmbeddingLoss(margin=0.25)");
}

TEST_F(ModulesTest, PrettyPrintTripletMarginLoss) {
  ASSERT_EQ(
      c10::str(TripletMarginLoss(TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false))),
      "torch::nn::TripletMarginLoss(margin=3, p=2, eps=1e-06, swap=false)");
}

TEST_F(ModulesTest, PrettyPrintMultiLabelMarginLoss) {
  ASSERT_EQ(c10::str(MultiLabelMarginLoss()), "torch::nn::MultiLabelMarginLoss()");
}

TEST_F(ModulesTest, PrettyPrintMultiLabelSoftMarginLoss) {
  ASSERT_EQ(c10::str(MultiLabelSoftMarginLoss()), "torch::nn::MultiLabelSoftMarginLoss()");
}

TEST_F(ModulesTest, PrettyPrintSoftMarginLoss) {
  ASSERT_EQ(c10::str(SoftMarginLoss()), "torch::nn::SoftMarginLoss()");
}

TEST_F(ModulesTest, PrettyPrintCosineSimilarity) {
  ASSERT_EQ(
      c10::str(CosineSimilarity()),
      "torch::nn::CosineSimilarity(dim=1, eps=1e-08)");
  ASSERT_EQ(
      c10::str(CosineSimilarity(CosineSimilarityOptions().dim(0).eps(0.5))),
      "torch::nn::CosineSimilarity(dim=0, eps=0.5)");
}

TEST_F(ModulesTest, PrettyPrintPairwiseDistance) {
  ASSERT_EQ(
      c10::str(PairwiseDistance()),
      "torch::nn::PairwiseDistance(p=2, eps=1e-06, keepdim=false)");
  ASSERT_EQ(
      c10::str(PairwiseDistance(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true))),
      "torch::nn::PairwiseDistance(p=3, eps=0.5, keepdim=true)");
}

TEST_F(ModulesTest, PrettyPrintReflectionPad) {
  ASSERT_EQ(
      c10::str(ReflectionPad1d(ReflectionPad1dOptions(2))),
      "torch::nn::ReflectionPad1d(padding=[2, 2])");
  ASSERT_EQ(
      c10::str(ReflectionPad1d(ReflectionPad1dOptions({3, 1}))),
      "torch::nn::ReflectionPad1d(padding=[3, 1])");
  ASSERT_EQ(
      c10::str(ReflectionPad2d(ReflectionPad2dOptions(2))),
      "torch::nn::ReflectionPad2d(padding=[2, 2, 2, 2])");
  ASSERT_EQ(
      c10::str(ReflectionPad2d(ReflectionPad2dOptions({1, 1, 2, 0}))),
      "torch::nn::ReflectionPad2d(padding=[1, 1, 2, 0])");
}

TEST_F(ModulesTest, PrettyPrintReplicationPad) {
  ASSERT_EQ(
      c10::str(ReplicationPad1d(ReplicationPad1dOptions(2))),
      "torch::nn::ReplicationPad1d(padding=[2, 2])");
  ASSERT_EQ(
      c10::str(ReplicationPad1d(ReplicationPad1dOptions({3, 1}))),
      "torch::nn::ReplicationPad1d(padding=[3, 1])");
  ASSERT_EQ(
      c10::str(ReplicationPad2d(ReplicationPad2dOptions(2))),
      "torch::nn::ReplicationPad2d(padding=[2, 2, 2, 2])");
  ASSERT_EQ(
      c10::str(ReplicationPad2d(ReplicationPad2dOptions({1, 1, 2, 0}))),
      "torch::nn::ReplicationPad2d(padding=[1, 1, 2, 0])");
  ASSERT_EQ(
      c10::str(ReplicationPad3d(ReplicationPad3dOptions(1))),
      "torch::nn::ReplicationPad3d(padding=[1, 1, 1, 1, 1, 1])");
  ASSERT_EQ(
      c10::str(ReplicationPad3d(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}))),
      "torch::nn::ReplicationPad3d(padding=[1, 2, 1, 2, 1, 2])");
}

TEST_F(ModulesTest, PrettyPrintZeroPad2d) {
  ASSERT_EQ(
      c10::str(ZeroPad2d(ZeroPad2dOptions(2))),
      "torch::nn::ZeroPad2d(padding=[2, 2, 2, 2])");
  ASSERT_EQ(
      c10::str(ZeroPad2d(ZeroPad2dOptions({1, 1, 2, 0}))),
      "torch::nn::ZeroPad2d(padding=[1, 1, 2, 0])");
}

TEST_F(ModulesTest, PrettyPrintConstantPad) {
  ASSERT_EQ(
      c10::str(ConstantPad1d(ConstantPad1dOptions(2, 3.5))),
      "torch::nn::ConstantPad1d(padding=[2, 2], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad1d(ConstantPad1dOptions({3, 1}, 3.5))),
      "torch::nn::ConstantPad1d(padding=[3, 1], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad2d(ConstantPad2dOptions(2, 3.5))),
      "torch::nn::ConstantPad2d(padding=[2, 2, 2, 2], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad2d(ConstantPad2dOptions({3, 0, 2, 1}, 3.5))),
      "torch::nn::ConstantPad2d(padding=[3, 0, 2, 1], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad3d(ConstantPad3dOptions(1, 3.5))),
      "torch::nn::ConstantPad3d(padding=[1, 1, 1, 1, 1, 1], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad3d(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5))),
      "torch::nn::ConstantPad3d(padding=[1, 2, 1, 2, 1, 2], value=3.5)");
}

TEST_F(ModulesTest, PrettyPrintNestedModel) {
  struct InnerTestModule : torch::nn::Module {
    InnerTestModule()
        : torch::nn::Module("InnerTestModule"),
          fc(register_module("fc", torch::nn::Linear(3, 4))),
          table(register_module("table", torch::nn::Embedding(10, 2))) {}

    torch::nn::Linear fc;
    torch::nn::Embedding table;
  };

  struct TestModule : torch::nn::Module {
    TestModule()
        : torch::nn::Module("TestModule"),
          fc(register_module("fc", torch::nn::Linear(4, 5))),
          table(register_module("table", torch::nn::Embedding(EmbeddingOptions(10, 2)))),
          inner(register_module("inner", std::make_shared<InnerTestModule>())) {
    }

    torch::nn::Linear fc;
    torch::nn::Embedding table;
    std::shared_ptr<InnerTestModule> inner;
  };

  ASSERT_EQ(
      c10::str(TestModule{}),
      "TestModule(\n"
      "  (fc): torch::nn::Linear(in_features=4, out_features=5, bias=true)\n"
      "  (table): torch::nn::Embedding(num_embeddings=10, embedding_dim=2)\n"
      "  (inner): InnerTestModule(\n"
      "    (fc): torch::nn::Linear(in_features=3, out_features=4, bias=true)\n"
      "    (table): torch::nn::Embedding(num_embeddings=10, embedding_dim=2)\n"
      "  )\n"
      ")");
}

TEST_F(ModulesTest, PrettyPrintELU) {
  ASSERT_EQ(c10::str(ELU()), "torch::nn::ELU(alpha=1)");
  ASSERT_EQ(c10::str(ELU(ELUOptions().alpha(42.42).inplace(true))),
            "torch::nn::ELU(alpha=42.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintSELU) {
  ASSERT_EQ(c10::str(SELU()), "torch::nn::SELU()");
  ASSERT_EQ(c10::str(SELU(SELUOptions().inplace(true))),
            "torch::nn::SELU(inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintHardshrink) {
  ASSERT_EQ(c10::str(Hardshrink()), "torch::nn::Hardshrink(0.5)");
  ASSERT_EQ(c10::str(Hardshrink(HardshrinkOptions().lambda(42.42))),
            "torch::nn::Hardshrink(42.42)");
}

TEST_F(ModulesTest, PrettyPrintHardtanh) {
  ASSERT_EQ(c10::str(Hardtanh()),
    "torch::nn::Hardtanh(min_val=-1, max_val=1)");
  ASSERT_EQ(c10::str(Hardtanh(
      HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true))),
    "torch::nn::Hardtanh(min_val=-42.42, max_val=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintLeakyReLU) {
  ASSERT_EQ(c10::str(LeakyReLU()),
    "torch::nn::LeakyReLU(negative_slope=0.01)");
  ASSERT_EQ(c10::str(LeakyReLU(
      LeakyReLUOptions().negative_slope(0.42).inplace(true))),
    "torch::nn::LeakyReLU(negative_slope=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintLogSigmoid) {
  ASSERT_EQ(c10::str(LogSigmoid()), "torch::nn::LogSigmoid()");
}

TEST_F(ModulesTest, PrettyPrintSoftmax) {
  ASSERT_EQ(c10::str(Softmax(SoftmaxOptions(1))), "torch::nn::Softmax(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintSoftmin) {
  ASSERT_EQ(c10::str(Softmin(SoftminOptions(1))), "torch::nn::Softmin(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintLogSoftmax) {
  ASSERT_EQ(c10::str(LogSoftmax(LogSoftmaxOptions(1))),
            "torch::nn::LogSoftmax(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintSoftmax2d) {
  ASSERT_EQ(c10::str(Softmax2d()), "torch::nn::Softmax2d()");
}

TEST_F(ModulesTest, PrettyPrintPReLU) {
  ASSERT_EQ(c10::str(PReLU()), "torch::nn::PReLU(num_parameters=1)");
  ASSERT_EQ(c10::str(PReLU(PReLUOptions().num_parameters(42))),
            "torch::nn::PReLU(num_parameters=42)");
}

TEST_F(ModulesTest, PrettyPrintReLU) {
  ASSERT_EQ(c10::str(ReLU()), "torch::nn::ReLU()");
  ASSERT_EQ(c10::str(ReLU(ReLUOptions().inplace(true))),
    "torch::nn::ReLU(inplace=true)");
  ASSERT_EQ(c10::str(ReLU(/*inplace=*/true)),
    "torch::nn::ReLU(inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintReLU6) {
  ASSERT_EQ(c10::str(ReLU6()), "torch::nn::ReLU6()");
  ASSERT_EQ(c10::str(ReLU6(ReLU6Options().inplace(true))),
    "torch::nn::ReLU6(inplace=true)");
  ASSERT_EQ(c10::str(ReLU6(/*inplace=*/true)),
    "torch::nn::ReLU6(inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintRReLU) {
  ASSERT_EQ(c10::str(RReLU()),
    "torch::nn::RReLU(lower=0.125, upper=0.333333)");
  ASSERT_EQ(c10::str(RReLU(
      RReLUOptions().lower(0.24).upper(0.42).inplace(true))),
    "torch::nn::RReLU(lower=0.24, upper=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintCELU) {
  ASSERT_EQ(c10::str(CELU()), "torch::nn::CELU(alpha=1)");
  ASSERT_EQ(c10::str(CELU(CELUOptions().alpha(42.42).inplace(true))),
            "torch::nn::CELU(alpha=42.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintSigmoid) {
  ASSERT_EQ(c10::str(Sigmoid()), "torch::nn::Sigmoid()");
}

TEST_F(ModulesTest, PrettyPrintPixelShuffle) {
  ASSERT_EQ(c10::str(PixelShuffle(PixelShuffleOptions(5))),
            "torch::nn::PixelShuffle(upscale_factor=5)");
}

TEST_F(ModulesTest, PrettyPrintSoftplus) {
  ASSERT_EQ(c10::str(Softplus()),
    "torch::nn::Softplus(beta=1, threshold=20)");
  ASSERT_EQ(c10::str(Softplus(
      SoftplusOptions().beta(0.24).threshold(42.42))),
    "torch::nn::Softplus(beta=0.24, threshold=42.42)");
}

TEST_F(ModulesTest, PrettyPrintSoftshrink) {
  ASSERT_EQ(c10::str(Softshrink()), "torch::nn::Softshrink(0.5)");
  ASSERT_EQ(c10::str(Softshrink(SoftshrinkOptions(42.42))),
            "torch::nn::Softshrink(42.42)");
}

TEST_F(ModulesTest, PrettyPrintSoftsign) {
  ASSERT_EQ(c10::str(Softsign()), "torch::nn::Softsign()");
}

TEST_F(ModulesTest, PrettyPrintTanh) {
  ASSERT_EQ(c10::str(Tanh()), "torch::nn::Tanh()");
}

TEST_F(ModulesTest, PrettyPrintTanhshrink) {
  ASSERT_EQ(c10::str(Tanhshrink()), "torch::nn::Tanhshrink()");
}

TEST_F(ModulesTest, PrettyPrintThreshold) {
  ASSERT_EQ(c10::str(Threshold(24.24, 42.42)),
    "torch::nn::Threshold(threshold=24.24, value=42.42)");
  ASSERT_EQ(c10::str(Threshold(
      ThresholdOptions(42.42, 24.24).inplace(true))),
    "torch::nn::Threshold(threshold=42.42, value=24.24, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintCTCLoss) {
  ASSERT_EQ(c10::str(CTCLoss()), "torch::nn::CTCLoss()");
  ASSERT_EQ(c10::str(CTCLoss(
    CTCLossOptions().blank(42).zero_infinity(false)
      .reduction(torch::kSum))), "torch::nn::CTCLoss()");
}

TEST_F(ModulesTest, PrettyPrintPoissonNLLLoss) {
  ASSERT_EQ(c10::str(PoissonNLLLoss()), "torch::nn::PoissonNLLLoss()");
  ASSERT_EQ(c10::str(PoissonNLLLoss(
    PoissonNLLLossOptions().log_input(false).full(true).eps(0.42)
    .reduction(torch::kSum))),
    "torch::nn::PoissonNLLLoss()");
}

TEST_F(ModulesTest, PrettyPrintMarginRankingLoss) {
  ASSERT_EQ(c10::str(MarginRankingLoss()), "torch::nn::MarginRankingLoss()");
  ASSERT_EQ(c10::str(MarginRankingLoss(
    MarginRankingLossOptions().margin(0.5).reduction(torch::kSum))),
    "torch::nn::MarginRankingLoss()");
}

TEST_F(ModulesTest, PrettyPrintCrossMapLRN2d) {
  ASSERT_EQ(c10::str(CrossMapLRN2d(4)),
    "torch::nn::CrossMapLRN2d(4, alpha=0.0001, beta=0.75, k=1)");
  ASSERT_EQ(c10::str(CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10))),
    "torch::nn::CrossMapLRN2d(3, alpha=1e-05, beta=0.1, k=10)");
}
