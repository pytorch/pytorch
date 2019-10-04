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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool1dReturnIndices) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = torch::ones({1, 1, 5}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1 ,2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));

  ASSERT_TRUE(torch::allclose(indices, torch::tensor({{{0, 2}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({1, 1, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dReturnIndices) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2 ,2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
  ASSERT_TRUE(torch::allclose(
    indices,
    torch::tensor({{{ 0,  2},
                    {10, 12}},
                   {{ 0,  2},
                    {10, 12}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({2, 2, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3dReturnIndices) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = torch::ones({2, 5, 5, 5}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));

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
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(ModulesTest, Identity) {
  Identity identity;
  auto input = torch::tensor({{1, 3, 4}, {2, 3, 4}}, torch::requires_grad());
  auto output = identity->forward(input);
  auto expected = torch::tensor({{1, 3, 4}, {2, 3, 4}}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(torch::equal(output, expected));
  ASSERT_TRUE(torch::equal(input.grad(), torch::ones_like(input)));
}

TEST_F(ModulesTest, AdaptiveMaxPool1d) {
  AdaptiveMaxPool1d model(3);
  auto x = torch::tensor({{{1, 2, 3, 4, 5}}}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool1dReturnIndices) {
  AdaptiveMaxPool1d model(3);
  auto x = torch::tensor({{{1, 2, 3, 4, 5}}}, torch::requires_grad());
  torch::Tensor y, indices;
  std::tie(y, indices) = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{2, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({{{1, 3, 4}}}, torch::kLong)));
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({1, 1, 3}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({
    {{6, 8, 9},
     {16, 18, 19},
     {21, 23, 24}},
    {{6, 8, 9},
     {16, 18, 19},
     {21, 23, 24}},
  }, torch::kLong)));
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({2, 3, 3}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 2}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(indices, torch::tensor({
    {{5, 7},
     {13, 15},
     {17, 19}},
    {{5, 7},
     {13, 15},
     {17, 19}},
  }, torch::kLong)));
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({2, 3, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 3, 3, 3}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 3, 3, 3}));

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
  ASSERT_EQ(indices.sizes(), torch::IntArrayRef({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool1d) {
  AdaptiveAvgPool1d model(3);
  auto x = torch::tensor({{{1, 2, 3, 4, 5}}}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{1.5, 3.0, 4.5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 2}));
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool1d) {
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  auto model = MaxUnpool1d{3};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y,
    torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 9}));

  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  model = MaxUnpool1d{MaxUnpool1dOptions(3).stride(2).padding(1)};
  y = model->forward(x, indices, c10::IntArrayRef({1, 1, 5}));

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(torch::allclose(y,
    torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 5}));
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
    {46, 48, 49}}}}, torch::requires_grad());
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 1, 5, 5}));
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
    unpool(output, indices, torch::IntArrayRef{1, 1, 5, 5}),
    torch::tensor({{{{ 0, 0, 0,  0, 0},
                     { 6, 0, 8,  0, 0},
                     { 0, 0, 0, 14, 0},
                     { 16, 0, 0, 0, 0},
                     { 0, 0, 0,  0, 0}}}}, torch::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool3d) {
  auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
  auto x = torch::tensor({{{{{26}}}}}, torch::requires_grad());
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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3, 3, 3}));
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
        {61, 63}}}}}, torch::requires_grad());
  auto model = MaxUnpool3d{MaxUnpool3dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices, torch::IntArrayRef({1, 1, 4, 4, 4}));

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
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 4, 4, 4}));
}

TEST_F(ModulesTest, MaxPool3d_MaxUnpool3d) {
  MaxPool3d pool {MaxPool3dOptions(3).stride(2)};
  MaxUnpool3d unpool {MaxUnpool3dOptions(3).stride(2)};
  auto input = torch::randn({20, 16, 51, 33, 15});
  torch::Tensor output, indices;
  std::tie(output, indices) = pool->forward_with_indices(input);
  auto unpooled_output = unpool(output, indices);
  ASSERT_EQ(unpooled_output.sizes(), torch::IntArrayRef({20, 16, 51, 33, 15}));
}

TEST_F(ModulesTest, Linear) {
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
}

TEST_F(ModulesTest, Fold) {
  Fold model(FoldOptions({4, 5}, {2, 2}));
  auto x = torch::randn({1, 3 * 2 * 2, 12}, torch::requires_grad());
  auto y = model(x);
  torch::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 1);
  ASSERT_EQ(y.size(1), 3);
  ASSERT_EQ(y.size(2), 4);
  ASSERT_EQ(y.size(3), 5);
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
  ASSERT_TRUE(bn->options.stateful());

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
  BatchNorm bn(BatchNormOptions(5).stateful(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());

  ASSERT_THROWS_WITH(
      bn(torch::ones({2, 5})),
      "Calling BatchNorm::forward is only permitted "
      "when the 'stateful' option is true (was false). "
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

  ASSERT_EQ(output.sizes(), torch::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HingeEmbeddingLoss) {
  HingeEmbeddingLoss loss(HingeEmbeddingLossOptions().margin(2));
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::requires_grad());
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  auto output = loss->forward(input, target);
  auto expected = torch::tensor({10}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, CosineSimilarity) {
  CosineSimilarity cos(CosineSimilarityOptions().dim(1));
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::requires_grad());
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::requires_grad());
  auto output = cos->forward(input1, input2);
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}

TEST_F(ModulesTest, PairwiseDistance) {
  PairwiseDistance dist(PairwiseDistanceOptions(1));
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::requires_grad());
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::requires_grad());
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
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
    auto y_exp = torch::max(torch::zeros_like(x), x) +
                 torch::min(torch::zeros_like(x), alpha * (torch::exp(x) - 1.0));
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
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
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
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
      ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
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
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({size, size, size}));
    auto y_exp = (x < 0) * x * negative_slope + (x >= 0) * x;
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, PrettyPrintIdentity) {
  ASSERT_EQ(c10::str(Identity()), "torch::nn::Identity()");
}

TEST_F(ModulesTest, PrettyPrintLinear) {
  ASSERT_EQ(
      c10::str(Linear(3, 4)), "torch::nn::Linear(in=3, out=4, with_bias=true)");
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
      Conv2dOptions(3, 4, torch::IntArrayRef{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(Conv2d(options)),
      "torch::nn::Conv2d(input_channels=3, output_channels=4, kernel_size=[5, 6], stride=[1, 2])");
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
      MaxPool2dOptions(torch::IntArrayRef{5, 6}).stride({1, 2});
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
      AvgPool2dOptions(torch::IntArrayRef{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(AvgPool2d(options)),
      "torch::nn::AvgPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0])");
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
      c10::str(AdaptiveMaxPool2d(torch::IntArrayRef{5, 6})),
      "torch::nn::AdaptiveMaxPool2d(output_size=[5, 6])");

  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(5)),
      "torch::nn::AdaptiveMaxPool3d(output_size=[5, 5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(torch::IntArrayRef{5, 6, 7})),
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
      c10::str(AdaptiveAvgPool2d(torch::IntArrayRef{5, 6})),
      "torch::nn::AdaptiveAvgPool2d(output_size=[5, 6])");

  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(5)),
      "torch::nn::AdaptiveAvgPool3d(output_size=[5, 5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(torch::IntArrayRef{5, 6, 7})),
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
      c10::str(MaxUnpool2d(torch::IntArrayRef{5, 6})),
      "torch::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[5, 6], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(MaxUnpool2d(MaxUnpool2dOptions(torch::IntArrayRef{5, 6}).stride({3, 4}).padding({1, 2}))),
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
          BatchNormOptions(4).eps(0.5).momentum(0.1).affine(false).stateful(
              true))),
      "torch::nn::BatchNorm(features=4, eps=0.5, momentum=0.1, affine=false, stateful=true)");
}

TEST_F(ModulesTest, PrettyPrintEmbedding) {
  ASSERT_EQ(
      c10::str(Embedding(10, 2)),
      "torch::nn::Embedding(count=10, dimension=2)");
}

TEST_F(ModulesTest, PrettyPrintHingeEmbeddingLoss) {
  ASSERT_EQ(
      c10::str(HingeEmbeddingLoss(HingeEmbeddingLossOptions().margin(4))),
      "torch::nn::HingeEmbeddingLoss(margin=4)");
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
      c10::str(PairwiseDistance(PairwiseDistanceOptions(3).eps(0.5).keepdim(true))),
      "torch::nn::PairwiseDistance(p=3, eps=0.5, keepdim=true)");
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
          table(register_module("table", torch::nn::Embedding(10, 2))),
          inner(register_module("inner", std::make_shared<InnerTestModule>())) {
    }

    torch::nn::Linear fc;
    torch::nn::Embedding table;
    std::shared_ptr<InnerTestModule> inner;
  };

  ASSERT_EQ(
      c10::str(TestModule{}),
      "TestModule(\n"
      "  (fc): torch::nn::Linear(in=4, out=5, with_bias=true)\n"
      "  (table): torch::nn::Embedding(count=10, dimension=2)\n"
      "  (inner): InnerTestModule(\n"
      "    (fc): torch::nn::Linear(in=3, out=4, with_bias=true)\n"
      "    (table): torch::nn::Embedding(count=10, dimension=2)\n"
      "  )\n"
      ")");
}

TEST_F(ModulesTest, PrettyPrintELU) {
  ASSERT_EQ(c10::str(ELU()), "torch::nn::ELU(alpha=1)");
  ASSERT_EQ(c10::str(ELU(ELUOptions().alpha(42.42).inplace(true))),
            "torch::nn::ELU(alpha=42.42, inplace=true)");
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
