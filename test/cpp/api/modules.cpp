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

TEST_F(ModulesTest, CosineSimilarity) {
  CosineSimilarity cos(CosineSimilarityOptions().dim(1));
  float data1[] = {1, 2, 3, 4, 5, 6};
  auto input1 = torch::from_blob(data1, {2, 3}, torch::requires_grad());
  float data2[] = {1, 8, 3, 2, 1, 6};
  auto input2 = torch::from_blob(data2, {2, 3}, torch::requires_grad());
  auto output = cos->forward(input1, input2);
  float data3[] = {0.8078, 0.8721};
  auto expected = torch::from_blob(data3, {2});
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}

TEST_F(ModulesTest, PairwiseDistance) {
  PairwiseDistance dist(PairwiseDistanceOptions(1));
  float data1[] = {1, 2, 3, 4, 5, 6};
  auto input1 = torch::from_blob(data1, {2, 3}, torch::requires_grad());
  float data2[] = {1, 8, 3, 2, 1, 6};
  auto input2 = torch::from_blob(data2, {2, 3}, torch::requires_grad());
  auto output = dist->forward(input1, input2);
  auto expected = torch::full({2}, 6);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
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
      "torch::nn::MaxPool1d(kernel_size=5, stride=5)");
  ASSERT_EQ(
      c10::str(MaxPool2d(5)),
      "torch::nn::MaxPool2d(kernel_size=[5, 5], stride=[5, 5])");
  ASSERT_EQ(
      c10::str(MaxPool2d(MaxPool2dOptions(5).stride(2))),
      "torch::nn::MaxPool2d(kernel_size=[5, 5], stride=[2, 2])");

  const auto options =
      MaxPool2dOptions(torch::IntArrayRef{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(MaxPool2d(options)),
      "torch::nn::MaxPool2d(kernel_size=[5, 6], stride=[1, 2])");
}

TEST_F(ModulesTest, PrettyPrintAvgPool) {
  ASSERT_EQ(
      c10::str(AvgPool1d(5)),
      "torch::nn::AvgPool1d(kernel_size=5, stride=5)");
  ASSERT_EQ(
      c10::str(AvgPool2d(5)),
      "torch::nn::AvgPool2d(kernel_size=[5, 5], stride=[5, 5])");
  ASSERT_EQ(
      c10::str(AvgPool2d(AvgPool2dOptions(5).stride(2))),
      "torch::nn::AvgPool2d(kernel_size=[5, 5], stride=[2, 2])");

  const auto options =
      AvgPool2dOptions(torch::IntArrayRef{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(AvgPool2d(options)),
      "torch::nn::AvgPool2d(kernel_size=[5, 6], stride=[1, 2])");
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
