#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct SequentialTest : torch::test::SeedingFixture {};

TEST_F(SequentialTest, CanContainThings) {
  Sequential sequential(Linear(3, 4), ReLU(), BatchNorm1d(3));
}

TEST_F(SequentialTest, ConstructsFromSharedPointer) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward() {
      return value;
    }
  };
  Sequential sequential(
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
  ASSERT_EQ(sequential->size(), 3);

  Sequential sequential_named(
      {{"m1", std::make_shared<M>(1)},
       {std::string("m2"), std::make_shared<M>(2)},
       {"m3", std::make_shared<M>(3)}});
  ASSERT_EQ(sequential->size(), 3);
}

TEST_F(SequentialTest, ConstructsFromConcreteType) {
  static int copy_count;

  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    M(const M& other) : torch::nn::Module(other) {
      copy_count++;
    }
    int value;
    int forward() {
      return value;
    }
  };

  copy_count = 0;
  Sequential sequential(M(1), M(2), M(3));
  ASSERT_EQ(sequential->size(), 3);
  // NOTE: The current implementation expects each module to be copied exactly
  // once, which happens when the module is passed into `std::make_shared<T>()`.
  // TODO: Find a way to avoid copying, and then delete the copy constructor of
  // `M`.
  ASSERT_EQ(copy_count, 3);

  copy_count = 0;
  Sequential sequential_named(
      {{"m1", M(1)}, {std::string("m2"), M(2)}, {"m3", M(3)}});
  ASSERT_EQ(sequential->size(), 3);
  ASSERT_EQ(copy_count, 3);
}

TEST_F(SequentialTest, ConstructsFromModuleHolder) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };

  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  Sequential sequential(M(1), M(2), M(3));
  ASSERT_EQ(sequential->size(), 3);

  Sequential sequential_named(
      {{"m1", M(1)}, {std::string("m2"), M(2)}, {"m3", M(3)}});
  ASSERT_EQ(sequential->size(), 3);
}

TEST_F(SequentialTest, PushBackAddsAnElement) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };

  // Test unnamed submodules
  Sequential sequential;
  ASSERT_EQ(sequential->size(), 0);
  ASSERT_TRUE(sequential->is_empty());
  sequential->push_back(Linear(3, 4));
  ASSERT_EQ(sequential->size(), 1);
  sequential->push_back(std::make_shared<M>(1));
  ASSERT_EQ(sequential->size(), 2);
  sequential->push_back(M(2));
  ASSERT_EQ(sequential->size(), 3);

  // Mix named and unnamed submodules
  Sequential sequential_named;
  ASSERT_EQ(sequential_named->size(), 0);
  ASSERT_TRUE(sequential_named->is_empty());

  sequential_named->push_back(Linear(3, 4));
  ASSERT_EQ(sequential_named->size(), 1);
  ASSERT_EQ(sequential_named->named_children()[0].key(), "0");
  sequential_named->push_back(std::string("linear2"), Linear(3, 4));
  ASSERT_EQ(sequential_named->size(), 2);
  ASSERT_EQ(sequential_named->named_children()[1].key(), "linear2");

  sequential_named->push_back("shared_m1", std::make_shared<M>(1));
  ASSERT_EQ(sequential_named->size(), 3);
  ASSERT_EQ(sequential_named->named_children()[2].key(), "shared_m1");
  sequential_named->push_back(std::make_shared<M>(1));
  ASSERT_EQ(sequential_named->size(), 4);
  ASSERT_EQ(sequential_named->named_children()[3].key(), "3");

  sequential_named->push_back(M(1));
  ASSERT_EQ(sequential_named->size(), 5);
  ASSERT_EQ(sequential_named->named_children()[4].key(), "4");
  sequential_named->push_back(std::string("m2"), M(1));
  ASSERT_EQ(sequential_named->size(), 6);
  ASSERT_EQ(sequential_named->named_children()[5].key(), "m2");

  // named and unnamed AnyModule's
  Sequential sequential_any;
  auto a = torch::nn::AnyModule(torch::nn::Linear(1, 2));
  ASSERT_EQ(sequential_any->size(), 0);
  ASSERT_TRUE(sequential_any->is_empty());
  sequential_any->push_back(a);
  ASSERT_EQ(sequential_any->size(), 1);
  ASSERT_EQ(sequential_any->named_children()[0].key(), "0");
  sequential_any->push_back("fc", a);
  ASSERT_EQ(sequential_any->size(), 2);
  ASSERT_EQ(sequential_any->named_children()[1].key(), "fc");
}

TEST_F(SequentialTest, AccessWithAt) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  Sequential sequential;
  for (auto& module : modules) {
    sequential->push_back(module);
  }
  ASSERT_EQ(sequential->size(), 3);

  // returns the correct module for a given index
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(&sequential->at<M>(i), modules[i].get());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(
      sequential->at<M>(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(
      sequential->at<M>(modules.size() + 1000000), "Index out of range");
}

TEST_F(SequentialTest, AccessWithPtr) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  Sequential sequential;
  for (auto& module : modules) {
    sequential->push_back(module);
  }
  ASSERT_EQ(sequential->size(), 3);

  // returns the correct module for a given index
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(sequential->ptr(i).get(), modules[i].get());
    ASSERT_EQ(sequential[i].get(), modules[i].get());
    ASSERT_EQ(sequential->ptr<M>(i).get(), modules[i].get());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(sequential->ptr(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(
      sequential->ptr(modules.size() + 1000000), "Index out of range");
}

TEST_F(SequentialTest, CallingForwardOnEmptySequentialIsDisallowed) {
  Sequential empty;
  ASSERT_THROWS_WITH(
      empty->forward<int>(), "Cannot call forward() on an empty Sequential");
}

TEST_F(SequentialTest, CallingForwardChainsCorrectly) {
  struct MockModule : torch::nn::Module {
    explicit MockModule(int value) : expected(value) {}
    int expected;
    int forward(int value) {
      assert(value == expected);
      return value + 1;
    }
  };

  Sequential sequential(MockModule{1}, MockModule{2}, MockModule{3});

  ASSERT_EQ(sequential->forward<int>(1), 4);
}

TEST_F(SequentialTest, CallingForwardWithTheWrongReturnTypeThrows) {
  struct M : public torch::nn::Module {
    int forward() {
      return 5;
    }
  };

  Sequential sequential(M{});
  ASSERT_EQ(sequential->forward<int>(), 5);
  ASSERT_THROWS_WITH(
      sequential->forward<float>(),
      "The type of the return value is int, but you asked for type float");
}

TEST_F(SequentialTest, TheReturnTypeOfForwardDefaultsToTensor) {
  struct M : public torch::nn::Module {
    torch::Tensor forward(torch::Tensor v) {
      return v;
    }
  };

  Sequential sequential(M{});
  auto variable = torch::ones({3, 3}, torch::requires_grad());
  ASSERT_TRUE(sequential->forward(variable).equal(variable));
}

TEST_F(SequentialTest, ForwardReturnsTheLastValue) {
  torch::manual_seed(0);
  Sequential sequential(Linear(10, 3), Linear(3, 5), Linear(5, 100));

  auto x = torch::randn({1000, 10}, torch::requires_grad());
  auto y = sequential->forward(x);
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(y.size(0), 1000);
  ASSERT_EQ(y.size(1), 100);
}

TEST_F(SequentialTest, SanityCheckForHoldingStandardModules) {
  Sequential sequential(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
}

TEST_F(SequentialTest, ExtendPushesModulesFromOtherSequential) {
  struct A : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  struct B : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  struct C : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  struct D : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  Sequential a(A{}, B{});
  Sequential b(C{}, D{});
  a->extend(*b);

  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(a[0]->as<A>());
  ASSERT_TRUE(a[1]->as<B>());
  ASSERT_TRUE(a[2]->as<C>());
  ASSERT_TRUE(a[3]->as<D>());

  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());

  std::vector<std::shared_ptr<A>> c = {
      std::make_shared<A>(), std::make_shared<A>()};
  b->extend(c);

  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());
  ASSERT_TRUE(b[2]->as<A>());
  ASSERT_TRUE(b[3]->as<A>());
}

TEST_F(SequentialTest, HasReferenceSemantics) {
  Sequential first(Linear(2, 3), Linear(4, 4), Linear(4, 5));
  Sequential second(first);

  ASSERT_EQ(first.get(), second.get());
  ASSERT_EQ(first->size(), second->size());
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const AnyModule& first, const AnyModule& second) {
        return &first == &second;
      }));
}

TEST_F(SequentialTest, IsCloneable) {
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone());
  ASSERT_EQ(sequential->size(), clone->size());

  for (size_t i = 0; i < sequential->size(); ++i) {
    // The modules should be the same kind (type).
    ASSERT_EQ(sequential[i]->name(), clone[i]->name());
    // But not pointer-equal (distinct objects).
    ASSERT_NE(sequential[i], clone[i]);
  }

  // Verify that the clone is deep, i.e. parameters of modules are cloned too.

  torch::NoGradGuard no_grad;

  auto params1 = sequential->named_parameters();
  auto params2 = clone->named_parameters();
  ASSERT_EQ(params1.size(), params2.size());
  for (auto& param : params1) {
    ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
    ASSERT_EQ(param->device(), params2[param.key()].device());
    ASSERT_TRUE(param->allclose(params2[param.key()]));
    param->add_(2);
  }
  for (auto& param : params1) {
    ASSERT_FALSE(param->allclose(params2[param.key()]));
  }
}

TEST_F(SequentialTest, RegistersElementsAsSubmodules) {
  Sequential sequential(Linear(10, 3), Conv2d(1, 2, 3), Dropout2d(0.5));

  auto modules = sequential->children();
  ASSERT_TRUE(modules[0]->as<Linear>());
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  ASSERT_TRUE(modules[2]->as<Dropout2d>());
}

TEST_F(SequentialTest, CloneToDevice_CUDA) {
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  torch::Device device(torch::kCUDA, 0);
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone(device));
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

TEST_F(SequentialTest, PrettyPrintSequential) {
  Sequential sequential(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
  ASSERT_EQ(
      c10::str(sequential),
      "torch::nn::Sequential(\n"
      "  (0): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (1): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (2): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (3): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (4): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (5): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");

  Sequential sequential_named(
      {{"linear", Linear(10, 3)},
       {"conv2d", Conv2d(1, 2, 3)},
       {"dropout", Dropout(0.5)},
       {"batchnorm2d", BatchNorm2d(5)},
       {"embedding", Embedding(4, 10)},
       {"lstm", LSTM(4, 5)}});
  ASSERT_EQ(
      c10::str(sequential_named),
      "torch::nn::Sequential(\n"
      "  (linear): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (conv2d): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (dropout): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (batchnorm2d): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (embedding): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (lstm): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}

TEST_F(SequentialTest, ModuleForwardMethodOptionalArg) {
  {
    Sequential sequential(
        Identity(),
        ConvTranspose1d(ConvTranspose1dOptions(3, 2, 3).stride(1).bias(false)));
    std::dynamic_pointer_cast<ConvTranspose1dImpl>(sequential[1])
        ->weight.set_data(torch::arange(18.).reshape({3, 2, 3}));
    auto x = torch::arange(30.).reshape({2, 3, 5});
    auto y = sequential->forward(x);
    auto expected = torch::tensor(
        {{{150., 333., 552., 615., 678., 501., 276.},
          {195., 432., 714., 804., 894., 654., 357.}},
         {{420., 918., 1497., 1560., 1623., 1176., 636.},
          {600., 1287., 2064., 2154., 2244., 1599., 852.}}});
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    Sequential sequential(
        Identity(),
        ConvTranspose2d(ConvTranspose2dOptions(3, 2, 3).stride(1).bias(false)));
    std::dynamic_pointer_cast<ConvTranspose2dImpl>(sequential[1])
        ->weight.set_data(torch::arange(54.).reshape({3, 2, 3, 3}));
    auto x = torch::arange(75.).reshape({1, 3, 5, 5});
    auto y = sequential->forward(x);
    auto expected = torch::tensor(
        {{{{2250., 4629., 7140., 7311., 7482., 5133., 2640.},
           {4995., 10272., 15837., 16206., 16575., 11364., 5841.},
           {8280., 17019., 26226., 26820., 27414., 18783., 9648.},
           {9225., 18954., 29196., 29790., 30384., 20808., 10683.},
           {10170., 20889., 32166., 32760., 33354., 22833., 11718.},
           {7515., 15420., 23721., 24144., 24567., 16800., 8613.},
           {4140., 8487., 13044., 13269., 13494., 9219., 4722.}},
          {{2925., 6006., 9246., 9498., 9750., 6672., 3423.},
           {6480., 13296., 20454., 20985., 21516., 14712., 7542.},
           {10710., 21960., 33759., 34596., 35433., 24210., 12402.},
           {12060., 24705., 37944., 38781., 39618., 27045., 13842.},
           {13410., 27450., 42129., 42966., 43803., 29880., 15282.},
           {9810., 20064., 30768., 31353., 31938., 21768., 11124.},
           {5355., 10944., 16770., 17076., 17382., 11838., 6045.}}}});
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    Sequential sequential(
        Identity(),
        ConvTranspose3d(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false)));
    std::dynamic_pointer_cast<ConvTranspose3dImpl>(sequential[1])
        ->weight.set_data(torch::arange(32.).reshape({2, 2, 2, 2, 2}));
    auto x = torch::arange(16.).reshape({1, 2, 2, 2, 2});
    auto y = sequential->forward(x);
    auto expected = torch::tensor(
        {{{{{128., 280., 154.}, {304., 664., 364.}, {184., 400., 218.}},
           {{352., 768., 420.}, {832., 1808., 984.}, {496., 1072., 580.}},
           {{256., 552., 298.}, {592., 1272., 684.}, {344., 736., 394.}}},
          {{{192., 424., 234.}, {464., 1016., 556.}, {280., 608., 330.}},
           {{544., 1184., 644.}, {1280., 2768., 1496.}, {752., 1616., 868.}},
           {{384., 824., 442.}, {880., 1880., 1004.}, {504., 1072., 570.}}}}});
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    auto weight = torch::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
    Sequential sequential(Identity(), EmbeddingBag::from_pretrained(weight));
    auto x = torch::tensor({{1, 0}}, torch::kLong);
    auto y = sequential->forward(x);
    auto expected = torch::tensor({2.5000, 3.7000, 4.6500});
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    torch::manual_seed(0);

    int64_t embed_dim = 8;
    int64_t num_heads = 4;
    int64_t batch_size = 8;
    int64_t src_len = 3;
    int64_t tgt_len = 1;

    auto query = torch::ones({batch_size, tgt_len, embed_dim});
    auto key = torch::ones({batch_size, src_len, embed_dim});
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto value = key;

    Sequential sequential(MultiheadAttention(embed_dim, num_heads));
    auto output = sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1));

    auto attn_output = std::get<0>(output);
    auto attn_output_expected = torch::tensor(
        {{{0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674, -0.0056, 0.1324, 0.0922, 0.0160, -0.0934, -0.1700, 0.1663},
          {0.0674,
           -0.0056,
           0.1324,
           0.0922,
           0.0160,
           -0.0934,
           -0.1700,
           0.1663}}});
    ASSERT_TRUE(
        torch::allclose(attn_output, attn_output_expected, 1e-05, 2e-04));

    auto attn_output_weights = std::get<1>(output);
    auto attn_output_weights_expected = torch::tensor(
        {{{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}},
         {{0.3333, 0.3333, 0.3333}}});
    ASSERT_TRUE(torch::allclose(
        attn_output_weights, attn_output_weights_expected, 1e-05, 2e-04));
  }
  {
    auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
    auto x = torch::tensor({{{2, 4, 5}}}, torch::dtype(torch::kFloat));
    Sequential sequential(MaxUnpool1d(3));
    auto y = sequential->forward(x, indices);
    auto expected =
        torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat);
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    auto indices = torch::tensor(
        {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
         {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
        torch::kLong);
    auto x = torch::tensor(
        {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
         {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
        torch::dtype(torch::kFloat));
    Sequential sequential(
        MaxUnpool2d(MaxUnpool2dOptions(3).stride(2).padding(1)));
    auto y = sequential->forward(x, indices);
    auto expected = torch::tensor(
        {{{{0, 0, 0, 0, 0},
           {0, 6, 0, 8, 9},
           {0, 0, 0, 0, 0},
           {0, 16, 0, 18, 19},
           {0, 21, 0, 23, 24}}},
         {{{0, 0, 0, 0, 0},
           {0, 31, 0, 33, 34},
           {0, 0, 0, 0, 0},
           {0, 41, 0, 43, 44},
           {0, 46, 0, 48, 49}}}},
        torch::kFloat);
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    auto indices = torch::tensor({{{{{26}}}}}, torch::kLong);
    auto x = torch::tensor(
        {{{{{26}}}}}, torch::dtype(torch::kFloat).requires_grad(true));
    Sequential sequential(MaxUnpool3d(3));
    auto y = sequential->forward(x, indices);
    auto expected = torch::tensor(
        {{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
           {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
           {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}}},
        torch::kFloat);
    ASSERT_TRUE(torch::allclose(y, expected));
  }
  {
    torch::manual_seed(0);
    Sequential sequential(Identity(), RNN(2, 3));
    auto x = torch::ones({2, 3, 2});
    auto rnn_output =
        sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(x);
    auto expected_output = torch::tensor(
        {{{-0.0645, -0.7274, 0.4531},
          {-0.0645, -0.7274, 0.4531},
          {-0.0645, -0.7274, 0.4531}},
         {{-0.3970, -0.6950, 0.6009},
          {-0.3970, -0.6950, 0.6009},
          {-0.3970, -0.6950, 0.6009}}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    Sequential sequential(Identity(), LSTM(2, 3));
    auto x = torch::ones({2, 3, 2});
    auto rnn_output = sequential->forward<
        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>>(x);
    auto expected_output = torch::tensor(
        {{{-0.2693, -0.1240, 0.0744},
          {-0.2693, -0.1240, 0.0744},
          {-0.2693, -0.1240, 0.0744}},
         {{-0.3889, -0.1919, 0.1183},
          {-0.3889, -0.1919, 0.1183},
          {-0.3889, -0.1919, 0.1183}}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    Sequential sequential(Identity(), GRU(2, 3));
    auto x = torch::ones({2, 3, 2});
    auto rnn_output =
        sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(x);
    auto expected_output = torch::tensor(
        {{{-0.1134, 0.0467, 0.2336},
          {-0.1134, 0.0467, 0.2336},
          {-0.1134, 0.0467, 0.2336}},
         {{-0.1189, 0.0502, 0.2960},
          {-0.1189, 0.0502, 0.2960},
          {-0.1189, 0.0502, 0.2960}}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    Sequential sequential(Identity(), RNNCell(2, 3));
    auto x = torch::ones({2, 2});
    auto rnn_output = sequential->forward<torch::Tensor>(x);
    auto expected_output =
        torch::tensor({{-0.0645, -0.7274, 0.4531}, {-0.0645, -0.7274, 0.4531}});
    ASSERT_TRUE(torch::allclose(rnn_output, expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    Sequential sequential(Identity(), LSTMCell(2, 3));
    auto x = torch::ones({2, 2});
    auto rnn_output =
        sequential->forward<std::tuple<torch::Tensor, torch::Tensor>>(x);
    auto expected_output =
        torch::tensor({{-0.2693, -0.1240, 0.0744}, {-0.2693, -0.1240, 0.0744}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output), expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    Sequential sequential(Identity(), GRUCell(2, 3));
    auto x = torch::ones({2, 2});
    auto rnn_output = sequential->forward<torch::Tensor>(x);
    auto expected_output =
        torch::tensor({{-0.1134, 0.0467, 0.2336}, {-0.1134, 0.0467, 0.2336}});
    ASSERT_TRUE(torch::allclose(rnn_output, expected_output, 1e-05, 2e-04));
  }
}
