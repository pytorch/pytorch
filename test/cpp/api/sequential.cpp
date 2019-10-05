#include <gtest/gtest.h>

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct SequentialTest : torch::test::SeedingFixture {};

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

  Sequential sequential_named(modules_ordered_dict({
    {"m1", std::make_shared<M>(1)},
    {std::string("m2"), std::make_shared<M>(2)},
    {"m3", std::make_shared<M>(3)}
  }));
  ASSERT_EQ(sequential->size(), 3);
}

TEST_F(SequentialTest, ConstructsFromConcreteType) {
  static int copy_count;

  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
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
  // NOTE: The current implementation expects each module to be copied exactly once,
  // which happens when the module is passed into `std::make_shared<T>()`.
  // TODO: Find a way to avoid copying, and then delete the copy constructor of `M`.
  ASSERT_EQ(copy_count, 3);

  copy_count = 0;
  Sequential sequential_named(modules_ordered_dict({
    {"m1", M(1)},
    {std::string("m2"), M(2)},
    {"m3", M(3)}
  }));
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

  Sequential sequential_named(modules_ordered_dict({
    {"m1", M(1)},
    {std::string("m2"), M(2)},
    {"m3", M(3)}
  }));
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
  for (size_t i = 0; i < modules.size(); ++i) {
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
  for (size_t i = 0; i < modules.size(); ++i) {
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
      BatchNorm(5),
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

  std::vector<std::shared_ptr<A>> c = {std::make_shared<A>(),
                                       std::make_shared<A>()};
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
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm(3));
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
  Sequential sequential(Linear(10, 3), Conv2d(1, 2, 3), FeatureDropout(0.5));

  auto modules = sequential->children();
  ASSERT_TRUE(modules[0]->as<Linear>());
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  ASSERT_TRUE(modules[2]->as<FeatureDropout>());
}

TEST_F(SequentialTest, CloneToDevice_CUDA) {
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm(3));
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
      BatchNorm(5),
      Embedding(4, 10),
      LSTM(4, 5));
  ASSERT_EQ(
      c10::str(sequential),
      "torch::nn::Sequential(\n"
      "  (0): torch::nn::Linear(in=10, out=3, with_bias=true)\n"
      "  (1): torch::nn::Conv2d(input_channels=1, output_channels=2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (2): torch::nn::Dropout(rate=0.5)\n"
      "  (3): torch::nn::BatchNorm(features=5, eps=1e-05, momentum=0.1, affine=true, stateful=true)\n"
      "  (4): torch::nn::Embedding(count=4, dimension=10)\n"
      "  (5): torch::nn::LSTM(input_size=4, hidden_size=5, layers=1, dropout=0)\n"
      ")");

  Sequential sequential_named(modules_ordered_dict({
      {"linear", Linear(10, 3)},
      {"conv2d", Conv2d(1, 2, 3)},
      {"dropout", Dropout(0.5)},
      {"batchnorm", BatchNorm(5)},
      {"embedding", Embedding(4, 10)},
      {"lstm", LSTM(4, 5)}
  }));
  ASSERT_EQ(
      c10::str(sequential_named),
      "torch::nn::Sequential(\n"
      "  (linear): torch::nn::Linear(in=10, out=3, with_bias=true)\n"
      "  (conv2d): torch::nn::Conv2d(input_channels=1, output_channels=2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (dropout): torch::nn::Dropout(rate=0.5)\n"
      "  (batchnorm): torch::nn::BatchNorm(features=5, eps=1e-05, momentum=0.1, affine=true, stateful=true)\n"
      "  (embedding): torch::nn::Embedding(count=4, dimension=10)\n"
      "  (lstm): torch::nn::LSTM(input_size=4, hidden_size=5, layers=1, dropout=0)\n"
      ")");
}
