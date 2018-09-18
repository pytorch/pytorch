#include "catch_utils.hpp"

#include <torch/nn/modules.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>
#include <torch/nn/modules/sequential.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/util.h>

using namespace torch::nn;
using namespace torch::test;

using Catch::StartsWith;

CATCH_TEST_CASE("Sequential/ConstructsFromSharedPointer") {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward() {
      return value;
    }
  };
  Sequential sequential(
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
  CATCH_REQUIRE(sequential->size() == 3);
}

CATCH_TEST_CASE("Sequential/ConstructsFromConcreteType") {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward() {
      return value;
    }
  };

  Sequential sequential(M(1), M(2), M(3));
  CATCH_REQUIRE(sequential->size() == 3);
}
CATCH_TEST_CASE("Sequential/ConstructsFromModuleHolder") {
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
  CATCH_REQUIRE(sequential->size() == 3);
}

CATCH_TEST_CASE("Sequential/PushBackAddsAnElement") {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int forward() {
      return value;
    }
    int value;
  };
  Sequential sequential;
  CATCH_REQUIRE(sequential->size() == 0);
  CATCH_REQUIRE(sequential->is_empty());
  sequential->push_back(Linear(3, 4));
  CATCH_REQUIRE(sequential->size() == 1);
  sequential->push_back(std::make_shared<M>(1));
  CATCH_REQUIRE(sequential->size() == 2);
  sequential->push_back(M(2));
  CATCH_REQUIRE(sequential->size() == 3);
}

CATCH_TEST_CASE("Sequential/AccessWithAt") {
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
  CATCH_REQUIRE(sequential->size() == 3);

  // returns the correct module for a given index
  for (size_t i = 0; i < modules.size(); ++i) {
    CATCH_REQUIRE(&sequential->at<M>(i) == modules[i].get());
  }

  // throws for a bad index
  CATCH_REQUIRE_THROWS_WITH(
      sequential->at<M>(modules.size() + 1), StartsWith("Index out of range"));
  CATCH_REQUIRE_THROWS_WITH(
      sequential->at<M>(modules.size() + 1000000),
      StartsWith("Index out of range"));
}

CATCH_TEST_CASE("Sequential/AccessWithPtr") {
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
  CATCH_REQUIRE(sequential->size() == 3);

  // returns the correct module for a given index
  for (size_t i = 0; i < modules.size(); ++i) {
    CATCH_REQUIRE(sequential->ptr(i).get() == modules[i].get());
    CATCH_REQUIRE(sequential[i].get() == modules[i].get());
    CATCH_REQUIRE(sequential->ptr<M>(i).get() == modules[i].get());
  }

  // throws for a bad index
  CATCH_REQUIRE_THROWS_WITH(
      sequential->ptr(modules.size() + 1), StartsWith("Index out of range"));
  CATCH_REQUIRE_THROWS_WITH(
      sequential->ptr(modules.size() + 1000000),
      StartsWith("Index out of range"));
}

CATCH_TEST_CASE("Sequential/CallingForwardOnEmptySequentialIsDisallowed") {
  Sequential empty;
  CATCH_REQUIRE_THROWS_WITH(
      empty->forward<int>(),
      StartsWith("Cannot call forward() on an empty Sequential"));
}

CATCH_TEST_CASE("Sequential/CallingForwardChainsCorrectly") {
  struct MockModule : torch::nn::Module {
    explicit MockModule(int value) : expected(value) {}
    int expected;
    int forward(int value) {
      CATCH_REQUIRE(value == expected);
      return value + 1;
    }
  };

  Sequential sequential(MockModule{1}, MockModule{2}, MockModule{3});

  CATCH_REQUIRE(sequential->forward<int>(1) == 4);
}

CATCH_TEST_CASE("Sequential/CallingForwardWithTheWrongReturnTypeThrows") {
  struct M : public torch::nn::Module {
    int forward() {
      return 5;
    }
  };

  Sequential sequential(M{});
  CATCH_REQUIRE(sequential->forward<int>() == 5);
  CATCH_REQUIRE_THROWS_WITH(
      sequential->forward<float>(),
      StartsWith("The type of the return value "
                 "is int, but you asked for type float"));
}

CATCH_TEST_CASE("Sequential/TheReturnTypeOfForwardDefaultsToTensor") {
  struct M : public torch::nn::Module {
    torch::Tensor forward(torch::Tensor v) {
      return v;
    }
  };

  Sequential sequential(M{});
  auto variable = torch::ones({3, 3}, torch::requires_grad());
  CATCH_REQUIRE(sequential->forward(variable).equal(variable));
}

CATCH_TEST_CASE("Sequential/ForwardReturnsTheLastValue") {
  torch::manual_seed(0);
  Sequential sequential(Linear(10, 3), Linear(3, 5), Linear(5, 100));

  auto x = torch::randn({1000, 10}, torch::requires_grad());
  auto y = sequential->forward(x);
  CATCH_REQUIRE(y.ndimension() == 2);
  CATCH_REQUIRE(y.size(0) == 1000);
  CATCH_REQUIRE(y.size(1) == 100);
}

CATCH_TEST_CASE("Sequential/SanityCheckForHoldingStandardModules") {
  Sequential sequential(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm(5),
      Embedding(4, 10),
      LSTM(4, 5));
}

CATCH_TEST_CASE("Sequential/ExtendPushesModulesFromOtherSequential") {
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

  CATCH_REQUIRE(a->size() == 4);
  CATCH_REQUIRE(a[0]->as<A>());
  CATCH_REQUIRE(a[1]->as<B>());
  CATCH_REQUIRE(a[2]->as<C>());
  CATCH_REQUIRE(a[3]->as<D>());

  CATCH_REQUIRE(b->size() == 2);
  CATCH_REQUIRE(b[0]->as<C>());
  CATCH_REQUIRE(b[1]->as<D>());

  std::vector<std::shared_ptr<A>> c = {std::make_shared<A>(),
                                       std::make_shared<A>()};
  b->extend(c);

  CATCH_REQUIRE(b->size() == 4);
  CATCH_REQUIRE(b[0]->as<C>());
  CATCH_REQUIRE(b[1]->as<D>());
  CATCH_REQUIRE(b[2]->as<A>());
  CATCH_REQUIRE(b[3]->as<A>());
}

CATCH_TEST_CASE("Sequential/HasReferenceSemantics") {
  Sequential first(Linear(2, 3), Linear(4, 4), Linear(4, 5));
  Sequential second(first);

  CATCH_REQUIRE(first.get() == second.get());
  CATCH_REQUIRE(first->size() == second->size());
  CATCH_REQUIRE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const AnyModule& first, const AnyModule& second) {
        return &first == &second;
      }));
}

CATCH_TEST_CASE("Sequential/IsCloneable") {
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm(3));
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone());
  CATCH_REQUIRE(sequential->size() == clone->size());

  for (size_t i = 0; i < sequential->size(); ++i) {
    // The modules should be the same kind (type).
    CATCH_REQUIRE(sequential[i]->name() == clone[i]->name());
    // But not pointer-equal (distinct objects).
    CATCH_REQUIRE(sequential[i] != clone[i]);
  }

  // Verify that the clone is deep, i.e. parameters of modules are cloned too.

  torch::NoGradGuard no_grad;

  auto params1 = sequential->parameters();
  auto params2 = clone->parameters();
  CATCH_REQUIRE(params1.size() == params2.size());
  for (auto& param : params1) {
    CATCH_REQUIRE(!pointer_equal(param.value, params2[param.key]));
    CATCH_REQUIRE(param->device() == params2[param.key].device());
    CATCH_REQUIRE(param->allclose(params2[param.key]));
    param->add_(2);
  }
  for (auto& param : params1) {
    CATCH_REQUIRE(!param->allclose(params2[param.key]));
  }
}

CATCH_TEST_CASE("Sequential/RegistersElementsAsSubmodules") {
  Sequential sequential(Linear(10, 3), Conv2d(1, 2, 3), FeatureDropout(0.5));

  auto modules = sequential->modules();
  CATCH_REQUIRE(modules.size() == sequential->children().size());

  CATCH_REQUIRE(modules[0]->as<Linear>());
  CATCH_REQUIRE(modules[1]->as<Conv2d>());
  CATCH_REQUIRE(modules[2]->as<FeatureDropout>());
}

CATCH_TEST_CASE("Sequential/CloneToDevice", "[cuda]") {
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm(3));
  torch::Device device(torch::kCUDA, 0);
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone(device));
  for (const auto& p : clone->parameters()) {
    CATCH_REQUIRE(p->device() == device);
  }
  for (const auto& b : clone->buffers()) {
    CATCH_REQUIRE(b->device() == device);
  }
}
