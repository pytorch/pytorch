#include <catch.hpp>

#include <torch/nn/modules.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <memory>
#include <vector>

#include <test/cpp/api/util.h>

using namespace torch::nn;
using namespace torch::test;

using Catch::StartsWith;

TEST_CASE("sequential") {
  SECTION("construction from shared pointer") {
    struct M : torch::nn::Module {
      explicit M(int value_) : value(value_) {}
      int value;
      int forward() {
        return value;
      }
    };
    Sequential sequential(
        std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
    REQUIRE(sequential->size() == 3);
  }
  SECTION("construction from concrete type") {
    struct M : torch::nn::Module {
      explicit M(int value_) : value(value_) {}
      int value;
      int forward() {
        return value;
      }
    };

    Sequential sequential(M(1), M(2), M(3));
    REQUIRE(sequential->size() == 3);
  }
  SECTION("construction from module holders") {
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
    REQUIRE(sequential->size() == 3);
  }
  SECTION("push_back") {
    struct M : torch::nn::Module {
      explicit M(int value_) : value(value_) {}
      int forward() {
        return value;
      }
      int value;
    };
    Sequential sequential;
    REQUIRE(sequential->size() == 0);
    REQUIRE(sequential->is_empty());
    sequential->push_back(Linear(3, 4));
    REQUIRE(sequential->size() == 1);
    sequential->push_back(std::make_shared<M>(1));
    REQUIRE(sequential->size() == 2);
    sequential->push_back(M(2));
    REQUIRE(sequential->size() == 3);
  }
  SECTION("access") {
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
    REQUIRE(sequential->size() == 3);

    SECTION("at()") {
      SECTION("returns the correct module for a given index") {
        for (size_t i = 0; i < modules.size(); ++i) {
          REQUIRE(&sequential->at<M>(i) == modules[i].get());
        }
      }
      SECTION("throws for a bad index") {
        REQUIRE_THROWS_WITH(
            sequential->at<M>(modules.size() + 1),
            StartsWith("Index out of range"));
        REQUIRE_THROWS_WITH(
            sequential->at<M>(modules.size() + 1000000),
            StartsWith("Index out of range"));
      }
    }

    SECTION("ptr()") {
      SECTION("returns the correct module for a given index") {
        for (size_t i = 0; i < modules.size(); ++i) {
          REQUIRE(sequential->ptr(i).get() == modules[i].get());
          REQUIRE(sequential[i].get() == modules[i].get());
          REQUIRE(sequential->ptr<M>(i).get() == modules[i].get());
        }
      }
      SECTION("throws for a bad index") {
        REQUIRE_THROWS_WITH(
            sequential->ptr(modules.size() + 1),
            StartsWith("Index out of range"));
        REQUIRE_THROWS_WITH(
            sequential->ptr(modules.size() + 1000000),
            StartsWith("Index out of range"));
      }
    }
  }
  SECTION("forward") {
    SECTION("calling forward() on an empty sequential is disallowed") {
      Sequential empty;
      REQUIRE_THROWS_WITH(
          empty->forward<int>(),
          StartsWith("Cannot call forward() on an empty Sequential"));
    }

    SECTION("calling forward() on a non-empty sequential chains correctly") {
      struct MockModule : torch::nn::Module {
        explicit MockModule(int value) : expected(value) {}
        int expected;
        int forward(int value) {
          REQUIRE(value == expected);
          return value + 1;
        }
      };

      Sequential sequential(MockModule{1}, MockModule{2}, MockModule{3});

      REQUIRE(sequential->forward<int>(1) == 4);
    }

    SECTION("calling forward() with the wrong return type throws") {
      struct M : public torch::nn::Module {
        int forward() {
          return 5;
        }
      };

      Sequential sequential(M{});
      REQUIRE(sequential->forward<int>() == 5);
      REQUIRE_THROWS_WITH(
          sequential->forward<float>(),
          StartsWith("The type of the return value "
                     "is int, but you asked for type float"));
    }

    SECTION("The return type of forward() defaults to Tensor") {
      struct M : public torch::nn::Module {
        torch::Tensor forward(torch::Tensor v) {
          return v;
        }
      };

      Sequential sequential(M{});
      auto variable = torch::ones({3, 3}, torch::requires_grad());
      REQUIRE(sequential->forward(variable).equal(variable));
    }
  }

  SECTION("returns the last value") {
    torch::manual_seed(0);
    Sequential sequential(Linear(10, 3), Linear(3, 5), Linear(5, 100));

    auto x = torch::randn({1000, 10}, torch::requires_grad());
    auto y = sequential->forward(x);
    REQUIRE(y.ndimension() == 2);
    REQUIRE(y.size(0) == 1000);
    REQUIRE(y.size(1) == 100);
  }

  SECTION("can hold other important modules") {
    Sequential sequential(
        Linear(10, 3),
        Conv2d(1, 2, 3),
        Dropout(0.5),
        BatchNorm(5),
        Embedding(4, 10),
        LSTM(4, 5));
  }

  SECTION("converts at::Tensor to torch::Tensor correctly") {
    struct M : torch::nn::Module {
      torch::Tensor forward(torch::Tensor input) {
        return input;
      }
    };

    Sequential sequential(M{});
    torch::Tensor variable = torch::ones(5);
    REQUIRE(sequential->forward(variable).sum().toCFloat() == 5);

    at::Tensor tensor_that_is_actually_a_variable = variable * 2;
    REQUIRE(
        sequential->forward(tensor_that_is_actually_a_variable)
            .sum()
            .toCFloat() == 10);
  }
  SECTION("extend() pushes modules from other Sequential") {
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

    REQUIRE(a->size() == 4);
    REQUIRE(a[0]->as<A>());
    REQUIRE(a[1]->as<B>());
    REQUIRE(a[2]->as<C>());
    REQUIRE(a[3]->as<D>());

    REQUIRE(b->size() == 2);
    REQUIRE(b[0]->as<C>());
    REQUIRE(b[1]->as<D>());

    std::vector<std::shared_ptr<A>> c = {std::make_shared<A>(),
                                         std::make_shared<A>()};
    b->extend(c);

    REQUIRE(b->size() == 4);
    REQUIRE(b[0]->as<C>());
    REQUIRE(b[1]->as<D>());
    REQUIRE(b[2]->as<A>());
    REQUIRE(b[3]->as<A>());
  }
  SECTION("has reference semantics") {
    Sequential first(Linear(2, 3), Linear(4, 4), Linear(4, 5));
    Sequential second(first);

    REQUIRE(first.get() == second.get());
    REQUIRE(first->size() == second->size());
    REQUIRE(std::equal(
        first->begin(),
        first->end(),
        second->begin(),
        [](const AnyModule& first, const AnyModule& second) {
          return &first == &second;
        }));
  }
  SECTION("Is cloneable") {
    Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm(3));
    Sequential clone =
        std::dynamic_pointer_cast<SequentialImpl>(sequential->clone());
    REQUIRE(sequential->size() == clone->size());

    for (size_t i = 0; i < sequential->size(); ++i) {
      // The modules should be the same kind (type).
      REQUIRE(sequential[i]->name() == clone[i]->name());
      // But not pointer-equal (distinct objects).
      REQUIRE(sequential[i] != clone[i]);
    }

    // Verify that the clone is deep, i.e. parameters of modules are cloned too.

    torch::NoGradGuard no_grad;

    auto params1 = sequential->parameters();
    auto params2 = clone->parameters();
    REQUIRE(params1.size() == params2.size());
    for (auto& param : params1) {
      REQUIRE(!pointer_equal(param.value, params2[param.key]));
      REQUIRE(param->device() == params2[param.key].device());
      REQUIRE(param->allclose(params2[param.key]));
      param->add_(2);
    }
    for (auto& param : params1) {
      REQUIRE(!param->allclose(params2[param.key]));
    }
  }
}

TEST_CASE("sequential/clone-to-device", "[cuda]") {
  Sequential sequential(Linear(3, 4), Functional(torch::relu), BatchNorm(3));
  torch::Device device(torch::kCUDA, 0);
  Sequential clone =
      std::dynamic_pointer_cast<SequentialImpl>(sequential->clone(device));
  for (const auto& p : clone->parameters()) {
    REQUIRE(p->device() == device);
  }
  for (const auto& b : clone->buffers()) {
    REQUIRE(b->device() == device);
  }
}
