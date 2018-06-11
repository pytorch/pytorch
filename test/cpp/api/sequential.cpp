#include <catch.hpp>

#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/tensor.h>

#include <vector>

using namespace torch;
using namespace torch::nn;

using Catch::StartsWith;

TEST_CASE("sequential") {
  SECTION("construction") {
    Sequential sequential(
        Linear(2, 3).build(), Linear(2, 3), Linear(2, 3).build());
    REQUIRE(sequential.size() == 3);
  }
  SECTION("push_back") {
    Sequential sequential;
    REQUIRE(sequential.size() == 0);
    REQUIRE(sequential.is_empty());
    sequential.push_back(Linear(3, 4).build());
    REQUIRE(sequential.size() == 1);
    sequential.push_back(Linear(4, 5).build());
    REQUIRE(sequential.size() == 2);
  }
  SECTION("access") {
    std::vector<std::shared_ptr<Linear>> modules = {
        Linear(2, 3).build(), Linear(3, 4).build(), Linear(4, 5).build()};

    Sequential sequential;
    for (auto& module : modules) {
      sequential.push_back(module);
    }
    REQUIRE(sequential.size() == 3);

    SECTION("at()") {
      SECTION("returns the correct module for a given index") {
        for (size_t i = 0; i < modules.size(); ++i) {
          REQUIRE(&sequential.at<Linear>(i) == modules[i].get());
        }
      }
      SECTION("throws for a bad index") {
        REQUIRE_THROWS_WITH(
            sequential.at<Linear>(modules.size() + 1),
            StartsWith("Index out of range"));
        REQUIRE_THROWS_WITH(
            sequential.at<Linear>(modules.size() + 1000000),
            StartsWith("Index out of range"));
      }
    }

    SECTION("ptr()") {
      SECTION("returns the correct module for a given index") {
        for (size_t i = 0; i < modules.size(); ++i) {
          REQUIRE(sequential.ptr(i).get() == modules[i].get());
          REQUIRE(sequential[i].get() == modules[i].get());
          REQUIRE(sequential.ptr<Linear>(i).get() == modules[i].get());
        }
      }
      SECTION("throws for a bad index") {
        REQUIRE_THROWS_WITH(
            sequential.ptr(modules.size() + 1),
            StartsWith("Index out of range"));
        REQUIRE_THROWS_WITH(
            sequential.ptr(modules.size() + 1000000),
            StartsWith("Index out of range"));
      }
    }
  }
  SECTION("forward") {
    SECTION("calling forward() on an empty sequential is disallowed") {
      Sequential empty;
      REQUIRE_THROWS_WITH(
          empty.forward<int>(),
          StartsWith("Cannot call forward() on an empty Sequential"));
    }

    SECTION("calling forward() on a non-empty sequential chains correctly") {
      struct MockModule : nn::Module {
        explicit MockModule(int value) : expected(value) {}
        int expected;
        int forward(int value) {
          REQUIRE(value == expected);
          return value + 1;
        }
      };

      Sequential sequential(MockModule{1}, MockModule{2}, MockModule{3});

      REQUIRE(sequential.forward<int>(1) == 4);
    }

    SECTION("calling forward() with the wrong return type throws") {
      struct M : public nn::Module {
        int forward() {
          return 5;
        }
      };

      Sequential sequential(M{});
      REQUIRE(sequential.forward<int>() == 5);
      REQUIRE_THROWS_WITH(
          sequential.forward<float>(),
          StartsWith("The type of the return value "
                     "is int, but you asked for type float"));
    }

    SECTION("The return type of forward() defaults to Variable") {
      struct M : public nn::Module {
        autograd::Variable forward(autograd::Variable v) {
          return v;
        }
      };

      Sequential sequential(M{});
      auto variable =
          autograd::make_variable(at::CPU(at::kFloat).ones({3, 3}), true);
      REQUIRE(sequential.forward(variable).equal(variable));
    }
  }

  SECTION("returns the last value") {
    Sequential sequential(
        Linear(10, 3).build(), Linear(3, 5).build(), Linear(5, 100).build());

    auto x = Var(at::CPU(at::kFloat).randn({1000, 10}));
    auto y = sequential.forward<std::vector<Variable>>(std::vector<Variable>{x}).front();
    REQUIRE(y.ndimension() == 2);
    REQUIRE(y.size(0) == 1000);
    REQUIRE(y.size(1) == 100);
  }
}
