#include <catch.hpp>

#include <torch/expanding_array.h>
#include <torch/nn/modules/linear.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <torch/csrc/utils/memory.h>

#include <ATen/optional.h>

using namespace torch;
using namespace torch::nn;

using Catch::StartsWith;

TEST_CASE("misc") {
  SECTION("no_grad") {
    NoGradGuard guard;
    auto model = Linear(5, 2).build();
    auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    s.backward();
    REQUIRE(!model->parameters()["weight"].grad().defined());
  }

  SECTION("CPU random seed") {
    int size = 100;
    torch::manual_seed(7);
    auto x1 = Var(at::CPU(at::kFloat).randn({size}));
    torch::manual_seed(7);
    auto x2 = Var(at::CPU(at::kFloat).randn({size}));

    auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
    REQUIRE(l_inf < 1e-10);
  }
}

TEST_CASE("misc_cuda", "[cuda]") {
  SECTION("CUDA random seed") {
    int size = 100;
    torch::manual_seed(7);
    auto x1 = Var(at::CUDA(at::kFloat).randn({size}));
    torch::manual_seed(7);
    auto x2 = Var(at::CUDA(at::kFloat).randn({size}));

    auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
    REQUIRE(l_inf < 1e-10);
  }
}

TEST_CASE("autograd") {
  auto x = autograd::make_variable(
      at::randn(at::CPU(at::kFloat), {3, 3}), /*requires_grad=*/true);
  auto y = autograd::make_variable(
      at::randn(at::CPU(at::kFloat), {3, 3}), /*requires_grad=*/false);
  auto z = x * y;
  SECTION("derivatives of zero-dim tensors") {
    z.sum().backward();
    REQUIRE(x.grad().allclose(y));
  }
  SECTION("derivatives of tensors") {
    z.backward();
    REQUIRE(x.grad().allclose(y));
  }
  SECTION("custom gradient inputs") {
    z.sum().backward(
        autograd::make_variable(at::ones(at::CPU(at::kFloat), {1}) * 2));
    REQUIRE(x.grad().allclose(y * 2));
  }
  // Assume everything else is safe from PyTorch tests.
}

TEST_CASE("expanding-array") {
  SECTION("successful construction") {
    SECTION("initializer_list") {
      ExpandingArray<5> e({1, 2, 3, 4, 5});
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("vector") {
      ExpandingArray<5> e(std::vector<int64_t>{1, 2, 3, 4, 5});
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("array") {
      ExpandingArray<5> e(std::array<int64_t, 5>({1, 2, 3, 4, 5}));
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("single value") {
      ExpandingArray<5> e(5);
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == 5);
      }
    }
  }
  SECTION("throws for incorrect size on construction") {
    SECTION("initializer_list") {
      REQUIRE_THROWS_WITH(
          ExpandingArray<5>({1, 2, 3, 4, 5, 6, 7}),
          StartsWith("Expected 5 values, but instead got 7"));
    }
    SECTION("vector") {
      REQUIRE_THROWS_WITH(
          ExpandingArray<5>(std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7})),
          StartsWith("Expected 5 values, but instead got 7"));
    }
  }
}

TEST_CASE("make_unique") {
  struct Test {
    explicit Test(const int& x) : lvalue_(x) {}
    explicit Test(int&& x) : rvalue_(x) {}

    at::optional<int> lvalue_;
    at::optional<int> rvalue_;
  };

  SECTION("forwards rvalues correctly") {
    auto ptr = torch::make_unique<Test>(123);
    REQUIRE(!ptr->lvalue_.has_value());
    REQUIRE(ptr->rvalue_.has_value());
    REQUIRE(*ptr->rvalue_ == 123);
  }

  SECTION("forwards lvalues correctly") {
    int x = 5;
    auto ptr = torch::make_unique<Test>(x);
    REQUIRE(ptr->lvalue_.has_value());
    REQUIRE(*ptr->lvalue_ == 5);
    REQUIRE(!ptr->rvalue_.has_value());
  }

  SECTION("Can construct unique_ptr of array") {
    auto ptr = torch::make_unique<int[]>(3);
    // Value initialization is required by the standard.
    REQUIRE(ptr[0] == 0);
    REQUIRE(ptr[1] == 0);
    REQUIRE(ptr[2] == 0);
  }
}
