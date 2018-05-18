#include <catch.hpp>

#include <torch/expanding_array.h>
#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

using Catch::StartsWith;

TEST_CASE("misc") {
  SECTION("no_grad") {
    no_grad_guard guard;
    auto model = Linear(5, 2).build();
    auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    REQUIRE(!model->parameters()["weight"].grad().defined());
  }

  SECTION("CPU random seed") {
    int size = 100;
    setSeed(7);
    auto x1 = Var(at::CPU(at::kFloat).randn({size}));
    setSeed(7);
    auto x2 = Var(at::CPU(at::kFloat).randn({size}));

    auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
    REQUIRE(l_inf < 1e-10);
  }
}

TEST_CASE("misc_cuda", "[cuda]") {
  SECTION("CUDA random seed") {
    int size = 100;
    setSeed(7);
    auto x1 = Var(at::CUDA(at::kFloat).randn({size}));
    setSeed(7);
    auto x2 = Var(at::CUDA(at::kFloat).randn({size}));

    auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
    REQUIRE(l_inf < 1e-10);
  }
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
