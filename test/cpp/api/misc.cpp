#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

TEST_CASE("misc") {
  SECTION("no_grad") {
    no_grad_guard guard;
    auto model = make(Linear(5, 2));
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
