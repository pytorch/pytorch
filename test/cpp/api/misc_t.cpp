#include "test.h"

CASE("misc/no_grad/1") {
  no_grad_guard guard;
  auto model = Linear(5, 2).make();
  auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
  auto y = model->forward({x})[0];
  Variable s = y.sum();

  backward(s);
  EXPECT(!model->parameters()["weight"].grad().defined());
};

CASE("misc/random/seed_cpu") {
  int size = 100;
  setSeed(7);
  auto x1 = Var(at::CPU(at::kFloat).randn({size}));
  setSeed(7);
  auto x2 = Var(at::CPU(at::kFloat).randn({size}));

  auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
  EXPECT(l_inf < 1e-10);
};

CASE("misc/random/seed_cuda") {
  CUDA_GUARD;
  int size = 100;
  setSeed(7);
  auto x1 = Var(at::CUDA(at::kFloat).randn({size}));
  setSeed(7);
  auto x2 = Var(at::CUDA(at::kFloat).randn({size}));

  auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
  EXPECT(l_inf < 1e-10);
};
