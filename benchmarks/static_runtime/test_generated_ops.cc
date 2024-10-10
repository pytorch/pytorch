// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

TEST(StaticRuntime, autogen_absolute) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::absolute(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_angle) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::angle(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sgn) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sgn(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_acos) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::acos(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arccos) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arccos(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__add_relu_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::_add_relu(%self, %other, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  auto alpha0 = 2;
  std::vector<IValue> args{self0, other0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, other1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addmv) {
  const std::string script = R"IR(
    graph(%self: Tensor, %mat: Tensor, %vec: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::addmv(%self, %mat, %vec, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2});
  auto mat0 = at::rand({2, 2});
  auto vec0 = at::rand({2});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, mat0, vec0, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({35});
  auto mat1 = at::rand({35, 35});
  auto vec1 = at::rand({35});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, mat1, vec1, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addr) {
  const std::string script = R"IR(
    graph(%self: Tensor, %vec1: Tensor, %vec2: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::addr(%self, %vec1, %vec2, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6});
  auto vec10 = at::rand({6});
  auto vec20 = at::rand({6});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, vec10, vec20, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22});
  auto vec11 = at::rand({22});
  auto vec21 = at::rand({22});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, vec11, vec21, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__test_functorch_fallback) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::_test_functorch_fallback(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_argmax) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int?, %keepdim: bool):
        %bias: None = prim::Constant()
        %ret = aten::argmax(%self, %dim, %keepdim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto keepdim0 = false;
  std::vector<IValue> args{self0, dim0, keepdim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto keepdim1 = false;
  std::vector<IValue> args2{self1, dim1, keepdim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_acosh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::acosh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2, 2, 2}) + at::ones({2, 2, 2});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({5, 5, 5}) + at::ones({5, 5, 5});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_asinh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::asinh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arcsinh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arcsinh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_atanh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::atanh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arctanh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arctanh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_asin) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::asin(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arcsin) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arcsin(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_atan) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::atan(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arctan) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arctan(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_baddbmm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %batch1: Tensor, %batch2: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::baddbmm(%self, %batch1, %batch2, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto batch10 = at::rand({6, 6, 6});
  auto batch20 = at::rand({6, 6, 6});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, batch10, batch20, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto batch11 = at::rand({22, 22, 22});
  auto batch21 = at::rand({22, 22, 22});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, batch11, batch21, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_not) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_not(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_copysign_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::copysign(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_not) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_not(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_xor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_xor(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_and) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_and(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logical_or) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logical_or(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ceil) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ceil(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_clamp_max) {
  const std::string script = R"IR(
    graph(%self: Tensor, %max: int):
        %bias: None = prim::Constant()
        %ret = aten::clamp_max(%self, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto max0 = 2;
  std::vector<IValue> args{self0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto max1 = 2;
  std::vector<IValue> args2{self1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_clamp_max_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %max: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::clamp_max(%self, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto max0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto max1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_clip) {
  const std::string script = R"IR(
    graph(%self: Tensor, %min: int?, %max: int?):
        %bias: None = prim::Constant()
        %ret = aten::clip(%self, %min, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto min0 = 2;
  auto max0 = 2;
  std::vector<IValue> args{self0, min0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto min1 = 2;
  auto max1 = 2;
  std::vector<IValue> args2{self1, min1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_complex) {
  const std::string script = R"IR(
    graph(%real: Tensor, %imag: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::complex(%real, %imag)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto real0 = at::rand({6, 6, 6});
  auto imag0 = at::rand({6, 6, 6});
  std::vector<IValue> args{real0, imag0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto real1 = at::rand({22, 22, 22});
  auto imag1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{real1, imag1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_polar) {
  const std::string script = R"IR(
    graph(%abs: Tensor, %angle: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::polar(%abs, %angle)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto abs0 = at::rand({6, 6, 6});
  auto angle0 = at::rand({6, 6, 6});
  std::vector<IValue> args{abs0, angle0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto abs1 = at::rand({22, 22, 22});
  auto angle1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{abs1, angle1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cos) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::cos(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cosh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::cosh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cumprod) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %dtype: int?):
        %bias: None = prim::Constant()
        %ret = aten::cumprod(%self, %dim, %dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{self0, dim0, dtype0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto dtype1 = at::ScalarType::Float;
  std::vector<IValue> args2{self1, dim1, dtype1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_diff) {
  const std::string script = R"IR(
    graph(%self: Tensor, %n: int, %dim: int, %prepend: Tensor?, %append: Tensor?):
        %bias: None = prim::Constant()
        %ret = aten::diff(%self, %n, %dim, %prepend, %append)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto n0 = 1;
  auto dim0 = 1;
  auto prepend0 = at::rand({6, 6, 6});
  auto append0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, n0, dim0, prepend0, append0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto n1 = 1;
  auto dim1 = 1;
  auto prepend1 = at::rand({22, 22, 22});
  auto append1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, n1, dim1, prepend1, append1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_divide_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::divide(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_true_divide_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::true_divide(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_dot) {
  const std::string script = R"IR(
    graph(%self: Tensor, %tensor: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::dot(%self, %tensor)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({16});
  auto tensor0 = at::rand({16});
  std::vector<IValue> args{self0, tensor0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({64});
  auto tensor1 = at::rand({64});
  std::vector<IValue> args2{self1, tensor1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_vdot) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::vdot(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({16});
  auto other0 = at::rand({16});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({64});
  auto other1 = at::rand({64});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_erf) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::erf(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_erfc) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::erfc(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_exp) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::exp(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_exp2) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::exp2(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_expm1) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::expm1(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_floor) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::floor(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_frac) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::frac(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gcd) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::gcd(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lcm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::lcm(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_index_copy) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %source: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::index_copy(%self, %dim, %index, %source)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2});
  auto dim0 = 0;
  auto index0 = at::randint(0, 1, {2}, at::kLong);
  auto source0 = at::rand({2});
  std::vector<IValue> args{self0, dim0, index0, source0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({32});
  auto dim1 = 0;
  auto index1 = at::randint(0, 10, {32}, at::kLong);
  auto source1 = at::rand({32});
  std::vector<IValue> args2{self1, dim1, index1, source1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isin_Tensor_Tensor) {
  const std::string script = R"IR(
    graph(%elements: Tensor, %test_elements: Tensor, %assume_unique: bool, %invert: bool):
        %bias: None = prim::Constant()
        %ret = aten::isin(%elements, %test_elements, %assume_unique, %invert)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto elements0 = at::rand({6, 6, 6});
  auto test_elements0 = at::rand({6, 6, 6});
  auto assume_unique0 = false;
  auto invert0 = false;
  std::vector<IValue> args{elements0, test_elements0, assume_unique0, invert0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto elements1 = at::rand({22, 22, 22});
  auto test_elements1 = at::rand({22, 22, 22});
  auto assume_unique1 = false;
  auto invert1 = false;
  std::vector<IValue> args2{elements1, test_elements1, assume_unique1, invert1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isin_Tensor_Scalar) {
  const std::string script = R"IR(
    graph(%elements: Tensor, %test_element: int, %assume_unique: bool, %invert: bool):
        %bias: None = prim::Constant()
        %ret = aten::isin(%elements, %test_element, %assume_unique, %invert)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto elements0 = at::rand({6, 6, 6});
  auto test_element0 = 2;
  auto assume_unique0 = false;
  auto invert0 = false;
  std::vector<IValue> args{elements0, test_element0, assume_unique0, invert0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto elements1 = at::rand({22, 22, 22});
  auto test_element1 = 2;
  auto assume_unique1 = false;
  auto invert1 = false;
  std::vector<IValue> args2{elements1, test_element1, assume_unique1, invert1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isin_Scalar_Tensor) {
  const std::string script = R"IR(
    graph(%element: int, %test_elements: Tensor, %assume_unique: bool, %invert: bool):
        %bias: None = prim::Constant()
        %ret = aten::isin(%element, %test_elements, %assume_unique, %invert)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto element0 = 2;
  auto test_elements0 = at::rand({6, 6, 6});
  auto assume_unique0 = false;
  auto invert0 = false;
  std::vector<IValue> args{element0, test_elements0, assume_unique0, invert0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto element1 = 2;
  auto test_elements1 = at::rand({22, 22, 22});
  auto assume_unique1 = false;
  auto invert1 = false;
  std::vector<IValue> args2{element1, test_elements1, assume_unique1, invert1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_kron) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::kron(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ldexp_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ldexp(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log10) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log10(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log1p) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log1p(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log2) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log2(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logaddexp) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logaddexp(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logaddexp2) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::logaddexp2(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_xlogy_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::xlogy(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__log_softmax) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %half_to_float: bool):
        %bias: None = prim::Constant()
        %ret = aten::_log_softmax(%self, %dim, %half_to_float)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto half_to_float0 = false;
  std::vector<IValue> args{self0, dim0, half_to_float0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto half_to_float1 = false;
  std::vector<IValue> args2{self1, dim1, half_to_float1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__log_softmax_backward_data) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %output: Tensor, %dim: int, %input_dtype: int):
        %bias: None = prim::Constant()
        %ret = aten::_log_softmax_backward_data(%grad_output, %output, %dim, %input_dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto output0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto input_dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{grad_output0, output0, dim0, input_dtype0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto output1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto input_dtype1 = at::ScalarType::Float;
  std::vector<IValue> args2{grad_output1, output1, dim1, input_dtype1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__logcumsumexp) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int):
        %bias: None = prim::Constant()
        %ret = aten::_logcumsumexp(%self, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  std::vector<IValue> args{self0, dim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  std::vector<IValue> args2{self1, dim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_logcumsumexp) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int):
        %bias: None = prim::Constant()
        %ret = aten::logcumsumexp(%self, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  std::vector<IValue> args{self0, dim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  std::vector<IValue> args2{self1, dim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_matrix_power) {
  const std::string script = R"IR(
    graph(%self: Tensor, %n: int):
        %bias: None = prim::Constant()
        %ret = aten::matrix_power(%self, %n)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto n0 = 1;
  std::vector<IValue> args{self0, n0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto n1 = 1;
  std::vector<IValue> args2{self1, n1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_mm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %mat2: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::mm(%self, %mat2)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({8, 8});
  auto mat20 = at::rand({8, 8});
  std::vector<IValue> args{self0, mat20};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({32, 32});
  auto mat21 = at::rand({32, 32});
  std::vector<IValue> args2{self1, mat21};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_multiply_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::multiply(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_mv) {
  const std::string script = R"IR(
    graph(%self: Tensor, %vec: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::mv(%self, %vec)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6});
  auto vec0 = at::rand({6});
  std::vector<IValue> args{self0, vec0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22});
  auto vec1 = at::rand({22});
  std::vector<IValue> args2{self1, vec1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_mvlgamma) {
  const std::string script = R"IR(
    graph(%self: Tensor, %p: int):
        %bias: None = prim::Constant()
        %ret = aten::mvlgamma(%self, %p)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto p0 = 1;
  std::vector<IValue> args{self0, p0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto p1 = 1;
  std::vector<IValue> args2{self1, p1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_rad2deg) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::rad2deg(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_deg2rad) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::deg2rad(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_reciprocal) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::reciprocal(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_neg) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::neg(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_negative) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::negative(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_round) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::round(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_round_decimals) {
  const std::string script = R"IR(
    graph(%self: Tensor, %decimals: int):
        %bias: None = prim::Constant()
        %ret = aten::round(%self, %decimals)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto decimals0 = 1;
  std::vector<IValue> args{self0, decimals0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto decimals1 = 1;
  std::vector<IValue> args2{self1, decimals1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gelu) {
  const std::string script = R"IR(
    graph(%self: Tensor, %approximate: str):
        %bias: None = prim::Constant()
        %ret = aten::gelu(%self, %approximate)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto approximate0 = "tanh";
  std::vector<IValue> args{self0, approximate0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto approximate1 = "tanh";
  std::vector<IValue> args2{self1, approximate1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gelu_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %approximate: str):
        %bias: None = prim::Constant()
        %ret = aten::gelu_backward(%grad_output, %self, %approximate)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  auto approximate0 = "tanh";
  std::vector<IValue> args{grad_output0, self0, approximate0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  auto approximate1 = "tanh";
  std::vector<IValue> args2{grad_output1, self1, approximate1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_hardshrink) {
  const std::string script = R"IR(
    graph(%self: Tensor, %lambd: int):
        %bias: None = prim::Constant()
        %ret = aten::hardshrink(%self, %lambd)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto lambd0 = 2;
  std::vector<IValue> args{self0, lambd0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto lambd1 = 2;
  std::vector<IValue> args2{self1, lambd1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_hardshrink_backward) {
  const std::string script = R"IR(
    graph(%grad_out: Tensor, %self: Tensor, %lambd: int):
        %bias: None = prim::Constant()
        %ret = aten::hardshrink_backward(%grad_out, %self, %lambd)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_out0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  auto lambd0 = 2;
  std::vector<IValue> args{grad_out0, self0, lambd0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_out1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  auto lambd1 = 2;
  std::vector<IValue> args2{grad_out1, self1, lambd1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_rsqrt) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::rsqrt(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_silu) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::silu(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_silu_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::silu_backward(%grad_output, %self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{grad_output0, self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{grad_output1, self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_mish) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::mish(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sin) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sin(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sinc) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sinc(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sinh) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sinh(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__softmax) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %half_to_float: bool):
        %bias: None = prim::Constant()
        %ret = aten::_softmax(%self, %dim, %half_to_float)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto half_to_float0 = false;
  std::vector<IValue> args{self0, dim0, half_to_float0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto half_to_float1 = false;
  std::vector<IValue> args2{self1, dim1, half_to_float1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__softmax_backward_data) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %output: Tensor, %dim: int, %input_dtype: int):
        %bias: None = prim::Constant()
        %ret = aten::_softmax_backward_data(%grad_output, %output, %dim, %input_dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto output0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto input_dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{grad_output0, output0, dim0, input_dtype0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto output1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto input_dtype1 = at::ScalarType::Float;
  std::vector<IValue> args2{grad_output1, output1, dim1, input_dtype1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sqrt) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sqrt(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_square) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::square(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_prod) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dtype: int?):
        %bias: None = prim::Constant()
        %ret = aten::prod(%self, %dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{self0, dtype0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({22, 22, 22});
  auto dtype1 = at::ScalarType::Float;
  std::vector<IValue> args2{self1, dtype1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_prod_dim_int) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %keepdim: bool, %dtype: int?):
        %bias: None = prim::Constant()
        %ret = aten::prod(%self, %dim, %keepdim, %dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto keepdim0 = false;
  auto dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{self0, dim0, keepdim0, dtype0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto keepdim1 = false;
  auto dtype1 = at::ScalarType::Float;
  std::vector<IValue> args2{self1, dim1, keepdim1, dtype1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_tan) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::tan(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_threshold) {
  const std::string script = R"IR(
    graph(%self: Tensor, %threshold: int, %value: int):
        %bias: None = prim::Constant()
        %ret = aten::threshold(%self, %threshold, %value)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto threshold0 = 2;
  auto value0 = 2;
  std::vector<IValue> args{self0, threshold0, value0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto threshold1 = 2;
  auto value1 = 2;
  std::vector<IValue> args2{self1, threshold1, value1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_threshold_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %threshold: int):
        %bias: None = prim::Constant()
        %ret = aten::threshold_backward(%grad_output, %self, %threshold)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  auto threshold0 = 2;
  std::vector<IValue> args{grad_output0, self0, threshold0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  auto threshold1 = 2;
  std::vector<IValue> args2{grad_output1, self1, threshold1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_trunc) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::trunc(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_fix) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::fix(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_nuclear_norm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %keepdim: bool):
        %bias: None = prim::Constant()
        %ret = aten::nuclear_norm(%self, %keepdim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({8, 8});
  auto keepdim0 = false;
  std::vector<IValue> args{self0, keepdim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({32, 32});
  auto keepdim1 = false;
  std::vector<IValue> args2{self1, keepdim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_subtract_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::subtract(%self, %other, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  auto alpha0 = 2;
  std::vector<IValue> args{self0, other0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, other1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_heaviside) {
  const std::string script = R"IR(
    graph(%self: Tensor, %values: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::heaviside(%self, %values)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto values0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, values0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto values1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, values1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__addmm_activation) {
  const std::string script = R"IR(
    graph(%self: Tensor, %mat1: Tensor, %mat2: Tensor, %beta: int, %alpha: int, %use_gelu: bool):
        %bias: None = prim::Constant()
        %ret = aten::_addmm_activation(%self, %mat1, %mat2, %beta, %alpha, %use_gelu)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({8, 8});
  auto mat10 = at::rand({8, 8});
  auto mat20 = at::rand({8, 8});
  auto beta0 = 2;
  auto alpha0 = 2;
  auto use_gelu0 = false;
  std::vector<IValue> args{self0, mat10, mat20, beta0, alpha0, use_gelu0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({32, 32});
  auto mat11 = at::rand({32, 32});
  auto mat21 = at::rand({32, 32});
  auto beta1 = 2;
  auto alpha1 = 2;
  auto use_gelu1 = false;
  std::vector<IValue> args2{self1, mat11, mat21, beta1, alpha1, use_gelu1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_index_add) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %source: Tensor, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::index_add(%self, %dim, %index, %source, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2});
  auto dim0 = 0;
  auto index0 = at::randint(0, 1, {2}, at::kInt);
  auto source0 = at::rand({2});
  auto alpha0 = 2;
  std::vector<IValue> args{self0, dim0, index0, source0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({16});
  auto dim1 = 0;
  auto index1 = at::randint(0, 10, {16}, at::kInt);
  auto source1 = at::rand({16});
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, dim1, index1, source1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_scatter_src) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %src: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::scatter(%self, %dim, %index, %src)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  auto dim0 = 1;
  auto index0 = at::randint(0, 1, {2, 2, 2}, torch::kInt64);
  auto src0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  std::vector<IValue> args{self0, dim0, index0, src0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  auto dim1 = 1;
  auto index1 = at::randint(0, 1, {5, 5, 5}, torch::kInt64);
  auto src1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  std::vector<IValue> args2{self1, dim1, index1, src1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_scatter_value) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %value: int):
        %bias: None = prim::Constant()
        %ret = aten::scatter(%self, %dim, %index, %value)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  auto dim0 = 1;
  auto index0 = at::randint(0, 1, {2, 2, 2}, torch::kInt64);
  auto value0 = 2;
  auto src0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  std::vector<IValue> args{self0, dim0, index0, value0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  auto dim1 = 1;
  auto index1 = at::randint(0, 1, {5, 5, 5}, torch::kInt64);
  auto value1 = 2;
  auto src1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  std::vector<IValue> args2{self1, dim1, index1, value1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_scatter_reduce) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %src: Tensor, %reduce: str):
        %bias: None = prim::Constant()
        %ret = aten::scatter(%self, %dim, %index, %src, %reduce)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  auto dim0 = 1;
  auto index0 = at::randint(0, 1, {2, 2, 2}, torch::kInt64);
  auto src0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  auto reduce0 = "add";
  std::vector<IValue> args{self0, dim0, index0, src0, reduce0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  auto dim1 = 1;
  auto index1 = at::randint(0, 1, {5, 5, 5}, torch::kInt64);
  auto src1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  auto reduce1 = "add";
  std::vector<IValue> args2{self1, dim1, index1, src1, reduce1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_scatter_value_reduce) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %value: int, %reduce: str):
        %bias: None = prim::Constant()
        %ret = aten::scatter(%self, %dim, %index, %value, %reduce)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  auto dim0 = 1;
  auto index0 = at::randint(0, 1, {2, 2, 2}, torch::kInt64);
  auto value0 = 2;
  auto reduce0 = "add";
  auto src0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  std::vector<IValue> args{self0, dim0, index0, value0, reduce0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  auto dim1 = 1;
  auto index1 = at::randint(0, 1, {5, 5, 5}, torch::kInt64);
  auto value1 = 2;
  auto reduce1 = "add";
  auto src1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  std::vector<IValue> args2{self1, dim1, index1, value1, reduce1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_scatter_add) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %src: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::scatter_add(%self, %dim, %index, %src)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  auto dim0 = 1;
  auto index0 = at::randint(0, 1, {2, 2, 2}, torch::kInt64);
  auto src0 = at::randint(1, 100, {2, 2, 2}, torch::kInt64);
  std::vector<IValue> args{self0, dim0, index0, src0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  auto dim1 = 1;
  auto index1 = at::randint(0, 1, {5, 5, 5}, torch::kInt64);
  auto src1 = at::randint(1, 100, {5, 5, 5}, torch::kInt64);
  std::vector<IValue> args2{self1, dim1, index1, src1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_scatter_reduce_two) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %src: Tensor, %reduce: str, %include_self: bool):
        %bias: None = prim::Constant()
        %ret = aten::scatter_reduce(%self, %dim, %index, %src, %reduce, %include_self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  auto index0 = at::randint(6, {6, 6, 6}, torch::kInt64);
  auto src0 = at::rand({6, 6, 6});
  auto reduce0 = "mean";
  auto include_self0 = false;
  std::vector<IValue> args{self0, dim0, index0, src0, reduce0, include_self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  auto index1 = at::randint(22, {22, 22, 22}, torch::kInt64);
  auto src1 = at::rand({22, 22, 22});
  auto reduce1 = "mean";
  auto include_self1 = false;
  std::vector<IValue> args2{self1, dim1, index1, src1, reduce1, include_self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_eq_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: int):
        %bias: None = prim::Constant()
        %ret = aten::eq(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = 2;
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = 2;
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_eq_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::eq(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_and_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_and(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_or_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_or(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_xor_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_xor(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 100, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 100, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_left_shift_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_left_shift(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 1 << 4, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 26, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 1 << 4, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 26, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_bitwise_right_shift_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_right_shift(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1 << 21, 1 << 30, {6, 6, 6}, at::kInt);
  auto other0 = at::randint(1, 22, {6, 6, 6}, at::kInt);
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1 << 21, 1 << 30, {22, 22, 22}, at::kInt);
  auto other1 = at::randint(1, 22, {22, 22, 22}, at::kInt);
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_tril) {
  const std::string script = R"IR(
    graph(%self: Tensor, %diagonal: int):
        %bias: None = prim::Constant()
        %ret = aten::tril(%self, %diagonal)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto diagonal0 = 1;
  std::vector<IValue> args{self0, diagonal0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto diagonal1 = 1;
  std::vector<IValue> args2{self1, diagonal1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_triu) {
  const std::string script = R"IR(
    graph(%self: Tensor, %diagonal: int):
        %bias: None = prim::Constant()
        %ret = aten::triu(%self, %diagonal)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto diagonal0 = 1;
  std::vector<IValue> args{self0, diagonal0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto diagonal1 = 1;
  std::vector<IValue> args2{self1, diagonal1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_digamma) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::digamma(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lerp_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %end: Tensor, %weight: int):
        %bias: None = prim::Constant()
        %ret = aten::lerp(%self, %end, %weight)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto end0 = at::rand({6, 6, 6});
  auto weight0 = 2;
  std::vector<IValue> args{self0, end0, weight0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto end1 = at::rand({22, 22, 22});
  auto weight1 = 2;
  std::vector<IValue> args2{self1, end1, weight1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lerp_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %end: Tensor, %weight: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::lerp(%self, %end, %weight)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto end0 = at::rand({6, 6, 6});
  auto weight0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, end0, weight0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto end1 = at::rand({22, 22, 22});
  auto weight1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, end1, weight1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addbmm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %batch1: Tensor, %batch2: Tensor, %beta: int, %alpha: int):
        %bias: None = prim::Constant()
        %ret = aten::addbmm(%self, %batch1, %batch2, %beta, %alpha)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6});
  auto batch10 = at::rand({6, 6, 6});
  auto batch20 = at::rand({6, 6, 6});
  auto beta0 = 2;
  auto alpha0 = 2;
  std::vector<IValue> args{self0, batch10, batch20, beta0, alpha0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22});
  auto batch11 = at::rand({22, 22, 22});
  auto batch21 = at::rand({22, 22, 22});
  auto beta1 = 2;
  auto alpha1 = 2;
  std::vector<IValue> args2{self1, batch11, batch21, beta1, alpha1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_diag) {
  const std::string script = R"IR(
    graph(%self: Tensor, %diagonal: int):
        %bias: None = prim::Constant()
        %ret = aten::diag(%self, %diagonal)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({8, 8});
  auto diagonal0 = 1;
  std::vector<IValue> args{self0, diagonal0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({32, 32});
  auto diagonal1 = 1;
  std::vector<IValue> args2{self1, diagonal1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cross) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor, %dim: int?):
        %bias: None = prim::Constant()
        %ret = aten::cross(%self, %other, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({3, 3, 3});
  auto other0 = at::rand({3, 3, 3});
  auto dim0 = 1;
  std::vector<IValue> args{self0, other0, dim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 3, 22});
  auto other1 = at::rand({22, 3, 22});
  auto dim1 = 1;
  std::vector<IValue> args2{self1, other1, dim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ne_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: int):
        %bias: None = prim::Constant()
        %ret = aten::ne(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = 2;
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = 2;
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ne_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ne(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ge_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: int):
        %bias: None = prim::Constant()
        %ret = aten::ge(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = 2;
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = 2;
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ge_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ge(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_le_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: int):
        %bias: None = prim::Constant()
        %ret = aten::le(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = 2;
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = 2;
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_le_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::le(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gt_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: int):
        %bias: None = prim::Constant()
        %ret = aten::gt(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = 2;
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = 2;
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gt_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::gt(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lt_Scalar) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: int):
        %bias: None = prim::Constant()
        %ret = aten::lt(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = 2;
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = 2;
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lt_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::lt(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_take) {
  const std::string script = R"IR(
    graph(%self: Tensor, %index: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::take(%self, %index)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto index0 = at::randint(0, 216, {20}, torch::kInt64);
  std::vector<IValue> args{self0, index0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto index1 = at::randint(0, 1000, {100}, torch::kInt64);
  std::vector<IValue> args2{self1, index1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_take_along_dim) {
  const std::string script = R"IR(
    graph(%self: Tensor, %indices: Tensor, %dim: int?):
        %bias: None = prim::Constant()
        %ret = aten::take_along_dim(%self, %indices, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto indices0 = at::argsort(self0, 1, true);
  auto dim0 = 1;
  std::vector<IValue> args{self0, indices0, dim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto indices1 = at::argsort(self1, 1, true);
  auto dim1 = 1;
  std::vector<IValue> args2{self1, indices1, dim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_masked_select) {
  const std::string script = R"IR(
    graph(%self: Tensor, %mask: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::masked_select(%self, %mask)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto mask0 = at::randn({6, 6, 6}) > 0.5;
  std::vector<IValue> args{self0, mask0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto mask1 = at::rand({22, 22, 22}) > 0.5;
  std::vector<IValue> args2{self1, mask1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_nonzero_static) {
  const std::string script = R"IR(
    graph(%self: Tensor, %size: int, %fill_value: int):
        %bias: None = prim::Constant()
        %ret = aten::nonzero_static(%self, %size, %fill_value)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto size0 = 1;
  auto fill_value0 = 1;
  std::vector<IValue> args{self0, size0, fill_value0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto size1 = 1;
  auto fill_value1 = 1;
  std::vector<IValue> args2{self1, size1, fill_value1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_gather) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor, %sparse_grad: bool):
        %bias: None = prim::Constant()
        %ret = aten::gather(%self, %dim, %index, %sparse_grad)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(1, 100, {2, 2, 2}, at::kInt);
  auto dim0 = 1;
  auto index0 = at::randint(0, 1, {2, 2, 2}, torch::kInt64);
  auto sparse_grad0 = false;
  std::vector<IValue> args{self0, dim0, index0, sparse_grad0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(1, 100, {5, 5, 5}, at::kInt);
  auto dim1 = 1;
  auto index1 = at::randint(0, 4, {5, 5, 5}, torch::kInt64);
  auto sparse_grad1 = false;
  std::vector<IValue> args2{self1, dim1, index1, sparse_grad1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addcmul) {
  const std::string script = R"IR(
    graph(%self: Tensor, %tensor1: Tensor, %tensor2: Tensor, %value: int):
        %bias: None = prim::Constant()
        %ret = aten::addcmul(%self, %tensor1, %tensor2, %value)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto tensor10 = at::rand({6, 6, 6});
  auto tensor20 = at::rand({6, 6, 6});
  auto value0 = 2;
  std::vector<IValue> args{self0, tensor10, tensor20, value0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto tensor11 = at::rand({22, 22, 22});
  auto tensor21 = at::rand({22, 22, 22});
  auto value1 = 2;
  std::vector<IValue> args2{self1, tensor11, tensor21, value1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_addcdiv) {
  const std::string script = R"IR(
    graph(%self: Tensor, %tensor1: Tensor, %tensor2: Tensor, %value: int):
        %bias: None = prim::Constant()
        %ret = aten::addcdiv(%self, %tensor1, %tensor2, %value)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto tensor10 = at::rand({6, 6, 6});
  auto tensor20 = at::rand({6, 6, 6});
  auto value0 = 2;
  std::vector<IValue> args{self0, tensor10, tensor20, value0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto tensor11 = at::rand({22, 22, 22});
  auto tensor21 = at::rand({22, 22, 22});
  auto value1 = 2;
  std::vector<IValue> args2{self1, tensor11, tensor21, value1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_solve_triangular) {
  const std::string script = R"IR(
    graph(%self: Tensor, %B: Tensor, %upper: bool, %left: bool, %unitriangular: bool):
        %bias: None = prim::Constant()
        %ret = aten::linalg_solve_triangular(%self, %B, %upper, %left, %unitriangular)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto B0 = at::rand({6, 6, 6});
  auto upper0 = false;
  auto left0 = false;
  auto unitriangular0 = false;
  std::vector<IValue> args{self0, B0, upper0, left0, unitriangular0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto B1 = at::rand({22, 22, 22});
  auto upper1 = false;
  auto left1 = false;
  auto unitriangular1 = false;
  std::vector<IValue> args2{self1, B1, upper1, left1, unitriangular1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cholesky_solve) {
  const std::string script = R"IR(
    graph(%self: Tensor, %input2: Tensor, %upper: bool):
        %bias: None = prim::Constant()
        %ret = aten::cholesky_solve(%self, %input2, %upper)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto input20 = at::rand({6, 6, 6});
  auto upper0 = false;
  std::vector<IValue> args{self0, input20, upper0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto input21 = at::rand({22, 22, 22});
  auto upper1 = false;
  std::vector<IValue> args2{self1, input21, upper1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_cholesky_inverse) {
  const std::string script = R"IR(
    graph(%self: Tensor, %upper: bool):
        %bias: None = prim::Constant()
        %ret = aten::cholesky_inverse(%self, %upper)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto upper0 = false;
  std::vector<IValue> args{self0, upper0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto upper1 = false;
  std::vector<IValue> args2{self1, upper1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_orgqr) {
  const std::string script = R"IR(
    graph(%self: Tensor, %input2: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::orgqr(%self, %input2)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto input20 = at::rand({6, 6});
  std::vector<IValue> args{self0, input20};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto input21 = at::rand({22, 22});
  std::vector<IValue> args2{self1, input21};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_ormqr) {
  const std::string script = R"IR(
    graph(%self: Tensor, %input2: Tensor, %input3: Tensor, %left: bool, %transpose: bool):
        %bias: None = prim::Constant()
        %ret = aten::ormqr(%self, %input2, %input3, %left, %transpose)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto input20 = at::rand({6, 6});
  auto input30 = at::rand({6, 6, 6});
  auto left0 = false;
  auto transpose0 = false;
  std::vector<IValue> args{self0, input20, input30, left0, transpose0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto input21 = at::rand({22, 22});
  auto input31 = at::rand({22, 22, 22});
  auto left1 = false;
  auto transpose1 = false;
  std::vector<IValue> args2{self1, input21, input31, left1, transpose1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_lgamma) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::lgamma(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_polygamma) {
  const std::string script = R"IR(
    graph(%n: int, %self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::polygamma(%n, %self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto n0 = 1;
  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{n0, self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto n1 = 1;
  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{n1, self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_erfinv) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::erfinv(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_i0) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::i0(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_signbit) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::signbit(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_atan2) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::atan2(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_arctan2) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::arctan2(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_histc) {
  const std::string script = R"IR(
    graph(%self: Tensor, %bins: int, %min: int, %max: int):
        %bias: None = prim::Constant()
        %ret = aten::histc(%self, %bins, %min, %max)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto bins0 = 1;
  auto min0 = 2;
  auto max0 = 2;
  std::vector<IValue> args{self0, bins0, min0, max0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({22, 22, 22});
  auto bins1 = 1;
  auto min1 = 2;
  auto max1 = 2;
  std::vector<IValue> args2{self1, bins1, min1, max1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_hypot) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::hypot(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_igamma) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::igamma(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_igammac) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::igammac(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_nextafter) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::nextafter(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_fmin) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::fmin(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_fmax) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::fmax(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_maximum) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::maximum(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_minimum) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::minimum(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_min_other) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::min(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_quantile) {
  const std::string script = R"IR(
    graph(%self: Tensor, %q: Tensor, %dim: int?, %keepdim: bool, %interpolation: str):
        %bias: None = prim::Constant()
        %ret = aten::quantile(%self, %q, %dim, %keepdim, %interpolation)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto q0 = at::rand({6});
  auto dim0 = 1;
  auto keepdim0 = false;
  auto interpolation0 = "linear";
  std::vector<IValue> args{self0, q0, dim0, keepdim0, interpolation0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto q1 = at::rand({22});
  auto dim1 = 1;
  auto keepdim1 = false;
  auto interpolation1 = "linear";
  std::vector<IValue> args2{self1, q1, dim1, keepdim1, interpolation1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_nanquantile) {
  const std::string script = R"IR(
    graph(%self: Tensor, %q: Tensor, %dim: int?, %keepdim: bool, %interpolation: str):
        %bias: None = prim::Constant()
        %ret = aten::nanquantile(%self, %q, %dim, %keepdim, %interpolation)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto q0 = at::rand({6});
  auto dim0 = 1;
  auto keepdim0 = false;
  auto interpolation0 = "linear";
  std::vector<IValue> args{self0, q0, dim0, keepdim0, interpolation0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto q1 = at::rand({22});
  auto dim1 = 1;
  auto keepdim1 = false;
  auto interpolation1 = "linear";
  std::vector<IValue> args2{self1, q1, dim1, keepdim1, interpolation1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_msort) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::msort(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_renorm) {
  const std::string script = R"IR(
    graph(%self: Tensor, %p: int, %dim: int, %maxnorm: int):
        %bias: None = prim::Constant()
        %ret = aten::renorm(%self, %p, %dim, %maxnorm)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto p0 = 2;
  auto dim0 = 1;
  auto maxnorm0 = 2;
  std::vector<IValue> args{self0, p0, dim0, maxnorm0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto p1 = 2;
  auto dim1 = 1;
  auto maxnorm1 = 2;
  std::vector<IValue> args2{self1, p1, dim1, maxnorm1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__convert_indices_from_coo_to_csr) {
  const std::string script = R"IR(
    graph(%self: Tensor, %size: int, %out_int32: bool):
        %bias: None = prim::Constant()
        %ret = aten::_convert_indices_from_coo_to_csr(%self, %size, %out_int32)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randint(0, 3, {2}, at::kInt);
  auto size0 = 10;
  auto out_int320 = false;
  std::vector<IValue> args{self0, size0, out_int320};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::randint(0, 3, {12}, at::kInt);
  auto size1 = 24;
  auto out_int321 = false;
  std::vector<IValue> args2{self1, size1, out_int321};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen__convert_indices_from_csr_to_coo) {
  const std::string script = R"IR(
    graph(%crow_indices: Tensor, %col_indices: Tensor, %out_int32: bool, %transpose: bool):
        %bias: None = prim::Constant()
        %ret = aten::_convert_indices_from_csr_to_coo(%crow_indices, %col_indices, %out_int32, %transpose)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto crow_indices0 = torch::tensor({1}, torch::kInt32);
  auto col_indices0 = torch::tensor({0, 1, 0}, torch::kInt32);
  auto out_int320 = false;
  auto transpose0 = false;
  std::vector<IValue> args{crow_indices0, col_indices0, out_int320, transpose0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto crow_indices1 = torch::tensor({0}, torch::kInt32);
  auto col_indices1 =
      torch::tensor({0, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2}, torch::kInt32);
  auto out_int321 = false;
  auto transpose1 = false;
  std::vector<IValue> args2{
      crow_indices1, col_indices1, out_int321, transpose1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_mse_loss) {
  const std::string script = R"IR(
    graph(%self: Tensor, %target: Tensor, %reduction: int):
        %bias: None = prim::Constant()
        %ret = aten::mse_loss(%self, %target, %reduction)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto target0 = at::rand({6, 6, 6});
  auto reduction0 = 1;
  std::vector<IValue> args{self0, target0, reduction0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto target1 = at::rand({22, 22, 22});
  auto reduction1 = 1;
  std::vector<IValue> args2{self1, target1, reduction1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_multi_margin_loss) {
  const std::string script = R"IR(
    graph(%self: Tensor, %target: Tensor, %p: int, %margin: int, %weight: Tensor?, %reduction: int):
        %bias: None = prim::Constant()
        %ret = aten::multi_margin_loss(%self, %target, %p, %margin, %weight, %reduction)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6});
  auto target0 = at::randint(6, {6}, torch::kInt64);
  auto p0 = 2;
  auto margin0 = 2;
  auto weight0 = at::rand({6});
  auto reduction0 = 1;
  std::vector<IValue> args{self0, target0, p0, margin0, weight0, reduction0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({22, 22});
  auto target1 = at::randint(22, {22}, torch::kInt64);
  auto p1 = 2;
  auto margin1 = 2;
  auto weight1 = at::rand({22});
  auto reduction1 = 1;
  std::vector<IValue> args2{self1, target1, p1, margin1, weight1, reduction1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_multilabel_margin_loss) {
  const std::string script = R"IR(
    graph(%self: Tensor, %target: Tensor, %reduction: int):
        %bias: None = prim::Constant()
        %ret = aten::multilabel_margin_loss(%self, %target, %reduction)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6});
  auto target0 = at::randint(6, {6, 6}, torch::kInt64);
  auto reduction0 = 1;
  std::vector<IValue> args{self0, target0, reduction0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);

  auto self1 = at::rand({22, 22});
  auto target1 = at::randint(22, {22, 22}, torch::kInt64);
  auto reduction1 = 1;
  std::vector<IValue> args2{self1, target1, reduction1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
}

TEST(StaticRuntime, autogen_soft_margin_loss) {
  const std::string script = R"IR(
    graph(%self: Tensor, %target: Tensor, %reduction: int):
        %bias: None = prim::Constant()
        %ret = aten::soft_margin_loss(%self, %target, %reduction)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto target0 = at::rand({6, 6, 6});
  auto reduction0 = 1;
  std::vector<IValue> args{self0, target0, reduction0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto target1 = at::rand({22, 22, 22});
  auto reduction1 = 1;
  std::vector<IValue> args2{self1, target1, reduction1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_elu) {
  const std::string script = R"IR(
    graph(%self: Tensor, %alpha: int, %scale: int, %input_scale: int):
        %bias: None = prim::Constant()
        %ret = aten::elu(%self, %alpha, %scale, %input_scale)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto alpha0 = 2;
  auto scale0 = 2;
  auto input_scale0 = 2;
  std::vector<IValue> args{self0, alpha0, scale0, input_scale0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto alpha1 = 2;
  auto scale1 = 2;
  auto input_scale1 = 2;
  std::vector<IValue> args2{self1, alpha1, scale1, input_scale1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_elu_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %alpha: int, %scale: int, %input_scale: int, %is_result: bool, %self_or_result: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::elu_backward(%grad_output, %alpha, %scale, %input_scale, %is_result, %self_or_result)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto alpha0 = 2;
  auto scale0 = 2;
  auto input_scale0 = 2;
  auto is_result0 = false;
  auto self_or_result0 = at::rand({6, 6, 6});
  std::vector<IValue> args{
      grad_output0, alpha0, scale0, input_scale0, is_result0, self_or_result0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto alpha1 = 2;
  auto scale1 = 2;
  auto input_scale1 = 2;
  auto is_result1 = false;
  auto self_or_result1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{
      grad_output1, alpha1, scale1, input_scale1, is_result1, self_or_result1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_glu) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int):
        %bias: None = prim::Constant()
        %ret = aten::glu(%self, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  std::vector<IValue> args{self0, dim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto dim1 = 1;
  std::vector<IValue> args2{self1, dim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_hardsigmoid) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::hardsigmoid(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_hardsigmoid_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::hardsigmoid_backward(%grad_output, %self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{grad_output0, self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{grad_output1, self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_hardtanh) {
  const std::string script = R"IR(
    graph(%self: Tensor, %min_val: int, %max_val: int):
        %bias: None = prim::Constant()
        %ret = aten::hardtanh(%self, %min_val, %max_val)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto min_val0 = 2;
  auto max_val0 = 2;
  std::vector<IValue> args{self0, min_val0, max_val0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto min_val1 = 2;
  auto max_val1 = 2;
  std::vector<IValue> args2{self1, min_val1, max_val1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_hardswish) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::hardswish(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_leaky_relu_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %negative_slope: int, %self_is_result: bool):
        %bias: None = prim::Constant()
        %ret = aten::leaky_relu_backward(%grad_output, %self, %negative_slope, %self_is_result)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  auto negative_slope0 = 2;
  auto self_is_result0 = false;
  std::vector<IValue> args{
      grad_output0, self0, negative_slope0, self_is_result0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  auto negative_slope1 = 2;
  auto self_is_result1 = false;
  std::vector<IValue> args2{
      grad_output1, self1, negative_slope1, self_is_result1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_log_sigmoid) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::log_sigmoid(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_softplus) {
  const std::string script = R"IR(
    graph(%self: Tensor, %beta: int, %threshold: int):
        %bias: None = prim::Constant()
        %ret = aten::softplus(%self, %beta, %threshold)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto beta0 = 2;
  auto threshold0 = 2;
  std::vector<IValue> args{self0, beta0, threshold0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto beta1 = 2;
  auto threshold1 = 2;
  std::vector<IValue> args2{self1, beta1, threshold1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_softplus_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %beta: int, %threshold: int):
        %bias: None = prim::Constant()
        %ret = aten::softplus_backward(%grad_output, %self, %beta, %threshold)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  auto beta0 = 2;
  auto threshold0 = 2;
  std::vector<IValue> args{grad_output0, self0, beta0, threshold0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  auto beta1 = 2;
  auto threshold1 = 2;
  std::vector<IValue> args2{grad_output1, self1, beta1, threshold1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_softshrink) {
  const std::string script = R"IR(
    graph(%self: Tensor, %lambd: int):
        %bias: None = prim::Constant()
        %ret = aten::softshrink(%self, %lambd)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto lambd0 = 2;
  std::vector<IValue> args{self0, lambd0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto lambd1 = 2;
  std::vector<IValue> args2{self1, lambd1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_softshrink_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %lambd: int):
        %bias: None = prim::Constant()
        %ret = aten::softshrink_backward(%grad_output, %self, %lambd)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto self0 = at::rand({6, 6, 6});
  auto lambd0 = 2;
  std::vector<IValue> args{grad_output0, self0, lambd0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto self1 = at::rand({22, 22, 22});
  auto lambd1 = 2;
  std::vector<IValue> args2{grad_output1, self1, lambd1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_adaptive_max_pool2d_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %indices: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::adaptive_max_pool2d_backward(%grad_output, %self, %indices)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({2, 2, 2}, at::kFloat);
  auto self0 = at::rand({2, 2, 2}, at::kFloat);
  auto indices0 = at::randint(0, 1, {2, 2, 2}, at::kLong);
  std::vector<IValue> args{grad_output0, self0, indices0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({3, 3, 3}, at::kFloat);
  auto self1 = at::rand({3, 3, 3}, at::kFloat);
  auto indices1 = at::randint(0, 1, {3, 3, 3}, at::kLong);
  std::vector<IValue> args2{grad_output1, self1, indices1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_adaptive_max_pool3d_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %indices: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::adaptive_max_pool3d_backward(%grad_output, %self, %indices)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({2, 2, 2, 2}, at::kFloat);
  auto self0 = at::rand({2, 2, 2, 2}, at::kFloat);
  auto indices0 = at::randint(0, 1, {2, 2, 2, 2}, at::kLong);
  std::vector<IValue> args{grad_output0, self0, indices0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({3, 3, 3, 3}, at::kFloat);
  auto self1 = at::rand({3, 3, 3, 3}, at::kFloat);
  auto indices1 = at::randint(0, 1, {3, 3, 3, 3}, at::kLong);
  std::vector<IValue> args2{grad_output1, self1, indices1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_sigmoid_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %output: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::sigmoid_backward(%grad_output, %output)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto output0 = at::rand({6, 6, 6});
  std::vector<IValue> args{grad_output0, output0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto output1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{grad_output1, output1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_tanh_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %output: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::tanh_backward(%grad_output, %output)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({6, 6, 6});
  auto output0 = at::rand({6, 6, 6});
  std::vector<IValue> args{grad_output0, output0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({22, 22, 22});
  auto output1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{grad_output1, output1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isposinf) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::isposinf(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_isneginf) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::isneginf(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_entr) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_entr(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_ndtri) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_ndtri(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_log_ndtr) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_log_ndtr(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_expm1) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_expm1(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_exp2) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_exp2(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_psi) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_psi(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_digamma) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_digamma(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_gammaln) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_gammaln(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_erf) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_erf(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_erfc) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_erfc(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_erfcx) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_erfcx(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_erfinv) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_erfinv(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_ndtr) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_ndtr(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_xlog1py) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_xlog1py(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_xlogy) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_xlogy(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_zeta) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_zeta(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2, 2, 2}, at::kDouble) + at::ones({2, 2, 2});
  auto other0 = at::rand({2, 2, 2}, at::kDouble) + at::ones({2, 2, 2});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({5, 5, 5}, at::kDouble) + at::ones({5, 5, 5});
  auto other1 = at::rand({5, 5, 5}, at::kDouble) + at::ones({5, 5, 5});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_i0) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_i0(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_i0e) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_i0e(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_i1) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_i1(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_i1e) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_i1e(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_polygamma) {
  const std::string script = R"IR(
    graph(%n: int, %self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_polygamma(%n, %self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto n0 = 1;
  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{n0, self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto n1 = 1;
  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{n1, self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_expit) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_expit(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_sinc) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_sinc(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_round) {
  const std::string script = R"IR(
    graph(%self: Tensor, %decimals: int):
        %bias: None = prim::Constant()
        %ret = aten::special_round(%self, %decimals)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto decimals0 = 1;
  std::vector<IValue> args{self0, decimals0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto decimals1 = 1;
  std::vector<IValue> args2{self1, decimals1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_log1p) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_log1p(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_gammainc) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_gammainc(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_gammaincc) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::special_gammaincc(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_special_multigammaln) {
  const std::string script = R"IR(
    graph(%self: Tensor, %p: int):
        %bias: None = prim::Constant()
        %ret = aten::special_multigammaln(%self, %p)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto p0 = 1;
  std::vector<IValue> args{self0, p0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto p1 = 1;
  std::vector<IValue> args2{self1, p1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_cross) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor, %dim: int):
        %bias: None = prim::Constant()
        %ret = aten::linalg_cross(%self, %other, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 3, 6});
  auto other0 = at::rand({6, 3, 6});
  auto dim0 = 1;
  std::vector<IValue> args{self0, other0, dim0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 3, 22});
  auto other1 = at::rand({22, 3, 22});
  auto dim1 = 1;
  std::vector<IValue> args2{self1, other1, dim1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_det) {
  const std::string script = R"IR(
    graph(%A: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::linalg_det(%A)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto A0 = at::rand({6, 6, 6});
  std::vector<IValue> args{A0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto A1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{A1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_matmul) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::linalg_matmul(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_eigvals) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::linalg_eigvals(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_inv) {
  const std::string script = R"IR(
    graph(%A: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::linalg_inv(%A)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto A0 = at::rand({6, 6, 6});
  std::vector<IValue> args{A0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto A1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{A1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_inverse) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::inverse(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_inner) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::inner(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto other1 = at::rand({22, 22, 22});
  std::vector<IValue> args2{self1, other1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_outer) {
  const std::string script = R"IR(
    graph(%self: Tensor, %vec2: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::outer(%self, %vec2)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({16});
  auto vec20 = at::rand({16});
  std::vector<IValue> args{self0, vec20};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({64});
  auto vec21 = at::rand({64});
  std::vector<IValue> args2{self1, vec21};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_cond) {
  const std::string script = R"IR(
    graph(%self: Tensor, %p: int?):
        %bias: None = prim::Constant()
        %ret = aten::linalg_cond(%self, %p)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto p0 = 2;
  std::vector<IValue> args{self0, p0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto p1 = 2;
  std::vector<IValue> args2{self1, p1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_solve) {
  const std::string script = R"IR(
    graph(%A: Tensor, %B: Tensor, %left: bool):
        %bias: None = prim::Constant()
        %ret = aten::linalg_solve(%A, %B, %left)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto A0 = at::rand({6, 6, 6});
  auto B0 = at::rand({6, 6, 6});
  auto left0 = false;
  std::vector<IValue> args{A0, B0, left0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto A1 = at::rand({22, 22, 22});
  auto B1 = at::rand({22, 22, 22});
  auto left1 = false;
  std::vector<IValue> args2{A1, B1, left1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_tensorinv) {
  const std::string script = R"IR(
    graph(%self: Tensor, %ind: int):
        %bias: None = prim::Constant()
        %ret = aten::linalg_tensorinv(%self, %ind)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6, 6});
  auto ind0 = 2;
  std::vector<IValue> args{self0, ind0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22, 22});
  auto ind1 = 2;
  std::vector<IValue> args2{self1, ind1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_linalg_matrix_power) {
  const std::string script = R"IR(
    graph(%self: Tensor, %n: int):
        %bias: None = prim::Constant()
        %ret = aten::linalg_matrix_power(%self, %n)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto n0 = 1;
  std::vector<IValue> args{self0, n0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto self1 = at::rand({22, 22, 22});
  auto n1 = 1;
  std::vector<IValue> args2{self1, n1};
  testStaticRuntime(
      script,
      args,
      args2,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);
}

TEST(StaticRuntime, autogen_view_as_real) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::view_as_real(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randn({6, 6, 6}, at::kComplexFloat);
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_view_as_complex) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::view_as_complex(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({2, 2});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_real) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::real(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_imag) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::imag(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randn({6, 6, 6}, at::kComplexFloat);
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen__conj) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::_conj(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::randn({6, 6, 6}, at::kComplexFloat);
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_conj) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::conj(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_resolve_conj) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::resolve_conj(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_resolve_neg) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::resolve_neg(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen__neg_view) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::_neg_view(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_diagonal) {
  const std::string script = R"IR(
    graph(%self: Tensor, %offset: int, %dim1: int, %dim2: int):
        %bias: None = prim::Constant()
        %ret = aten::diagonal(%self, %offset, %dim1, %dim2)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto offset0 = 0;
  auto dim10 = 2;
  auto dim20 = 1;
  std::vector<IValue> args{self0, offset0, dim10, dim20};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_linalg_diagonal) {
  const std::string script = R"IR(
    graph(%A: Tensor, %offset: int, %dim1: int, %dim2: int):
        %bias: None = prim::Constant()
        %ret = aten::linalg_diagonal(%A, %offset, %dim1, %dim2)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto A0 = at::rand({6, 6, 6});
  auto offset0 = 0;
  auto dim10 = 2;
  auto dim20 = 1;
  std::vector<IValue> args{A0, offset0, dim10, dim20};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_movedim_int) {
  const std::string script = R"IR(
    graph(%self: Tensor, %source: int, %destination: int):
        %bias: None = prim::Constant()
        %ret = aten::movedim(%self, %source, %destination)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto source0 = 1;
  auto destination0 = 1;
  std::vector<IValue> args{self0, source0, destination0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_moveaxis_int) {
  const std::string script = R"IR(
    graph(%self: Tensor, %source: int, %destination: int):
        %bias: None = prim::Constant()
        %ret = aten::moveaxis(%self, %source, %destination)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto source0 = 1;
  auto destination0 = 1;
  std::vector<IValue> args{self0, source0, destination0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_numpy_T) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::numpy_T(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_matrix_H) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::matrix_H(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({8, 8});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_mT) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::mT(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_mH) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::mH(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_adjoint) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::adjoint(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_ravel) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::ravel(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_t) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::t(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({8, 8});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_unsqueeze) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int):
        %bias: None = prim::Constant()
        %ret = aten::unsqueeze(%self, %dim)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim0 = 1;
  std::vector<IValue> args{self0, dim0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_view_as) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::view_as(%self, %other)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto other0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0, other0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_positive) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::positive(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen__autocast_to_reduced_precision) {
  const std::string script = R"IR(
    graph(%self: Tensor, %cuda_enabled: bool, %cpu_enabled: bool, %cuda_dtype: int, %cpu_dtype: int):
        %bias: None = prim::Constant()
        %ret = aten::_autocast_to_reduced_precision(%self, %cuda_enabled, %cpu_enabled, %cuda_dtype, %cpu_dtype)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto cuda_enabled0 = false;
  auto cpu_enabled0 = false;
  auto cuda_dtype0 = at::ScalarType::Float;
  auto cpu_dtype0 = at::ScalarType::Float;
  std::vector<IValue> args{
      self0, cuda_enabled0, cpu_enabled0, cuda_dtype0, cpu_dtype0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen__autocast_to_full_precision) {
  const std::string script = R"IR(
    graph(%self: Tensor, %cuda_enabled: bool, %cpu_enabled: bool):
        %bias: None = prim::Constant()
        %ret = aten::_autocast_to_full_precision(%self, %cuda_enabled, %cpu_enabled)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto cuda_enabled0 = false;
  auto cpu_enabled0 = false;
  std::vector<IValue> args{self0, cuda_enabled0, cpu_enabled0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_swapaxes) {
  const std::string script = R"IR(
    graph(%self: Tensor, %axis0: int, %axis1: int):
        %bias: None = prim::Constant()
        %ret = aten::swapaxes(%self, %axis0, %axis1)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto axis00 = 1;
  auto axis10 = 1;
  std::vector<IValue> args{self0, axis00, axis10};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_swapdims) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim0: int, %dim1: int):
        %bias: None = prim::Constant()
        %ret = aten::swapdims(%self, %dim0, %dim1)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dim00 = 1;
  auto dim10 = 1;
  std::vector<IValue> args{self0, dim00, dim10};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_unfold) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dimension: int, %size: int, %step: int):
        %bias: None = prim::Constant()
        %ret = aten::unfold(%self, %dimension, %size, %step)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  auto dimension0 = 1;
  auto size0 = 1;
  auto step0 = 1;
  std::vector<IValue> args{self0, dimension0, size0, step0};
  testStaticRuntime(script, args);
}

TEST(StaticRuntime, autogen_alias) {
  const std::string script = R"IR(
    graph(%self: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::alias(%self)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6, 6, 6});
  std::vector<IValue> args{self0};
  testStaticRuntime(script, args);
}
