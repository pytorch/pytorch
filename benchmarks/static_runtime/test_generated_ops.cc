// @lint-ignore-every CLANGTIDY HOWTOEVEN
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

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

TEST(StaticRuntime, autogen_bitwise_right_shift_Tensor) {
  const std::string script = R"IR(
    graph(%self: Tensor, %other: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::bitwise_right_shift(%self, %other)
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

TEST(StaticRuntime, autogen_nll_loss_backward) {
  const std::string script = R"IR(
    graph(%grad_output: Tensor, %self: Tensor, %target: Tensor, %weight: Tensor?, %reduction: int, %ignore_index: int, %total_weight: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::nll_loss_backward(%grad_output, %self, %target, %weight, %reduction, %ignore_index, %total_weight)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto grad_output0 = at::rand({});
  auto self0 = at::rand({6});
  auto target0 = at::randint(0, 5, {6}, torch::kInt64);
  auto weight0 = at::rand({6});
  auto reduction0 = 1;
  auto ignore_index0 = 1;
  auto total_weight0 = at::rand({});
  std::vector<IValue> args{
      grad_output0,
      self0,
      target0,
      weight0,
      reduction0,
      ignore_index0,
      total_weight0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::rand({});
  auto self1 = at::rand({36});
  auto target1 = at::randint(0, 11, {36}, torch::kInt64);
  auto weight1 = at::rand({36});
  auto reduction1 = 1;
  auto ignore_index1 = 1;
  auto total_weight1 = at::rand({});
  std::vector<IValue> args2{
      grad_output1,
      self1,
      target1,
      weight1,
      reduction1,
      ignore_index1,
      total_weight1};
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

  auto grad_output0 = at::randint(-3, 2, {2, 2, 2});
  auto self0 = at::randint(-3, 2, {2, 2, 2});
  auto indices0 = at::randint(0, 1, {2, 2, 2}, at::kLong);
  std::vector<IValue> args{grad_output0, self0, indices0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::randint(-3, 3, {3, 3, 3});
  auto self1 = at::randint(-3, 2, {3, 3, 3});
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

  auto grad_output0 = at::randint(-3, 2, {2, 2, 2, 2});
  auto self0 = at::randint(-3, 2, {2, 2, 2, 2});
  auto indices0 = at::randint(0, 1, {2, 2, 2, 2}, at::kLong);
  std::vector<IValue> args{grad_output0, self0, indices0};
  testStaticRuntime(
      script,
      args,
      {},
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/true);

  auto grad_output1 = at::randint(-3, 3, {3, 3, 3, 3});
  auto self1 = at::randint(-3, 2, {3, 3, 3, 3});
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
