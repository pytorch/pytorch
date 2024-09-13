#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Operators.h>

using namespace at;

template <class F, F Func, class Output, class... Args>
Output pass_through_wrapper(Args... args) {
  return Func(std::forward<Args>(args)...);
}

TEST(OperatorsTest, TestFunctionDecltype) {
  Tensor a = at::randn({5, 5});
  Tensor b = at::randn({5, 5});
  auto expected = a * b;

  auto result = pass_through_wrapper<
    decltype(&ATEN_FN2(mul, Tensor)), &ATEN_FN2(mul, Tensor),
    Tensor, const Tensor&, const Tensor&>(a, b);
  ASSERT_TRUE(at::allclose(result, a * b));
}

TEST(OperatorsTest, TestMethodOnlyDecltype) {
  Tensor a = at::randn({5, 5});
  Tensor b = at::randn({5, 5});
  auto expected = a * b;

  // NB: add_ overloads are guaranteed to be method-only
  // because that is how the tensor API works.
  auto& result = pass_through_wrapper<
    decltype(&ATEN_FN2(mul_, Tensor)), &ATEN_FN2(mul_, Tensor),
    Tensor&, Tensor&, const Tensor&>(a, b);
  ASSERT_TRUE(at::allclose(result, expected));
}

TEST(OperatorsTest, Test_ATEN_FN) {
  Tensor a = at::rand({5, 5});

  auto result = pass_through_wrapper<
    decltype(&ATEN_FN(sin)), &ATEN_FN(sin),
    Tensor, const Tensor&>(a);
  ASSERT_TRUE(at::allclose(result, a.sin()));
}

TEST(OperatorsTest, TestOutVariantIsFaithful) {
  Tensor a = at::rand({5, 5});
  Tensor b = at::empty({5, 5});

  auto& result = pass_through_wrapper<
    decltype(&ATEN_FN2(sin, out)), &ATEN_FN2(sin, out),
    Tensor&, const Tensor&, Tensor&>(a, b);
  ASSERT_TRUE(at::allclose(result, a.sin()));
}
