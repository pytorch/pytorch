#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/special.h>

#include <test/cpp/api/support.h>

// Simple test that verifies the special namespace is registered properly
//   properly in C++
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SpecialTest, special) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    auto t = torch::randn(128, torch::kDouble);
    torch::special::gammaln(t);
}

template <typename Op, typename OpOut>
void test_out_variant(Op op, OpOut op_out, const char* op_name) {
  // TODO: Take range (low, high) for generating values of `t`.
  //       For now we use `torch::rand` which generates values
  //       acceptable for all the functions currently supported
  const auto t = torch::rand(128, torch::kDouble);
  auto out = torch::empty_like(t);

  auto expected = op(t);
  op_out(out, t);
  ASSERT_TRUE(torch::allclose(out, expected))
      << "Op: " << op_name << " failed unary out variants test";
}

// Simple test to verify the out variant API exists and is correct
TEST(SpecialTest, special_unary_out_variants) {
  test_out_variant(
      torch::special::gammaln, torch::special::gammaln_out, "gammaln");
  test_out_variant(torch::special::entr, torch::special::entr_out, "entr");
  test_out_variant(torch::special::erf, torch::special::erf_out, "erf");
  test_out_variant(torch::special::erfc, torch::special::erfc_out, "erfc");
  test_out_variant(
      torch::special::erfinv, torch::special::erfinv_out, "erfinv");
  test_out_variant(torch::special::logit, torch::special::logit_out, "logit");
  test_out_variant(torch::special::expit, torch::special::expit_out, "expit");
  test_out_variant(torch::special::exp2, torch::special::exp2_out, "exp2");
  test_out_variant(torch::special::expm1, torch::special::expm1_out, "expm1");
  test_out_variant(torch::special::i0e, torch::special::i0e_out, "i0e");
  test_out_variant(torch::special::i1, torch::special::i1_out, "i1");
  test_out_variant(torch::special::i1e, torch::special::i1e_out, "i1e");
}
