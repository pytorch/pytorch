#if defined(USE_CUDA)
#include <gtest/gtest.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_simplifier.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

TEST_F(NVFuserTest, FusionAssociativeAndCommutativeReordering_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto c = IrBuilder::create<NamedScalar>("c", DataType::Int);
  auto d = IrBuilder::create<NamedScalar>("d", DataType::Int);
  auto e = IrBuilder::create<NamedScalar>("e", DataType::Int);
  auto f = IrBuilder::create<NamedScalar>("f", DataType::Int);
  auto three = IrBuilder::create<Int>(3);
  auto five = IrBuilder::create<Int>(5);
  auto six = IrBuilder::create<Int>(6);
  auto eight = IrBuilder::create<Int>(8);

  // ((((c * d) + ((e + f) + 3)) + 3) * ((((a + b) + 3) + 5) + c)) * a
  auto val =
      mul(mul(add(add(mul(c, d), add(add(e, f), three)), three),
              add(add(add(add(a, b), three), five), c)),
          a);
  std::vector<ValInfo> variables(6);
  variables[0].variable = a;
  variables[1].variable = b;
  variables[2].variable = c;
  variables[3].variable = d;
  variables[4].variable = e;
  variables[5].variable = f;
  auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});

  // simplify it, expecting to get
  // (a * (((8 + a) + b) + c)) * (((6 + (c * d)) + e) + f)
  auto expect =
      mul(mul(a, add(add(add(eight, a), b), c)),
          add(add(add(six, mul(c, d)), e), f));
  TORCH_CHECK(
      expect->sameAs(simplified) && simplified->sameAs(expect),
      "Expect the simplified expression",
      simplified->toInlineString(),
      " to be the same as ",
      expect->toInlineString());
}

TEST_F(NVFuserTest, FusionEliminateTrivialComputation_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto i = IrBuilder::create<NamedScalar>("i", DataType::Int);
  auto d = IrBuilder::create<NamedScalar>("d", DataType::Double);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Bool);
  auto i0 = IrBuilder::create<Int>(0);
  auto i1 = IrBuilder::create<Int>(1);
  auto i2 = IrBuilder::create<Int>(2);
  auto d0 = IrBuilder::create<Double>(0);
  auto d1 = IrBuilder::create<Double>(1);
  auto d2 = IrBuilder::create<Double>(2);
  auto t = IrBuilder::create<Bool>(true);
  auto f = IrBuilder::create<Bool>(false);

  // 1 * a -> a
  TORCH_CHECK(simplifyExpr(mul(i1, i))->sameAs(i));
  TORCH_CHECK(simplifyExpr(mul(d1, d))->sameAs(d));
  // a * 1 -> a
  TORCH_CHECK(simplifyExpr(mul(i, i1))->sameAs(i));
  TORCH_CHECK(simplifyExpr(mul(d, d1))->sameAs(d));
  // 0 * a -> 0
  TORCH_CHECK(simplifyExpr(mul(i0, i))->sameAs(i0));
  // a * 0 -> 0
  TORCH_CHECK(simplifyExpr(mul(i, i0))->sameAs(i0));

  // 0 + a -> a
  TORCH_CHECK(simplifyExpr(add(i0, i))->sameAs(i));
  TORCH_CHECK(simplifyExpr(add(d0, d))->sameAs(d));
  // a + 0 -> a
  TORCH_CHECK(simplifyExpr(add(i, i0))->sameAs(i));
  TORCH_CHECK(simplifyExpr(add(d, d0))->sameAs(d));

  // true && a -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(t, b))->sameAs(b));
  // a && true -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(b, t))->sameAs(b));
  // false && a -> false
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(f, b))->sameAs(f));
  // a && false -> false
  TORCH_CHECK(simplifyExpr(IrBuilder::andExpr(b, f))->sameAs(f));

  // true || a -> true
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(t, b))->sameAs(t));
  // a || true -> true
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(b, t))->sameAs(t));
  // false || a -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(f, b))->sameAs(b));
  // a || false -> a
  TORCH_CHECK(simplifyExpr(IrBuilder::orExpr(b, f))->sameAs(b));

  // a / 1 -> a
  TORCH_CHECK(simplifyExpr(cpp_div(i, i1))->sameAs(i));
  TORCH_CHECK(simplifyExpr(cpp_div(d, d1))->sameAs(d));
  // 0 / a -> 0
  auto tdimx = NamedScalar::getParallelDim(ParallelType::TIDx);
  TORCH_CHECK(simplifyExpr(cpp_div(i0, tdimx))->sameAs(i0));
  // a % 1 -> 0
  TORCH_CHECK(simplifyExpr(mod(i, i1))->sameAs(i0));

  // Test constant folding
  TORCH_CHECK(simplifyExpr(add(add(i1, i), i1))->sameAs(add(i, i2)));
  TORCH_CHECK(simplifyExpr(add(add(d1, d), d1))->sameAs(add(d, d2)));

  // Test that FlattenedAssocCommOp::sameAs ignores order
  auto x = IrBuilder::create<NamedScalar>("x", DataType::Int);
  auto y = IrBuilder::create<NamedScalar>("y", DataType::Int);
  TORCH_CHECK(simplifyExpr(sub(mul(x, y), mul(y, x)))->isZeroInt());
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
