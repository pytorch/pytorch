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

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
