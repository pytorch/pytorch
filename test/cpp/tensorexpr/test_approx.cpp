#ifdef TORCH_ENABLE_LLVM

#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>
#include <cstring>

namespace te = torch::jit::tensorexpr;

static void vectorize(te::LoopNest* ln, te::Tensor* target, int width) {
  auto loops = ln->getLoopStmtsFor(target);
  te::For *outer, *inner, *tail;
  ln->splitWithTail(loops[0], width, &outer, &inner, &tail);
  ln->vectorize(inner);
}

TEST(Approx, log_vml) {
  te::KernelScope ks;
  te::VarHandle N("N", te::kInt);
  te::Placeholder A("A", te::kFloat, {N});
  te::Tensor* B = te::Compute(
      "B", {N}, [&](const te::VarHandle& i) { return log_vml(A.load(i)); });

  te::LoopNest ln({B});
  ln.prepareForCodegen();
  vectorize(&ln, B, 8);
  te::Stmt* s = ln.root_stmt();
  s = te::IRSimplifier::simplify(s);
  te::LLVMCodeGen cg(s, {A, B, N});

  auto test = [&](const at::Tensor& A_t) {
    at::Tensor B_ref = at::log(A_t);
    at::Tensor B_t = at::empty_like(A_t);
    cg.call({A_t.data_ptr<float>(), B_t.data_ptr<float>(), A_t.numel()});
    // Results should be bit-identical.
    ASSERT_TRUE(
        memcmp(
            B_ref.data_ptr<float>(), B_t.data_ptr<float>(), B_ref.nbytes()) ==
        0);
  };

  // Generate every single-precision FP value in [1.0, 2.0).
  auto eps = std::numeric_limits<float>::epsilon();
  at::Tensor A_t = torch::arange(1.0f, 2.0f, eps);
  ASSERT_EQ(A_t.numel(), 1 << 23);

  test(A_t);

  test(A_t * 2.0f);
  test(A_t * 0.5f);

  test(A_t * 4.0f);
  test(A_t * 0.25f);

  test(A_t * powf(2.0f, 16));
  test(A_t * powf(2.0f, -16));

  test(A_t * powf(2.0f, 126));
  test(A_t * powf(2.0f, -126));

  test(torch::full({32}, INFINITY));
  test(torch::full({32}, NAN));

  auto min = std::numeric_limits<float>::min();
  auto denorm_min = std::numeric_limits<float>::denorm_min();

  // Denormals aren't bit precise, because sleef isn't bit-precise either.
  A_t = torch::arange(0.0f, min, denorm_min);
  ASSERT_EQ(A_t.numel(), 1 << 23);
  auto B_ref = at::log(A_t);
  auto B_t = at::empty_like(B_ref);
  cg.call({A_t.data_ptr<float>(), B_t.data_ptr<float>(), A_t.numel()});
  ASSERT_TRUE(torch::allclose(B_t, B_ref));
}

#endif // TORCH_ENABLE_LLVM
