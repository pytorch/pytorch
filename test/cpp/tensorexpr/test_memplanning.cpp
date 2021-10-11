#include <gtest/gtest.h>
#include <test/cpp/tensorexpr/test_base.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

extern void checkIR(StmtPtr s, const std::string& pattern);

TEST(MemPlanning, MemReuse1onEval) {
  int M = 1024;
  int N = 1024;
  int K = 2048;

  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  Tensor CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  Tensor DT = Compute(
      "relu",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });
  Tensor ET = Compute(
      "add",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return DT.load(m, n) + DT.load(m, n);
      });
  Tensor FT = Compute(
      "mul",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return ET.load(m, n) + ET.load(m, n);
      });
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});
  SimpleIREvaluator cg(stmt, {AP, BP, FT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Map(add,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
}

TEST(MemPlanning, MemReuse2onEval) {
  int M = 1024;
  int N = 1024;
  int K = 2048;

  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  Tensor CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  Tensor DT = Compute(
      "relu",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });
  Tensor ET = Compute(
      "add",
      {{M * 2, "EM"}, {N * 2, "EN"}},
      [&](const ExprHandle& em, const ExprHandle& en) {
        return DT.load(em / 2, en / 2) + DT.load(em / 2, en / 2);
      });
  Tensor FT = Compute(
      "mul",
      {{M * 2, "FM"}, {N * 2, "FN"}},
      [&](const ExprHandle& fm, const ExprHandle& fn) {
        return ET.load(fm, fn) + ET.load(fm, fn);
      });
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});
  SimpleIREvaluator cg(stmt, {AP, BP, FT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK-NOT: Map(add,gemm);
# CHECK: Allocate(add); // dtype=float, dims=[2048, 2048]
# CHECK: Free(add);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
}

#ifdef TORCH_ENABLE_LLVM
TEST(MemPlanning, MemReuse1onLLVM) {
  int M = 1024;
  int N = 1024;
  int K = 2048;

  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  Tensor CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  Tensor DT = Compute(
      "relu",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });
  Tensor ET = Compute(
      "add",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return DT.load(m, n) + DT.load(m, n);
      });
  Tensor FT = Compute(
      "mul",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return ET.load(m, n) + ET.load(m, n);
      });
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  LoopNest loop(stmt, {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg(loop.root_stmt(), {AP, BP, FT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Map(add,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
}

TEST(MemPlanning, MemReuse2onLLVM) {
  int M = 1024;
  int N = 1024;
  int K = 2048;

  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  Tensor CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  Tensor DT = Compute(
      "relu",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });
  Tensor ET = Compute(
      "add",
      {{M * 2, "EM"}, {N * 2, "EN"}},
      [&](const ExprHandle& em, const ExprHandle& en) {
        return DT.load(em / 2, en / 2) + DT.load(em / 2, en / 2);
      });
  Tensor FT = Compute(
      "mul",
      {{M * 2, "FM"}, {N * 2, "FN"}},
      [&](const ExprHandle& fm, const ExprHandle& fn) {
        return ET.load(fm, fn) + ET.load(fm, fn);
      });
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  LoopNest loop(stmt, {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg(loop.root_stmt(), {AP, BP, FT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK-NOT: Map(add,gemm);
# CHECK: Allocate(add); // dtype=float, dims=[2048, 2048]
# CHECK: Free(add);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
}

#endif

} // namespace jit
} // namespace torch
