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

TEST(BufLiveRange, SingleRangeLine) {
  VarHandle i("i", kInt), j("j", kInt);
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32, 32}, kFloat);

  // Construct Stmt:
  // {
  //   for (int i = 0; i < 32; i++) {
  //     a[i] = 0;
  //     for (int j = 0; j < 32; j++) {
  //       a[i] = (a[i]) + (b[i, j]);
  //     }
  //   }
  // }

  StorePtr aInit = Store::make(a, {i}, 0);
  ExprHandle reduce = a.load({i}) + b.load({i, j});
  StorePtr aReduce = Store::make(a, {i}, reduce);
  StmtPtr loop =
      For::make(i, 0, 32, Block::make({aInit, For::make(j, 0, 32, aReduce)}));

  StmtPtr stmt = Block::make({loop});

  auto range = BufLiveRange::liveRange(stmt, a.node());
  ASSERT_TRUE(std::get<0>(range) == 0);
  ASSERT_TRUE(std::get<1>(range) == 0);
}

TEST(BufLiveRange, MulRangeLine) {
  VarHandle i("i", kInt);
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32}, kFloat);

  // Construct Stmt:
  // {
  //   for (int i = 0; i < 32; i++) {
  //     if (i<10 ? 1 : 0) {
  //       a[i] = i + i;
  //       b[i] = i * i;
  //     }
  //   }
  //   for (int i = 0; i < 32; i++) {
  //     if (i>10 ? 1 : 0) {
  //       a[i] = i * i;
  //       b[i] = i + i;
  //     }
  //   }
  // }

  StorePtr aStore_1 = Store::make(a, {i}, i + i);
  StorePtr bStore_1 = Store::make(b, {i}, i * i);
  StmtPtr loop_1 = For::make(
      i, 0, 32, Cond::make(i < 10, Block::make({aStore_1, bStore_1}), NULL));

  StorePtr aStore_2 = Store::make(a, {i}, i * i);
  StorePtr bStore_2 = Store::make(b, {i}, i + i);
  StmtPtr loop_2 = For::make(
      i, 0, 32, Cond::make(i > 10, Block::make({aStore_2, bStore_2}), NULL));

  StmtPtr stmt = Block::make({loop_1, loop_2});

  auto range_a = BufLiveRange::liveRange(stmt, a.node());
  ASSERT_TRUE(std::get<0>(range_a) == 0);
  ASSERT_TRUE(std::get<1>(range_a) == 1);

  auto range_b = BufLiveRange::liveRange(stmt, b.node());
  ASSERT_TRUE(std::get<0>(range_b) == 0);
  ASSERT_TRUE(std::get<1>(range_b) == 1);
}

TEST(MemPlanning, SameBufSizeMemReuse) {
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
        return ET.load(m, n) * ET.load(m, n);
      });
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  // Constructed stmt:
  // Intermediate buffers and their liveness ranges: gemm [0, 1], relu [1, 2],
  // add [2, 3] Buffer 'gemm' and 'add' are the same size; we'll reuse 'gemm'
  // for 'add'.
  //{
  //  for (int M = 0; M < 1024; M++) {
  //    for (int N = 0; N < 1024; N++) {
  //      gemm[M, N] = float(0);
  //      for (int K = 0; K < 2048; K++) {
  //        gemm[M, N] = ReduceOp((gemm[M, N]) + (A[M, K]) * (B[K, N]),
  //        reduce_args={K});
  //      }
  //    }
  //  }
  //  for (int M_1 = 0; M_1 < 1024; M_1++) {
  //    for (int N_1 = 0; N_1 < 1024; N_1++) {
  //      relu[M_1, N_1] = (gemm[M_1, N_1])<float(0) ? float(0) : (gemm[M_1,
  //      N_1]);
  //    }
  //  }
  //  for (int M_2 = 0; M_2 < 1024; M_2++) {
  //    for (int N_2 = 0; N_2 < 1024; N_2++) {
  //      add[M_2, N_2] = (relu[M_2, N_2]) + (relu[M_2, N_2]);
  //    }
  //  }
  //  for (int M_3 = 0; M_3 < 1024; M_3++) {
  //    for (int N_3 = 0; N_3 < 1024; N_3++) {
  //      mul[M_3, N_3] = (add[M_3, N_3]) * (add[M_3, N_3]);
  //    }
  //  }
  //}

  SimpleIREvaluator cg(stmt, {AP, BP, FT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

#ifdef TORCH_ENABLE_LLVM
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}

TEST(MemPlanning, SameBufSizeMultiMemReuses) {
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
        return ET.load(m, n) * ET.load(m, n);
      });
  Tensor GT = Compute(
      "sub",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return FT.load(m, n) - ET.load(m, n);
      });

  auto stmt =
      Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt(), GT.stmt()});

  // Constructed stmt:
  // Intermediate buffers and their liveness ranges: gemm [0, 1], relu [1, 2],
  // add [2, 3], mul [3, 4] Buffer 'gemm', 'relu, ''add' and 'mul' are the same
  // size; we'll reuse 'gemm' for 'add', and reuse 'relu' for 'mul'
  //{
  //  for (int M = 0; M < 1024; M++) {
  //    for (int N = 0; N < 1024; N++) {
  //      gemm[M, N] = float(0);
  //      for (int K = 0; K < 2048; K++) {
  //        gemm[M, N] = ReduceOp((gemm[M, N]) + (A[M, K]) * (B[K, N]),
  //        reduce_args={K});
  //      }
  //    }
  //  }
  //  for (int M_1 = 0; M_1 < 1024; M_1++) {
  //    for (int N_1 = 0; N_1 < 1024; N_1++) {
  //      relu[M_1, N_1] = (gemm[M_1, N_1])<float(0) ? float(0) : (gemm[M_1,
  //      N_1]);
  //    }
  //  }
  //  for (int M_2 = 0; M_2 < 1024; M_2++) {
  //    for (int N_2 = 0; N_2 < 1024; N_2++) {
  //      add[M_2, N_2] = (relu[M_2, N_2]) + (relu[M_2, N_2]);
  //    }
  //  }
  //  for (int M_3 = 0; M_3 < 1024; M_3++) {
  //    for (int N_3 = 0; N_3 < 1024; N_3++) {
  //      mul[M_3, N_3] = (add[M_3, N_3]) * (add[M_3, N_3]);
  //    }
  //  }
  //  for (int M_4 = 0; M_4 < 1024; M_4++) {
  //    for (int N_4 = 0; N_4 < 1024; N_4++) {
  //      sub[M_4, N_4] = (mul[M_4, N_4]) - (add[M_4, N_4]);
  //    }
  //  }
  //}

  SimpleIREvaluator cg(stmt, {AP, BP, GT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Alias(mul,relu);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

#ifdef TORCH_ENABLE_LLVM
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Alias(mul,relu);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}

TEST(MemPlanning, SameBufSizeMultiMemReusesOfOneBuf) {
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
        return ET.load(m, n) * ET.load(m, n);
      });
  Tensor GT = Compute(
      "sub",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return FT.load(m, n) - 1;
      });
  Tensor HT = Compute(
      "div",
      {{M, "M"}, {N, "N"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return GT.load(m, n) / 2;
      });

  auto stmt = Block::make(
      {CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt(), GT.stmt(), HT.stmt()});

  // Constructed stmt:
  // Intermediate buffers and their liveness ranges: gemm [0, 1], relu [1, 2],
  // add [2, 3], mul [3, 4], sub [4, 5] Buffer 'gemm', 'relu, ''add', 'mul' and
  // 'sub' are the same size; we'll reuse 'gemm' for 'add', reuse 'relu' for
  // 'mul', and reuse 'gemm' for 'sub'.
  //{
  //  for (int M = 0; M < 1024; M++) {
  //    for (int N = 0; N < 1024; N++) {
  //      gemm[M, N] = float(0);
  //      for (int K = 0; K < 2048; K++) {
  //        gemm[M, N] = ReduceOp((gemm[M, N]) + (A[M, K]) * (B[K, N]),
  //        reduce_args={K});
  //      }
  //    }
  //  }
  //  for (int M_1 = 0; M_1 < 1024; M_1++) {
  //    for (int N_1 = 0; N_1 < 1024; N_1++) {
  //      relu[M_1, N_1] = (gemm[M_1, N_1])<float(0) ? float(0) : (gemm[M_1,
  //      N_1]);
  //    }
  //  }
  //  for (int M_2 = 0; M_2 < 1024; M_2++) {
  //    for (int N_2 = 0; N_2 < 1024; N_2++) {
  //      add[M_2, N_2] = (relu[M_2, N_2]) + (relu[M_2, N_2]);
  //    }
  //  }
  //  for (int M_3 = 0; M_3 < 1024; M_3++) {
  //    for (int N_3 = 0; N_3 < 1024; N_3++) {
  //      mul[M_3, N_3] = (add[M_3, N_3]) * (add[M_3, N_3]);
  //    }
  //  }
  //  for (int M_4 = 0; M_4 < 1024; M_4++) {
  //    for (int N_4 = 0; N_4 < 1024; N_4++) {
  //      sub[M_4, N_4] = (mul[M_4, N_4]) - float(1);
  //    }
  //  }
  //  for (int M_5 = 0; M_5 < 1024; M_5++) {
  //    for (int N_5 = 0; N_5 < 1024; N_5++) {
  //      div[M_5, N_5] = (sub[M_5, N_5]) / float(2);
  //    }
  //  }
  //}

  SimpleIREvaluator cg(stmt, {AP, BP, HT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Alias(mul,relu);
# CHECK: Alias(sub,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

#ifdef TORCH_ENABLE_LLVM
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Alias(mul,relu);
# CHECK: Alias(sub,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}

TEST(MemPlanning, SmallerBufSizeNonMemReuse) {
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
        return ET.load(fm, fn) * ET.load(fm, fn);
      });
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  // Constructed stmt:
  // Intermediate buffers and their liveness ranges: gemm [0, 1], relu [1, 2],
  // add [2, 3] We do not reuse buffer 'gemm' for 'add' because the size of
  // buffer 'gemm' is smaller.
  //{
  //  for (int M = 0; M < 1024; M++) {
  //    for (int N = 0; N < 1024; N++) {
  //      gemm[M, N] = float(0);
  //      for (int K = 0; K < 2048; K++) {
  //        gemm[M, N] = ReduceOp((gemm[M, N]) + (A[M, K]) * (B[K, N]),
  //        reduce_args={K});
  //      }
  //    }
  //  }
  //  for (int M_1 = 0; M_1 < 1024; M_1++) {
  //    for (int N_1 = 0; N_1 < 1024; N_1++) {
  //      relu[M_1, N_1] = (gemm[M_1, N_1])<float(0) ? float(0) : (gemm[M_1,
  //      N_1]);
  //    }
  //  }
  //  for (int EM = 0; EM < 2048; EM++) {
  //    for (int EN = 0; EN < 2048; EN++) {
  //      add[EM, EN] = (relu[EM / 2, EN / 2]) + (relu[EM / 2, EN / 2]);
  //    }
  //  }
  //  for (int FM = 0; FM < 2048; FM++) {
  //    for (int FN = 0; FN < 2048; FN++) {
  //      mul[FM, FN] = (add[FM, FN]) * (add[FM, FN]);
  //    }
  //  }
  //}
  //

  SimpleIREvaluator cg(stmt, {AP, BP, FT});

  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK-NOT: Alias(add,gemm);
# CHECK: Allocate(add); // dtype=float, dims=[2048, 2048]
# CHECK: Free(add);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

#ifdef TORCH_ENABLE_LLVM
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK-NOT: Alias(add,gemm);
# CHECK: Allocate(add); // dtype=float, dims=[2048, 2048]
# CHECK: Free(add);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}

} // namespace jit
} // namespace torch
