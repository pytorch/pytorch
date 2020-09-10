#include <test/cpp/tensorexpr/test_base.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/function.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

void testExprSimple01() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{16, "X"}, {5, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  For* x_outer;
  For* x_inner;
  For* x_tail;
  std::vector<For*> loops = l.getLoopStmtsFor(tensor);
  l.splitWithTail(loops[0], 2, &x_outer, &x_inner, &x_tail);

  For* x_2;
  For* x_1;
  For* x_tail_2;
  l.splitWithTail(x_outer, 2, &x_2, &x_1, &x_tail_2);
}

void testExprLower01() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{16, "x"}, {5, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 20);
  ASSERT_LT(oss.str().size(), 200);
}

void testExprSimple02() {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor* tensor = Compute("f", {{26, "x"}, {5, "y"}}, func);
  LoopNest l({tensor});
  For* x_outer;
  For* x_inner;
  For* x_tail;
  std::vector<For*> loops = l.getLoopStmtsFor(tensor);
  l.splitWithTail(loops[0], 4, &x_outer, &x_inner, &x_tail);

  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    VarHandle x_outer("x_outer", kInt);
    VarHandle x_inner("x_inner", kInt);
    VarHandle y("y", kInt);
    VarHandle x_tail("x_tail", kInt);
    BufHandle f("f", {26, 5}, kFloat);
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(26) - 0) / 4;
    For* stmt1 = For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y), 1))));
    ExprHandle x_2 = x_tail + x_outer_end * 4;
    For* stmt2 = For::make(
        x_tail,
        0,
        (ExprHandle(26) - 0) % 4,
        For::make(y, 0, 5, Store::make(f, {x_2, y}, func(x_2, y), 1)));
    Stmt* stmt = Block::make({stmt1, stmt2});

    std::ostringstream oss_ref;
    oss_ref << *stmt;
    ASSERT_EQ(oss.str(), oss_ref.str());
  }

  {
    PaddedBuffer<float> f_v(26, 5, "f_v");
    PaddedBuffer<float> f_ref(26, 5, "f_res");

    stmt = FlattenIndexes(stmt);
    SimpleIREvaluator ir_eval(stmt, tensor);
    ir_eval(f_v);

    for (int x = 0; x < 26; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

void testExprSplitWithTail() {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x) {
    return ExprHandle(1.0f) + cast<float>(x);
  };
  Tensor* tensor = Compute("f", {{199, "x"}}, func);
  LoopNest l({tensor});
  For* x_outer;
  For* x_inner;
  For* x_tail;
  std::vector<For*> loops = l.getLoopStmtsFor(tensor);
  l.splitWithTail(loops[0], 17, &x_outer, &x_inner, &x_tail);

  For* a;
  For* b;
  For* c;
  l.splitWithTail(x_outer, 7, &a, &b, &c);

  Stmt* stmt = l.root_stmt();
  Stmt* simplified = IRSimplifier::simplify(stmt);
  Block* body = dynamic_cast<Block*>(simplified);
  ASSERT_EQ(body->nstmts(), 3);
  auto biter = body->begin();

  // Verify that the split loops are ordered correctly.
  For* loop = dynamic_cast<For*>(*biter);
  ++biter;
  ASSERT_NE(loop, nullptr);
  const IntImm* bound = dynamic_cast<const IntImm*>(loop->stop());
  ASSERT_NE(bound, nullptr);
  ASSERT_EQ(bound->value(), 7);

  loop = dynamic_cast<For*>(*biter);
  ++biter;
  ASSERT_NE(loop, nullptr);
  bound = dynamic_cast<const IntImm*>(loop->stop());
  ASSERT_NE(bound, nullptr);
  ASSERT_EQ(bound->value(), 4);

  loop = dynamic_cast<For*>(*biter);
  ASSERT_NE(loop, nullptr);
  bound = dynamic_cast<const IntImm*>(loop->stop());
  ASSERT_NE(bound, nullptr);
  ASSERT_EQ(bound->value(), 12);
}

void testExprSplitWithTailNone() {
  KernelScope kernel_scope;
  auto func = [](const ExprHandle& x, const ExprHandle& y) {
    return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
  };
  Tensor* tensor = Compute("f", {{24, "x"}, {5, "y"}}, func);
  LoopNest l({tensor});
  For* x_outer;
  For* x_inner;
  For* x_tail;
  std::vector<For*> loops = l.getLoopStmtsFor(tensor);
  l.splitWithTail(loops[0], 4, &x_outer, &x_inner, &x_tail);

  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 200);
  ASSERT_LT(oss.str().size(), 600);

  {
    // Compare to a reference loop structure structure.
    VarHandle x_outer("x_outer", kInt);
    VarHandle x_inner("x_inner", kInt);
    VarHandle y("y", kInt);
    VarHandle x_tail("x_tail", kInt);
    BufHandle f("f", {24, 5}, kFloat);
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(24) - 0) / 4;
    Stmt* stmt = new Block({For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y), 1))))});

    std::ostringstream oss_ref;
    oss_ref << *stmt;
    ASSERT_EQ(oss.str(), oss_ref.str());
  }

  {
    PaddedBuffer<float> f_v(24, 5, "f_v");
    PaddedBuffer<float> f_ref(24, 5, "f_res");

    SimpleIREvaluator ir_eval(stmt, tensor);
    ir_eval(f_v);

    for (int x = 0; x < 24; x++) {
      for (int y = 0; y < 5; y++) {
        f_ref(x, y) = 1 + x * x + y * y;
      }
    }

    ExpectAllNear(f_v, f_ref, 1e-5);
  }
}

void testExprSplitWithMask01() {
  KernelScope kernel_scope;
  const int M = 26;
  const int N = 5;
  Buffer a_buf("a", kFloat, {M, N});
  Buffer b_buf("b", kFloat, {M, N});
  Tensor* tensor = Compute(
      "f", {{M, "m"}, {N, "n"}}, [&](const ExprHandle& m, const ExprHandle& n) {
        return a_buf(m, n) + b_buf(m, n) + 1.0f;
      });
  For* n_outer;
  For* n_inner;

  LoopNest l({tensor});
  std::vector<For*> loops = l.getLoopStmtsFor(tensor);
  l.splitWithMask(loops[1], 4, &n_outer, &n_inner);

  Stmt* stmt = l.root_stmt();

  PaddedBuffer<float> a_v(M, N, "a");
  PaddedBuffer<float> b_v(M, N, "b");
  PaddedBuffer<float> c_v(M, N, "c");
  PaddedBuffer<float> c_ref(M, N, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 2 * m;
      b_v(m, n) = 3 * n;
      c_ref(m, n) = a_v(m, n) + b_v(m, n) + 1.0f;
    }
  }

  SimpleIREvaluator(stmt, a_buf, b_buf, tensor)(a_v, b_v, c_v);

  ExpectAllNear(c_v, c_ref, 1e-5);
}

void testSplitWithTailWithLoopOptions() {
  KernelScope kernel_scope;
  const int M = 21;
  Buffer a_buf("a", kFloat, {M});
  Buffer b_buf("b", kFloat, {M});
  Tensor* tensor = Compute("f", {{M, "m"}}, [&](const ExprHandle& m) {
    return a_buf(m) + b_buf(m) + 1.0f;
  });
  For *outer, *inner, *tail;

  LoopNest l({tensor});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  ASSERT_GT(loops.size(), 0);
  l.setGPUBlockIndex(loops[0], LoopOptions::IDX_Y);
  l.splitWithTail(loops[0], 4, &outer, &inner, &tail);
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(tail, nullptr);

  // Outer loop carries loop axis bindings.
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // Inner loop has none.
  ASSERT_TRUE(inner->loop_options().isDefault());

  // Tail loop has none.
  ASSERT_TRUE(tail->loop_options().isDefault());
}

void testSplitWithMaskWithLoopOptions() {
  KernelScope kernel_scope;
  const int M = 21;
  Buffer a_buf("a", kFloat, {M});
  Buffer b_buf("b", kFloat, {M});
  Tensor* tensor = Compute("f", {{M, "m"}}, [&](const ExprHandle& m) {
    return a_buf(m) + b_buf(m) + 1.0f;
  });
  For *outer, *inner;

  LoopNest l({tensor});
  auto loops = NodeFinder<For>::find(l.root_stmt());
  l.setGPUBlockIndex(loops[0], LoopOptions::IDX_Y);
  l.splitWithMask(loops[0], 4, &outer, &inner);

  // Outer loop carries loop axis bindings.
  ASSERT_TRUE(outer->loop_options().is_gpu_block_index());
  ASSERT_EQ(outer->loop_options().gpu_block_index(), LoopOptions::IDX_Y);

  // Inner loop has none.
  ASSERT_TRUE(inner->loop_options().isDefault());
}

void testScheduleBroadcastAddBuffer() {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat, {M, N});
  Buffer b_buf("b", kFloat, {N, K});
  Tensor* c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf(m, n) + b_buf(n, k);
      });
  LoopNest l({c});
  Stmt* stmt = l.root_stmt();

  PaddedBuffer<float> a_v(M, N, "a_v");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      a_v(m, n) = 7 * m * n;
    }
  }
  a_v.Backup();

  PaddedBuffer<float> b_v(N, K, "b_v");
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b_v(n, k) = 11 * n * k;
    }
  }
  b_v.Backup();

  PaddedBuffer<float> c_v(M, N, K, "c_buf");
  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c);
  ir_eval(a_v, b_v, c_v);

  a_v.CheckBackup();
  b_v.CheckBackup();
  PaddedBuffer<float> c_ref(M, N, K, "c_ref");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        c_ref(m, n, k) = 7 * m * n + 11 * n * k;
      }
    }
  }
  ExpectAllNear(c_v, c_ref, 1e-5);
}

void testScheduleFunctionCall01() {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat, {M, N});
  Buffer b_buf("b", kFloat, {N, K});
  Tensor* c = Compute(
      "broadcast_add",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf(m, n) + b_buf(n, k);
      });
  Tensor* d = Compute(
      "d",
      {{M, "m"}, {N, "n"}, {K, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c->call(m, n, k) + 1;
      });

  LoopNest l({d});
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  std::ostringstream oss;
  oss << *stmt;
  ASSERT_GT(oss.str().size(), 100);

  PaddedBuffer<float> a_v(M, N);
  PaddedBuffer<float> b_v(N, K);
  PaddedBuffer<float> c_v(M, N, K);
  PaddedBuffer<float> d_v(M, N, K);
  PaddedBuffer<float> d_ref(M, N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a_v(i, j) = i * i;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      b_v(i, j) = j * j;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        d_ref(i, j, k) = a_v(i, j) + b_v(j, k) + 1;
      }
    }
  }

  SimpleIREvaluator eval(stmt, a_buf, b_buf, d);
  eval(a_v, b_v, d_v);

  ExpectAllNear(d_v, d_ref, 1e-5);
}

static std::string remove_space(const std::string& str) {
  std::string str_new = str;
  str_new.erase(
      remove_if(str_new.begin(), str_new.end(), isspace), str_new.end());
  return str_new;
}

void InlineFunc01Helper(const std::vector<std::string>& inline_order) {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat, {M, N});
  Buffer b_buf("b", kFloat, {N, K});
  Buffer c_buf("c", kFloat, {M, N});
  Buffer d_buf("d", kFloat, {M, K});

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf(m, n) * b_buf(n, k);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf(m, n) * d_buf(m, k) + x->call(m, n, k);
      });
  Tensor* z = Compute(
      "z",
      {{M, "m3"}, {N, "n3"}, {K, "k3"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->call(m, n, k) + y->call(m, n, k);
      });

  LoopNest l({z});
  for (const std::string& order : inline_order) {
    if (order == "x") {
      l.computeInline(l.getLoopBodyFor(x));
    } else if (order == "y") {
      l.computeInline(l.getLoopBodyFor(y));
    } else {
      throw std::runtime_error("Invalid order: " + order);
    }
  }
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();

  std::ostringstream oss;
  oss << *stmt;
  std::string str1 = remove_space(oss.str());

  {
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        a_v(i, j) = j * j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    PaddedBuffer<float> z_v(M, N, K);
    PaddedBuffer<float> z_ref(M, N, K);
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    SimpleIREvaluator eval(stmt, a_buf, b_buf, c_buf, d_buf, z);
    eval(a_v, b_v, c_v, d_v, z_v);
    ExpectAllNear(z_v, z_ref, 1e-5);
  }

  if (inline_order.size() == 2) {
    Tensor* z2 = Compute(
        "z",
        {{M, "m3"}, {N, "n3"}, {K, "k3"}},
        [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
          return a_buf(m, n) * b_buf(n, k) +
              (c_buf(m, n) * d_buf(m, k) + a_buf(m, n) * b_buf(n, k));
        });
    LoopNest l2({z2});
    l2.prepareForCodegen();
    Stmt* stmt2 = l2.root_stmt();

    std::ostringstream oss2;
    oss2 << *stmt2;
    std::string str2 = remove_space(oss2.str());

    ASSERT_EQ(str1, str2);
    ASSERT_GT(str1.size(), 100);
  }
}

void testScheduleInlineFunc01() {
  InlineFunc01Helper({"x", "y"});
  InlineFunc01Helper({"y", "x"});
  InlineFunc01Helper({"x"});
  InlineFunc01Helper({"y"});
  InlineFunc01Helper({});
}

void testScheduleFuserStyle() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));

  Tensor* b = Compute(
      "f", {{kTotalSize, "i"}}, [&](const std::vector<VarHandle>& axes) {
        return a_buf(axes[0]) + 11.0f;
      });

  Tensor* c = Compute(
      "g", {{kTotalSize, "i"}}, [&](const std::vector<VarHandle>& axes) {
        return b->call(axes[0]) + 1.0f;
      });

  LoopNest l({b, c});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

  std::vector<float> a_data(kTotalSize, 7.0f);
  std::vector<float> b_data(kTotalSize, 0.0f);
  std::vector<float> c_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, a_buf, b, c)(a_data, b_data, c_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(b_data[i], 18.0f);
    ASSERT_EQ(c_data[i], 19.0f);
  }
}

void testScheduleFuserThreeArg() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Buffer b(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Buffer c(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Buffer d(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  Tensor* e = Compute("e", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return a(i) + b(i);
  });
  Tensor* f = Compute("f", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return (*e)(i) + c(i);
  });
  Tensor* g = Compute("g", {{kTotalSize, "i"}}, [&](const VarHandle& i) {
    return (*f)(i) + d(i);
  });

  LoopNest l({g});
  l.computeInline(l.getLoopBodyFor(e));
  l.computeInline(l.getLoopBodyFor(f));
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

  std::vector<float> a_data(kTotalSize, 1.0f);
  std::vector<float> b_data(kTotalSize, 2.0f);
  std::vector<float> c_data(kTotalSize, 3.0f);
  std::vector<float> d_data(kTotalSize, 4.0f);
  std::vector<float> g_data(kTotalSize, 0.0f);
  SimpleIREvaluator(s, a, b, c, d, g)(a_data, b_data, c_data, d_data, g_data);

  for (int i = 0; i < kTotalSize; i++) {
    ASSERT_EQ(g_data[i], 10.0f);
  }
}

void testScheduleDynamicShape2D() {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    Buffer a(BufHandle("a", {m, n}, kFloat));
    Buffer b(BufHandle("b", {m, n}, kFloat));
    Tensor* c = Compute(
        "c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a(i, j) + b(i, j);
        });
    LoopNest l({c});
    Stmt* s = l.root_stmt();
    SimpleIREvaluator cg(s, {a, b, c, m, n});
    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    cg.call({aData, bData, cData, M, N});
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
  };
  testWithSize(1, 8);
  testWithSize(16, 32);
  testWithSize(37, 11);
}

void testLoopNestComputeAt_1() {
  // Verify that compute_at works on the following example:
  //
  // for (int i_a = 0; i_a < N; i_a++) {
  //   A[i_a] = i_a * i_a
  // }
  // for (int i_b = 0; i_b < N; i_b++) {
  //   B[i_b] = A[i_b]
  // }
  //
  // After the transformation the i_b loop should have an allocation for a temp
  // buffer and that buffer should be used in computation of B. No use of A
  // should be in that loop after the transformation. Also, computation of A
  // should not be inlined into B. Instead, it should be computed into the temp,
  // and the temp should be used in B.
  KernelScope kernel_scope;
  VarHandle N("N", kInt);
  Tensor* A = Compute(
      "A", {{N, "i_a"}}, [&](const VarHandle& i_a) { return i_a * i_a; });
  Tensor* B = Compute(
      "B", {{N, "i_b"}}, [&](const VarHandle& i_b) { return A->call(i_b); });
  LoopNest l({B});
  std::vector<For*> loops = l.getLoopStmtsFor(B);
  l.computeAt(l.getLoopBodyFor(A), loops[0]);
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

  std::ostringstream oss;
  oss << *s;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int i_b = 0; i_b < N; i_b++)
# CHECK:   Allocate(temp, int, {1})
# CHECK:   temp[
# CHECK-NOT: A[
# CHECK:   B[i_b] = temp[0]
# CHECK:   Free(temp))IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // Now check that the loop still produces the correct result.
  std::vector<int> b_data(100, 0);
  SimpleIREvaluator cg(s, {B, N});
  cg.call({b_data, 100});

  std::vector<int> b_ref(100, 0);
  for (int i = 0; i < 100; i++) {
    b_ref[i] = i * i;
  }
  assertAllEqual(b_data, b_ref);
}

void testLoopNestComputeAt_2() {
  // Verify that compute_at works on the following example:
  //
  // for (int py = 0; py < H+1; py++) {
  //   for (int px = 0; px < W+1; px++) {
  //     p[py, px] = py*px
  //   }
  // }
  // for (int cy = 0; cy < H; cy++) {
  //   for (int cx = 0; cx < W; cx++) {
  //     c[py, px] = p[cy,cx]   + p[cy+1,cx] +
  //                 p[cy,cx+1] + p[cy+1,cx+1]
  //   }
  // }
  KernelScope kernel_scope;

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  Tensor* p = Compute(
      "prod",
      {{H + 1, "py"}, {W + 1, "px"}},
      [&](const VarHandle& py, const VarHandle& px) { return px * py; });
  Tensor* c = Compute(
      "cons",
      {{H, "cy"}, {W, "cx"}},
      [&](const VarHandle& y, const VarHandle& x) {
        return p->call(y, x) + p->call(y + 1, x) + p->call(y, x + 1) +
            p->call(y + 1, x + 1);
      });

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = y * x + (y + 1) * x + y * (x + 1) + (y + 1) * (x + 1);
    }
  }

  {
    // First let's try to compute P at axis cy (the outer loop)
    LoopNest l({c});
    std::vector<For*> loops = l.getLoopStmtsFor(c);
    l.computeAt(l.getLoopBodyFor(p), loops[0]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   Allocate(temp, int, {2, W + 1})
# CHECK:   for
# CHECK:     for
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK:   Free(temp))IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {c, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute P at axis cx (the inner loop)
    LoopNest l({c});
    std::vector<For*> loops = l.getLoopStmtsFor(c);
    l.computeAt(l.getLoopBodyFor(p), loops[1]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK:     Allocate(temp, int, {2, 2})
# CHECK:     for
# CHECK:       for
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK:     Free(temp))IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {c, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}

void testLoopNestComputeAt_3() {
  // Verify that compute_at works on the following example:
  //
  // A(x,y) = x*y
  // B(x,y) = A(x, y)
  // C(x,y) = B(x+1, y)
  // D(x,y) = A(x, y+1) + C(x, y)
  //
  // i.e. when 'A' comes to 'D' directly and indirectly through 'C'.
  KernelScope kernel_scope;

  const int kW = 16, kH = 16;
  VarHandle W("W", kInt);
  VarHandle H("H", kInt);
  Tensor* A = Compute(
      "A",
      {{H + 1, "ay"}, {W + 1, "ax"}},
      [&](const VarHandle& ay, const VarHandle& ax) { return ax * ay; });
  Tensor* B = Compute(
      "B",
      {{H + 1, "by"}, {W + 1, "bx"}},
      [&](const VarHandle& by, const VarHandle& bx) {
        return A->call(by, bx);
      });
  Tensor* C = Compute(
      "C",
      {{H, "cy"}, {W, "cx"}},
      [&](const VarHandle& cy, const VarHandle& cx) {
        return B->call(cy, cx + 1);
      });
  Tensor* D = Compute(
      "D",
      {{H, "dy"}, {W, "dx"}},
      [&](const VarHandle& dy, const VarHandle& dx) {
        return A->call(dy + 1, dx) + C->call(dy, dx);
      });

  std::vector<int> c_ref(kW * kH, 0);
  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++) {
      c_ref[y * kW + x] = (y + 1) * x + y * (x + 1);
    }
  }

  {
    // First let's try to compute A at axis dy (the outer loop)
    LoopNest l({D});
    std::vector<For*> loops = l.getLoopStmtsFor(D);
    l.computeAt(l.getLoopBodyFor(A), loops[0]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int ay = 0; ay < H + 1; ay++)
# CHECK:   for (int ax = 0; ax < W + 1; ax++)
# CHECK:     A[
# CHECK: for (int by = 0; by < H + 1; by++)
# CHECK:   for (int bx = 0; bx < W + 1; bx++)
# CHECK:     B[
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK:     C[
# CHECK: for (int dy = 0; dy < H; dy++)
# CHECK:   Allocate(temp, int, {1, W})
# CHECK:   for (int dx = 0; dx < W; dx++)
# CHECK-NOT: A[)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {D, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
  {
    // Now let's try to compute A at axis dx (the inner loop)
    LoopNest l({D});
    std::vector<For*> loops = l.getLoopStmtsFor(D);
    l.computeAt(l.getLoopBodyFor(A), loops[1]);
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for (int ay = 0; ay < H + 1; ay++)
# CHECK:   for (int ax = 0; ax < W + 1; ax++)
# CHECK:     A[
# CHECK: for (int by = 0; by < H + 1; by++)
# CHECK:   for (int bx = 0; bx < W + 1; bx++)
# CHECK:     B[
# CHECK: for (int cy = 0; cy < H; cy++)
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK:     C[
# CHECK: for (int dy = 0; dy < H; dy++)
# CHECK:   for (int dx = 0; dx < W; dx++)
# CHECK:     Allocate(temp, int, {1, 1})
# CHECK-NOT: A[)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // Now check that the loop still produces the correct result.
    std::vector<int> c_data(kW * kH, 0);
    SimpleIREvaluator cg(s, {D, W, H});
    cg.call({c_data, kW, kH});

    assertAllEqual(c_data, c_ref);
  }
}

void testLoopNestComputeAt_4() {
  // TODO: Verify that computeAt works with reduction axis
}

class LoopOrderHelper : public IRVisitor {
  std::stringstream ordering;

 public:
  std::string getOrder(Stmt* s) {
    ordering.str("");
    s->accept(this);
    return ordering.str();
  }

  void visit(const For* v) {
    ordering << v->var()->name_hint() << ",";
    IRVisitor::visit(v);
  }
};

void testLoopNestReorderAxis1() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{2, "x"}, {3, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt1_output(6, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[0], loops[1]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  ASSERT_NE(stmt1, stmt2);
  LoopOrderHelper loopOrderHelper;
  std::string order1 = loopOrderHelper.getOrder(stmt1);
  std::string order2 = loopOrderHelper.getOrder(stmt2);

  ASSERT_EQ(order1, "x,y,");
  ASSERT_EQ(order2, "y,x,");

  std::vector<int> stmt2_output(6, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg.call({stmt2_output});

  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  // Reorder them back.
  loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[0], loops[1]);
  Stmt* stmt3 = l.root_stmt();

  std::string order3 = loopOrderHelper.getOrder(stmt3);
  ASSERT_EQ(order3, order1);

  std::ostringstream oss1, oss2;
  oss1 << *stmt1;
  oss2 << *stmt3;

  // Should be identical to the unreordered statement.
  ASSERT_EQ(oss1.str(), oss2.str());
}

void testLoopNestReorderPartialAxes() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f",
      {{2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  Stmt* stmt1 = Stmt::clone(l.root_stmt());
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "x,y,z,");

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[0], loops[1]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "y,x,z,");

  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }

  loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[1], loops[2]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "y,z,x,");

  Stmt* stmt3 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt3_output(24, 0);
  SimpleIREvaluator cg3(stmt3, {tensor});
  cg3.call({stmt3_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt3_output[i]);
  }
}

void testLoopNestReorderInternalAxis() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f",
      {{1, "w"}, {2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  Stmt* stmt1 = Stmt::clone(l.root_stmt());
  ASSERT_EQ(loopOrderHelper.getOrder(stmt1), "w,x,y,z,");

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[2], loops[1]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "w,y,x,z,");

  Stmt* stmt2 = l.root_stmt();

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

void testLoopNestReorderEnclosingAxis() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f",
      {{1, "w"}, {2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& w,
         const VarHandle& x,
         const VarHandle& y,
         const VarHandle& z) {
        return ExprHandle(1.0f) + w + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  LoopOrderHelper loopOrderHelper;
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> stmt1_output(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor});
  cg.call({stmt1_output});

  auto loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[0], loops[3]);
  ASSERT_EQ(loopOrderHelper.getOrder(l.root_stmt()), "z,x,y,w,");

  Stmt* stmt2 = l.root_stmt();

  std::vector<int> stmt2_output(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor});
  cg2.call({stmt2_output});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(stmt1_output[i], stmt2_output[i]);
  }
}

void testLoopNestReorderSameAxis() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{2, "x"}, {3, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  auto loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[1], loops[1]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;
  ASSERT_EQ(oss.str(), oss2.str());
}

void testLoopNestReorderExtraStatements() {
  /* We're going for a structure like this:
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *     for z in ...
   *       Stmt 3
   *     Stmt 4
   */

  KernelScope kernel_scope;

  Tensor* tensor = Compute(
      "f",
      {{2, "x"}, {3, "y"}, {4, "z"}},
      [](const VarHandle& x, const VarHandle& y, const VarHandle& z) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y +
            cast<float>(z) * z;
      });
  LoopNest l({tensor});

  Buffer extra(BufHandle("res", {6, 3}, kFloat));

  auto loops = l.getLoopStmtsFor(tensor);

  VarHandle i = VarHandle(loops[0]->var());

  Stmt* store_1 = Store::make(extra, {i, 0}, ExprHandle(1.f), 1);
  Stmt* store_2 = Store::make(extra, {i, 1}, ExprHandle(2.f), 1);
  // stmt 3 is the Function body.
  Stmt* store_3 = Store::make(extra, {i, 2}, ExprHandle(4.f), 1);

  loops[0]->body()->prepend_stmt(store_1);
  loops[1]->body()->prepend_stmt(store_2);
  loops[1]->body()->append_stmt(store_3);
  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> extra1(6, 0);
  std::vector<int> res1(24, 0);
  SimpleIREvaluator cg(stmt1, {tensor, extra});
  cg.call({res1, extra1});

  /* Then we reorder loop y and z, we want it to look like:
   *
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *   for z in ...
   *    for y in ...
   *       Stmt 3
   *   for y in ...
   *     Stmt 4
   *
   * We need extra loops because we don't have dependency info about stmt 3
   * and 4.
   *
   */

  l.reorderAxis(loops[1], loops[2]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  std::ostringstream oss;
  oss << *l.root_stmt();

  // Check the IR we produced
  const std::string& verification_pattern1 =
      R"IR(
# CHECK: for (int x
# CHECK:   res[x, 0] = 1
# CHECK:   for (int y
# CHECK:     res[x, 1] = 2
# CHECK:   for (int z
# CHECK:     for (int y
# CHECK:       f[
# CHECK:   for (int y
# CHECK:     res[x, 2] = 4
)IR";
  torch::jit::testing::FileCheck().run(verification_pattern1, oss.str());

  std::vector<int> extra2(6, 0);
  std::vector<int> res2(24, 0);
  SimpleIREvaluator cg2(stmt2, {tensor, extra});
  cg2.call({res2, extra2});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(res1[i], res2[i]);
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(extra1[i], extra2[i]);
  }

  /* Now reorder x and the y above stmt 3:
   *
   *
   * for x in ...
   *   Stmt 1
   *   for y in ...
   *     Stmt 2
   *
   * for y in ...
   *   for z in ...
   *    for x in ...
   *       Stmt 3
   *
   * for x in ...
   *   for y in ...
   *     Stmt 4
   *
   *
   */
  loops = l.getLoopStmtsFor(tensor);
  l.reorderAxis(loops[0], loops[2]);
  Stmt* stmt3 = Stmt::clone(l.root_stmt());

  std::ostringstream oss2;
  oss2 << *stmt3;

  // Check the IR we produced
  const std::string& verification_pattern2 =
      R"IR(
# CHECK: for (int x
# CHECK:   res[x, 0] = 1
# CHECK:   for (int y
# CHECK:     res[x, 1] = 2
# CHECK: for (int y
# CHECK:   for (int z
# CHECK:     for (int x
# CHECK:       f[
# CHECK: for (int x
# CHECK:   for (int y
# CHECK:     res[x, 2] = 4
)IR";
  torch::jit::testing::FileCheck().run(verification_pattern2, oss2.str());

  std::vector<int> extra3(6, 0);
  std::vector<int> res3(24, 0);
  SimpleIREvaluator cg3(stmt3, {tensor, extra});
  cg3.call({res3, extra3});

  for (int i = 0; i < 24; ++i) {
    ASSERT_EQ(res1[i], res3[i]);
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(extra1[i], extra3[i]);
  }
}

void LoopNestReorderTestHelper(
    bool prepend,
    bool append,
    int index1,
    int index2) {
  KernelScope kernel_scope;

  Tensor* c = Compute(
      "5d",
      {{2, "a"}, {3, "b"}, {2, "c"}, {3, "d"}, {2, "e"}},
      [](const std::vector<VarHandle>&) { return -1; });
  LoopNest l({c});

  Buffer extra(BufHandle("extra", {5}, kInt));

  auto loops = l.getLoopStmtsFor(c);
  int j = 0;
  for (auto* l : loops) {
    // Add an increment at each layer of the loop which counts the number of
    // times the loop executes.
    Load* load = new Load(extra, {new IntImm(j)}, new IntImm(1));
    Add* add = new Add(load, new IntImm(1));
    Stmt* store = Store::make(extra, {j}, ExprHandle(add), 1);
    if (prepend) {
      l->body()->prepend_stmt(store);
    }
    if (append) {
      l->body()->append_stmt(Stmt::clone(store));
    }

    j++;
  }

  Stmt* stmt1 = Stmt::clone(l.root_stmt());

  std::vector<int> extra1(5, 0);
  std::vector<int> res1(2 * 3 * 2 * 3 * 2, 0);
  SimpleIREvaluator cg(stmt1, {c, extra});
  cg.call({res1, extra1});

  std::vector<int> loopExtents = {2, 3, 2, 3, 2};

  int expected_loops = 0;
  if (prepend) {
    expected_loops++;
  }
  if (append) {
    expected_loops++;
  }
  for (int i = 0; i < 5; ++i) {
    expected_loops *= loopExtents[i];
    ASSERT_EQ(extra1[i], expected_loops);
  }

  loops = l.getLoopStmtsFor(c);
  l.reorderAxis(loops[index1], loops[index2]);
  Stmt* stmt2 = Stmt::clone(l.root_stmt());

  std::ostringstream oss, oss2;
  oss << *stmt1;
  oss2 << *stmt2;
  ASSERT_NE(oss.str(), oss2.str());

  std::vector<int> extra2(5, 0);
  std::vector<int> res2(2 * 3 * 2 * 3 * 2, 0);
  SimpleIREvaluator cg2(stmt2, {c, extra});
  cg2.call({res2, extra2});

  expected_loops = 0;
  if (prepend) {
    expected_loops++;
  }
  if (append) {
    expected_loops++;
  }

  for (int i = 0; i < 5; ++i) {
    expected_loops *= loopExtents[i];
    ASSERT_EQ(extra2[i], expected_loops);
  }

  for (int i = 0; i < 2 * 3 * 2 * 3 * 2; ++i) {
    ASSERT_EQ(res2[i], res1[i]);
  }
}

void testLoopNestReorderLongStringOfPreOrphans() {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // skip noops, since we check the loop isn't the same after reordering.
      if (i != j) {
        LoopNestReorderTestHelper(true, false, i, j);
      }
    }
  }
}

void testLoopNestReorderLongStringOfPostOrphans() {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // skip noops, since we check the loop isn't the same after reordering.
      if (i != j) {
        LoopNestReorderTestHelper(false, true, i, j);
      }
    }
  }
}

void testLoopNestReorderLongStringFull() {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      // skip noops, since we check the loop isn't the same after reordering.
      if (i != j) {
        LoopNestReorderTestHelper(true, true, i, j);
      }
    }
  }
}

void testLoopNestReorderInternalLoopNest() {
  KernelScope kernel_scope;
  const int M = 4;
  const int N = 5;
  const int K = 6;
  Buffer a_buf("a", kFloat, {M, N});
  Buffer b_buf("b", kFloat, {N, K});
  Buffer c_buf("c", kFloat, {M, N});
  Buffer d_buf("d", kFloat, {M, K});

  Tensor* x = Compute(
      "x",
      {{M, "m1"}, {N, "n1"}, {K, "k1"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf(m, n) * b_buf(n, k);
      });
  Tensor* y = Compute(
      "y",
      {{M, "m2"}, {N, "n2"}, {K, "k2"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c_buf(m, n) * d_buf(m, k) + x->call(m, n, k);
      });
  Tensor* z = Compute(
      "z",
      {{M, "m3"}, {N, "n3"}, {K, "k3"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return x->call(m, n, k) + y->call(m, n, k);
      });

  LoopNest l({z});
  For* a = nullptr;
  For* b = nullptr;
  auto fors = NodeFinder<For>::find(l.root_stmt());
  for (auto* f : fors) {
    if (f->var()->name_hint() == "m2") {
      a = f;
    } else if (f->var()->name_hint() == "k2") {
      b = f;
    }
  }
  l.reorderAxis(a, b);

  l.prepareForCodegen();
  Stmt* stmt = IRSimplifier::simplify(l.root_stmt());

  std::ostringstream oss;
  oss << *stmt;

  // Check the IR we produced has the 3 nests in the right order, but k and m
  // swapped in the middle.
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int m1
# CHECK:   for (int n1
# CHECK:     for (int k1
# CHECK: for (int k2
# CHECK:   for (int n2
# CHECK:     for (int m2
# CHECK: for (int m3
# CHECK:   for (int n3
# CHECK:     for (int k3)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  {
    PaddedBuffer<float> a_v(M, N);
    PaddedBuffer<float> b_v(N, K);
    PaddedBuffer<float> c_v(M, N);
    PaddedBuffer<float> d_v(M, K);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        a_v(i, j) = i * i;
      }
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        b_v(i, j) = j * j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c_v(i, j) = i + j;
      }
    }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        d_v(i, j) = i * j;
      }
    }

    PaddedBuffer<float> z_v(M, N, K);
    PaddedBuffer<float> z_ref(M, N, K);
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          z_ref(m, n, k) = a_v(m, n) * b_v(n, k) * 2 + c_v(m, n) * d_v(m, k);
        }
      }
    }

    SimpleIREvaluator eval(stmt, a_buf, b_buf, c_buf, d_buf, z);
    eval(a_v, b_v, c_v, d_v, z_v);
    ExpectAllNear(z_v, z_ref, 1e-5);
  }
}

void testOuterLoopVectorization() {
  KernelScope kernel_scope;
  Tensor* tensor = Compute(
      "f", {{8, "X"}, {8, "y"}}, [](const VarHandle& x, const VarHandle& y) {
        return ExprHandle(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  LoopNest l({tensor});

  l.vectorize(l.getLoopStmtsFor(tensor)[0]);

  Stmt* root_stmt = l.root_stmt();
  Block* outer_block = dynamic_cast<Block*>(root_stmt);
  ASSERT_NE(outer_block, nullptr);
  while (Block* inner_block = dynamic_cast<Block*>(outer_block->front())) {
    outer_block = inner_block;
  }

  // Verify that we have only a single loop level remaining after
  // vectorization.
  ASSERT_EQ(outer_block->nstmts(), 1);
  For* for_loop = dynamic_cast<For*>(outer_block->front());
  ASSERT_NE(for_loop, nullptr);
  Block* for_body = for_loop->body();
  ASSERT_EQ(for_body->nstmts(), 1);
  ASSERT_EQ(dynamic_cast<For*>(for_body->front()), nullptr);
}

namespace {

std::string constantUpperBoundLoopIR(int upper_bound_val) {
  KernelScope kernel_scope;
  ExprHandle upper_bound(upper_bound_val);
  Tensor* A = Compute(
      "A", {{upper_bound, "x"}}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  std::vector<For*> loops = l.getLoopStmtsFor(A);
  Stmt* unrolled = nullptr;
  LoopNest::unroll(loops[0], &unrolled);
  std::ostringstream oss;
  oss << *unrolled;
  return oss.str();
}

} // namespace

void testUnroll() {
  const std::string actual = constantUpperBoundLoopIR(3);
  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0] = 0;
# CHECK: A[1] = 2;
# CHECK: A[2] = 4)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

void testUnrollOuter() {
  KernelScope kernel_scope;
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  Tensor* A = Compute(
      "A",
      {{outer_bound, "x"}, {inner_bound, "y"}},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  std::vector<For*> loops = l.getLoopStmtsFor(A);
  Stmt* unrolled = nullptr;
  LoopNest::unroll(loops[0], &unrolled);
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int y = 0; y < 4; y++) {
# CHECK: A[0, y] = y;
# CHECK: }
# CHECK: for (int y = 0; y < 4; y++) {
# CHECK: A[1, y] = y + 1;
# CHECK: }
# CHECK: for (int y = 0; y < 4; y++) {
# CHECK: A[2, y] = y + 2;
# CHECK: })IR";

  std::ostringstream oss;
  oss << *unrolled;
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

void testUnrollInner() {
  KernelScope kernel_scope;
  ExprHandle outer_bound(3);
  ExprHandle inner_bound(4);
  Tensor* A = Compute(
      "A",
      {{outer_bound, "x"}, {inner_bound, "y"}},
      [&](const VarHandle& x, const VarHandle& y) { return x + y; });
  LoopNest l({A});
  std::vector<For*> loops = l.getLoopStmtsFor(A);
  Stmt* unrolled = nullptr;
  LoopNest::unroll(
      static_cast<For*>(loops[0]->body()->stmts().front()), &unrolled);
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x = 0; x < 3; x++) {
# CHECK: A[x, 0] = x;
# CHECK: A[x, 1] = x + 1;
# CHECK: A[x, 2] = x + 2;
# CHECK: A[x, 3] = x + 3;
# CHECK: })IR";

  std::ostringstream oss;
  oss << *loops[0];
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

void testUnrollMultipleStatements() {
  KernelScope kernel_scope;
  const int kTotalSize = 3;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  VarHandle x("x", kInt);
  auto f = For::make(
      x,
      0,
      kTotalSize,
      Block::make({Store::make(a_buf, {x}, x * 2),
                   Store::make(b_buf, {x}, Load::make(a_buf, {x}, 1))}));
  Block::make({f});
  Stmt* unrolled = nullptr;
  LoopNest::unroll(f, &unrolled);
  std::ostringstream oss;
  oss << *unrolled;
  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0] = 0;
# CHECK: B[0] = A[0];
# CHECK: A[1] = 2;
# CHECK: B[1] = A[1];
# CHECK: A[2] = 4
# CHECK: B[2] = A[2];)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

void testUnrollEmpty() {
  const std::string actual = constantUpperBoundLoopIR(0);
  const std::string& verification_pattern = R"IR(
# CHECK-NOT: A[
  )IR";

  torch::jit::testing::FileCheck().run(verification_pattern, actual);
}

void testNoUnroll() {
  KernelScope kernel_scope;
  VarHandle upper_bound("N", kInt);
  Tensor* A = Compute(
      "A", {{upper_bound, "x"}}, [&](const VarHandle& x) { return x * 2; });
  LoopNest l({A});
  std::vector<For*> loops = l.getLoopStmtsFor(A);
  Stmt* unrolled = nullptr;
  ASSERT_THROWS_WITH(
      LoopNest::unroll(loops[0], &unrolled), "non-constant loop");
}

void testUnrollWithLet() {
  KernelScope kernel_scope;
  const int kTotalSize = 3;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  VarHandle e("e", kInt);
  VarHandle x("x", kInt);
  auto f = For::make(
      x,
      0,
      kTotalSize,
      Block::make({Let::make(e, 7),
                   Store::make(a_buf, {x}, e),
                   Store::make(b_buf, {x}, e + 1)}));
  Block::make({f});
  Stmt* unrolled = nullptr;
  LoopNest::unroll(f, &unrolled);
  std::ostringstream oss;
  oss << *unrolled;
  const std::string& verification_pattern =
      R"IR(
# CHECK: int e = 7;
# CHECK: A[0] = e;
# CHECK: B[0] = e + 1;
# CHECK: A[1] = e;
# CHECK: B[1] = e + 1;
# CHECK: A[2] = e;
# CHECK: B[2] = e + 1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<int> a_v(kTotalSize, 0);
  std::vector<int> b_v(kTotalSize, 0);
  SimpleIREvaluator eval(unrolled, a_buf, b_buf);
  eval(a_v, b_v);
  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v[i], 7);
    ASSERT_EQ(b_v[i], 8);
  }
}

void testNormalizeStartPositive() {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int x = 50; x < 100; x++) {
  //     A[x] = B[x];
  //     B[x] = x * 2;
  //   }
  const int kTotalSize = 50;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_body =
      Block::make({Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x}, 1), 1),
                   Store::make(b_buf, {x}, x * 2)});
  auto for_stmt = For::make(x, 50, 100, for_body);
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(for_stmt, &normalized);

  auto result = IRSimplifier::simplify(normalized);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 50; x++) {
        # CHECK:   A[x + 50] = B[x + 50];
        # CHECK:   B[x + 50] = 2 * (x + 50);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

void testNormalizeStartNegative() {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int x = -50; x < 100; x++) {
  //     A[x + 50] = B[x + 50];
  //     B[x + 50] = x * 2;
  //   }
  const int kTotalSize = 150;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_body = Block::make(
      {Store::make(a_buf, {x + 50}, Load::make(kInt, b_buf, {x + 50}, 1), 1),
       Store::make(b_buf, {x + 50}, x * 2)});
  auto for_stmt = For::make(x, -50, 100, for_body);
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(for_stmt, &normalized);

  auto result = IRSimplifier::simplify(normalized);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 150; x++) {
        # CHECK:   A[x] = B[x];
        # CHECK:   B[x] = 2 * (x - 50);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

void testNormalizeStartZero() {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int x = 0; x < 100; x++) {
  //     A[x] = B[x];
  //     B[x] = x * 2;
  //   }
  // Should not be modified.

  const int kTotalSize = 100;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_body =
      Block::make({Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x}, 1), 1),
                   Store::make(b_buf, {x}, x * 2)});
  auto for_stmt = For::make(x, 0, 100, for_body);
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(for_stmt, &normalized);

  auto result = IRSimplifier::simplify(normalized);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100; x++) {
        # CHECK:   A[x] = B[x];
        # CHECK:   B[x] = 2 * x;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

void testNormalizeStartVariable() {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int x = y; x < 100; x++) {
  //     A[x] = B[x];
  //     B[x] = x * 2;
  //   }

  const int kTotalSize = 100;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto for_body =
      Block::make({Store::make(a_buf, {x}, Load::make(kInt, b_buf, {x}, 1), 1),
                   Store::make(b_buf, {x}, x * 2)});
  auto for_stmt = For::make(x, y, 100, for_body);
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(for_stmt, &normalized);

  auto result = IRSimplifier::simplify(normalized);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 100 - y; x++) {
        # CHECK:   A[y + x] = B[y + x];
        # CHECK:   B[y + x] = 2 * (y + x);
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

void testNormalizeOnNestedOuterLoop() {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int x = 50; x < 100; x++) {
  //     for (int y = 10; y < 100; y++) {
  //       A[x] = A[x] + B[y] + y * 2;
  //     }
  //   }

  BufHandle a_buf("A", {ExprHandle(50)}, kInt);
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto inner_for_body = Store::make(
      a_buf,
      {x},
      Load::make(a_buf, {x}, 1) + Load::make(b_buf, {y}, 1) + y * 2,
      1);
  auto inner_for = For::make(y, 10, 100, inner_for_body);
  auto for_stmt = For::make(x, 50, 100, inner_for);
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(for_stmt, &normalized);

  auto result = IRSimplifier::simplify(normalized);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 0; x < 50; x++) {
        # CHECK:   for (int y = 10; y < 100; y++) {
        # CHECK:     A[x + 50] = ((B[y]) + (A[x + 50])) + 2 * y;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

void testNormalizeOnNestedInnerLoop() {
  KernelScope kernel_scope;

  // Input IR:
  //   for (int x = 50; x < 100; x++) {
  //     for (int y = 10; y < 100; y++) {
  //       A[x] = A[x] + B[y] + y * 2;
  //     }
  //   }

  BufHandle a_buf("A", {ExprHandle(50)}, kInt);
  BufHandle b_buf("B", {ExprHandle(100)}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  auto inner_for_body = Store::make(
      a_buf,
      {x},
      Load::make(a_buf, {x}, 1) + Load::make(b_buf, {y}, 1) + y * 2,
      1);
  auto inner_for = For::make(y, 10, 100, inner_for_body);
  auto for_stmt = For::make(x, 50, 100, inner_for);
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(inner_for, &normalized);

  auto result = IRSimplifier::simplify(for_stmt);
  std::ostringstream oss;
  oss << *result;
  const std::string& expected_ir =
      R"IR(
        # CHECK: for (int x = 50; x < 100; x++) {
        # CHECK:   for (int y = 0; y < 90; y++) {
        # CHECK:     A[x] = (((B[y + 10]) + (A[x])) + 2 * y) + 20;
      )IR";
  torch::jit::testing::FileCheck().run(expected_ir, oss.str());
}

void testNormalizeAndSplitWithTail() {
  KernelScope kernel_scope;

  // Create a dummy tensor to construct LoopNest.
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});

  // Input IR:
  //   for (int x = 5; x < 10; x++) {
  //     A[x] = x * 2;
  //   }
  const int kTotalSize = 5;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  VarHandle x("x", kInt);
  auto for_stmt = For::make(x, 5, 10, Store::make(a_buf, {x}, x * 2));
  Block::make({for_stmt});

  For* normalized = nullptr;
  LoopNest::normalize(for_stmt, &normalized);

  For* x_outer;
  For* x_inner;
  For* x_tail;
  l.splitWithTail(normalized, 10, &x_outer, &x_inner, &x_tail);

  auto x_outer_result = IRSimplifier::simplify(x_outer);
  std::ostringstream oss_outer;
  oss_outer << *x_outer_result;
  const std::string& expected_outer_ir =
      R"IR(
        # CHECK: {
        # CHECK: }
      )IR";
  torch::jit::testing::FileCheck().run(expected_outer_ir, oss_outer.str());

  auto x_tail_result = IRSimplifier::simplify(x_tail);
  std::ostringstream oss_tail;
  oss_tail << *x_tail_result;
  const std::string& expected_tail_ir =
      R"IR(
        # CHECK: for (int x_tail = 0; x_tail < 5; x_tail++) {
        # CHECK:   A[x_tail + 5] = 2 * (x_tail + 5);
      )IR";
  torch::jit::testing::FileCheck().run(expected_tail_ir, oss_tail.str());
}

} // namespace jit
} // namespace torch
