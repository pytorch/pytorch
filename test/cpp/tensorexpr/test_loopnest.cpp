#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <test/cpp/tensorexpr/test_base.h>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/function.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;
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
    BufHandle f("f", {26, 5});
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
            For::make(
                y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y), 1))));
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
    BufHandle f("f", {24, 5});
    ExprHandle x_1 = x_outer * 4 + x_inner;
    ExprHandle x_outer_end = (ExprHandle(24) - 0) / 4;
    For* stmt = For::make(
        x_outer,
        0,
        x_outer_end,
        For::make(
            x_inner,
            0,
            4,
            For::make(
                y, 0, 5, Store::make(f, {x_1, y}, func(x_1, y), 1))));

    std::ostringstream oss_ref;
    oss_ref << *stmt;
    oss_ref << "\n"; // TODO: fix printing instead of adding \n here
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

  Buffer a_buf(BufHandle("A", {ExprHandle(kTotalSize)}), kFloat);

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

  Buffer a(BufHandle("A", {ExprHandle(kTotalSize)}), kFloat);
  Buffer b(BufHandle("B", {ExprHandle(kTotalSize)}), kFloat);
  Buffer c(BufHandle("C", {ExprHandle(kTotalSize)}), kFloat);
  Buffer d(BufHandle("D", {ExprHandle(kTotalSize)}), kFloat);

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
    Buffer a(BufHandle("a", {m, n}), kFloat);
    Buffer b(BufHandle("b", {m, n}), kFloat);
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

static std::unordered_map<const Buf*, TensorAccessBoundsInfo>
convertBoundsInfoToMap(const std::vector<TensorAccessBoundsInfo>& v) {
  std::unordered_map<const Buf*, TensorAccessBoundsInfo> res;
  for (const auto& el : v) {
    res[el.buf] = el;
  }
  return res;
}

static void verifyConstBounds(
    const TensorAccessBoundsInfo& access_info,
    const std::vector<std::pair<int, int>>& ref) {
  size_t ndim = ref.size();
  ASSERT_EQ(access_info.start.size(), ndim);
  ASSERT_EQ(access_info.stop.size(), ndim);
  for (size_t i = 0; i < ndim; i++) {
    if (ref[i].first >= 0) { // Negative values are used to skip the check
      auto start_imm = dynamic_cast<const IntImm*>(access_info.start[i]);
      ASSERT_TRUE(start_imm);
      ASSERT_EQ(start_imm->value(), ref[i].first);
    }
    if (ref[i].second >= 0) {
      auto stop_imm = dynamic_cast<const IntImm*>(access_info.stop[i]);
      ASSERT_TRUE(stop_imm);
      ASSERT_EQ(stop_imm->value(), ref[i].second);
    }
  }
}

void testBoundsInference_1() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 99}}
  KernelScope kernel_scope;
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n}), kFloat);
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});
  const std::vector<TensorAccessBoundsInfo>& bounds_info =
      inferBounds(l.root_stmt());
  auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info_map.size(), 2);
  ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
  verifyConstBounds(bounds_info_map.at(a.data()), {{0, 99}});
  ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kStore);
  verifyConstBounds(bounds_info_map.at(b->buf()), {{0, 99}});
}

void testBoundsInference_2() {
  // Verify that bounds inference works for the following example:
  // for i in 0..n:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, n-1}, {a, kLoad, 0, n-1}}
  KernelScope kernel_scope;
  VarHandle n("n", kInt);
  Buffer a(BufHandle("a", {n}), kFloat);
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});
  const std::vector<TensorAccessBoundsInfo>& bounds_info =
      inferBounds(l.root_stmt());
  auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info_map.size(), 2);
  ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
  verifyConstBounds(bounds_info_map.at(a.data()), {{0, -1}});
  ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kStore);
  verifyConstBounds(bounds_info_map.at(b->buf()), {{0, -1}});
}

void testBoundsInference_3() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i] * a[i+10]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 109}}
  KernelScope kernel_scope;
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n + 10}), kFloat);
  Tensor* b = Compute(
      "b", {{n, "i"}}, [&](const VarHandle& i) { return a(i) * a(i + 10); });
  LoopNest l({b});
  const std::vector<TensorAccessBoundsInfo>& bounds_info =
      inferBounds(l.root_stmt());
  auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info_map.size(), 2);
  ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
  verifyConstBounds(bounds_info_map.at(a.data()), {{0, 109}});
  ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kStore);
  verifyConstBounds(bounds_info_map.at(b->buf()), {{0, 99}});
}

void testBoundsInference_4() {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..200:
  //   for x in 0..320:
  //     c[y,x] = a[y,x] * b[y,x]
  KernelScope kernel_scope;
  ExprHandle W(320);
  ExprHandle H(200);
  Buffer a(BufHandle("a", {H, W}), kFloat);
  Tensor* b = Compute(
      "b", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return x * y;
      });
  Tensor* c = Compute(
      "c", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return a(y, x) * b->call(y, x);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  Stmt* body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    const std::vector<TensorAccessBoundsInfo>& bounds_info =
        inferBounds(loops[0]);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info_map.at(c->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(c->buf()), {{0, 199}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop scope
    const std::vector<TensorAccessBoundsInfo>& bounds_info =
        inferBounds(loops[1]);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info_map.at(c->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(c->buf()), {{-1, -1}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop body's scope
    const std::vector<TensorAccessBoundsInfo>& bounds_info =
        inferBounds(body);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info_map.at(c->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(c->buf()), {{-1, -1}, {-1, -1}});
  }
}

void testBoundsInference_5() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  //
  // ==> split ==>
  //
  // for i_outer in 0..100/16:
  //   for i_inner in 0..16:
  //     b[i_outer * 16 + i_inner] = a[i_outer * 16 + i_inner]
  // for i_tail in 0..100%16:
  //   b[i_tail + (100/16)*16] = a[i_tail + (100/16)*16];
  KernelScope kernel_scope;
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n}), kFloat);
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});

  For* outer;
  For* inner;
  For* tail;
  std::vector<For*> loops = l.getLoopStmtsFor(b);
  l.splitWithTail(loops[0], 16, &outer, &inner, &tail);

  {
    // Verify inferred bounds for the outer loop
    const std::vector<TensorAccessBoundsInfo>& bounds_info = inferBounds(outer);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);
    ASSERT_EQ(bounds_info_map.size(), 2);
    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{0, 95}});
    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{0, 95}});
  }
  {
    // Verify inferred bounds for the tail loop
    const std::vector<TensorAccessBoundsInfo>& bounds_info = inferBounds(tail);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);
    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{96, 99}});
    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{96, 99}});
  }
}

void testBoundsInference_6() {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..20:
  //   for x in 0..32:
  //     c[y,x] = a[y+100,x+100] * b[y*2,x*5]
  KernelScope kernel_scope;
  ExprHandle W(320);
  ExprHandle H(200);
  ExprHandle CW(32);
  ExprHandle CH(20);
  Buffer a(BufHandle("a", {H, W}), kFloat);
  Tensor* b = Compute(
      "b", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return x * y;
      });
  Tensor* c = Compute(
      "c", {{CH, "y"}, {CW, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return a(y + 100, x + 100) * b->call(y * 2, x * 5);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  Stmt* body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    const std::vector<TensorAccessBoundsInfo>& bounds_info =
        inferBounds(loops[0]);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{100, 119}, {100, 131}});

    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{0, 38}, {0, 155}});

    ASSERT_EQ(bounds_info_map.at(c->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(c->buf()), {{0, 19}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop scope
    const std::vector<TensorAccessBoundsInfo>& bounds_info =
        inferBounds(loops[1]);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{-1, -1}, {100, 131}});

    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{-1, -1}, {0, 155}});

    ASSERT_EQ(bounds_info_map.at(c->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(c->buf()), {{-1, -1}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop body's scope
    const std::vector<TensorAccessBoundsInfo>& bounds_info =
        inferBounds(body);
    auto bounds_info_map = convertBoundsInfoToMap(bounds_info);

    ASSERT_EQ(bounds_info_map.at(a.data()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(a.data()), {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info_map.at(b->buf()).kind, kLoad);
    verifyConstBounds(bounds_info_map.at(b->buf()), {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info_map.at(c->buf()).kind, kStore);
    verifyConstBounds(bounds_info_map.at(c->buf()), {{-1, -1}, {-1, -1}});
  }
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
# CHECK:  Allocate
# CHECK-NOT: A[
# CHECK:  B[i_b] =
# CHECK:  Free)IR";

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
# CHECK:   Allocate
# CHECK:   for
# CHECK:     for
# CHECK:   for (int cx = 0; cx < W; cx++)
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK:  Free)IR";
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
# CHECK: {2, 2}
# CHECK:     for
# CHECK:       for
# CHECK-NOT: prod[
# CHECK:     cons[
# CHECK:     Free)IR";
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
# CHECK:  {1, W}
# CHECK:   for (int dx = 0; dx < W; dx++)
# CHECK-NOT: A[
# CHECK:  Free)IR";
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
# CHECK:  {1, 1}
# CHECK-NOT: A[
# CHECK:  Free)IR";
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

} // namespace jit
} // namespace torch
