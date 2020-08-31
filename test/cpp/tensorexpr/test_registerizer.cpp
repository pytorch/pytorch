#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/registerizer.h"

#include <iostream>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

// Can replace a simple scalar access with a local variable.
void testRegisterizerSimple() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt =
      Block::make({Store::make(a, {0}, 0, 1),
                   For::make(
                       x,
                       0,
                       10,
                       Block::make({Store::make(
                           a, {0}, Add::make(Load::make(a, {0}, 1), x), 1)}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  registerize(stmt);

  /*
   * int A_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_ = x + A_;
   * }
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't do replacement of a loop access.
void testRegisterizerLoop() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {10}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt =
      Block::make({Store::make(a, {0}, 0, 1),
                   For::make(
                       x,
                       0,
                       10,
                       Block::make({Store::make(
                           a, {x}, Add::make(Load::make(a, {x}, 1), x), 1)}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  // No change.
  registerize(stmt);

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK: A[0] = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
# CHECK:   A[x] =
# CHECK-NOT: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't replace even if the load is a fixed scalar, since the store could
// invalidate it.
void testRegisterizerLoopFixedLoad() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt =
      Block::make({Store::make(a, {0}, 0, 1),
                   For::make(
                       x,
                       0,
                       10,
                       Block::make({Store::make(
                           a, {x}, Add::make(Load::make(a, {0}, 1), x), 1)}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  // No change.
  registerize(stmt);

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK: A[0] = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
# CHECK:   A[x] =
# CHECK-NOT: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Will registerize multiple accesses of different items of the same buffer.
void testRegisterizerMultiVar() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {2}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({
      Store::make(a, {0}, 0, 1),
      Store::make(a, {1}, 0, 1),
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {0}, Add::make(Load::make(a, {0}, 1), x), 1),
               Store::make(a, {1}, Sub::make(Load::make(a, {1}, 1), x), 1)})),
  });

  /*
   * A[0] = 0;
   * A[1] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   *   A[1] = (A[1]) - x;
   * }
   */

  registerize(stmt);

  /*
   * int A_ = 0;
   * int A__1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A__1 = x + A__1;
   *   A_ = A_ - x;
   * }
   * A[1] = A__1;
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: int A__1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK:   A__1 =
# CHECK: A[1] = A__1
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Will registerize the valid accesses while skipping invalid replacements.
void testRegisterizerVariableLoad() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  Buffer b(BufHandle("B", {10}, kInt));
  VarHandle x("x", kInt);
  VarHandle x2("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0, 1),
       For::make(x, 0, 10, Store::make(b, {x}, x, 1)),
       For::make(
           x2,
           0,
           10,
           Block::make({Store::make(
               a,
               {0},
               Add::make(Load::make(a, {0}, 1), Load::make(b, {x2}, 1)),
               1)}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x;
   * }
   * for (int x_1 = 0; x_1 < 10; x_1++) {
   *   A[0] = (A[0]) + (B[x_1]);
   * }
   */

  registerize(stmt);

  /*
   * int A_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x;
   * }
   * for (int x_1 = 0; x_1 < 10; x_1++) {
   *   A_ = A_ + (B[x_1]);
   * }
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   B[x] = x
# CHECK: for (int x_1 = 0; x_1 < 10; x_1++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize variable accesses so long as the variable does not change.
void testRegisterizerSymbolicIndices() {
  KernelScope kernel_scope;
  VarHandle i("i", kInt);
  VarHandle N("N", kInt);
  Buffer a(BufHandle("A", {N}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt =
      Block::make({Store::make(a, {i}, 0, 1),
                   For::make(
                       x,
                       0,
                       10,
                       Block::make({Store::make(
                           a, {i}, Add::make(Load::make(a, {i}, 1), x), 1)}))});

  /*
   * A[i] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[i] = (A[i]) + x;
   * }
   */

  registerize(stmt);

  /*
   * int A_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_ = x + A_;
   * }
   * A[i] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK: A[i] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Will not registerize if a variable usage of the sclar may overlap the target
// scalar.
// TODO: we can support this by writing back to the buffer before the variable
// access, but we'd need temporal analysis of dependencies which we don't have
// yet. Will have to fix soon though.
void testRegisterizerEarlyStop() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0, 1),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}, 1), x), 1)})),
       For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}, 1), 1))});

  std::ostringstream before;
  before << *stmt;

  // No change.
  registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Can registerize accesses dependent on multiple loop vars.
void testRegisterizerMultiLoop() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0, 1),
       For::make(
           x,
           0,
           10,
           For::make(
               y,
               0,
               10,
               Block::make({Store::make(
                   a,
                   {0},
                   Mul::make(Add::make(Load::make(a, {0}, 1), x), y),
                   1)})))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     A[0] = x * y + (A[0]) * y;
   *   }
   * }
   */

  registerize(stmt);

  /*
   * int A_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     A_ = x * y + y * A_l
   *   }
   * }
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   for (int y = 0; y < 10; y++)
# CHECK-NOT: A[
# CHECK:     A_ =
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize correctly if scalars already exist in the program.
void testRegisterizerRepeated() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {2}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({
      Store::make(a, {0}, 0, 1),
      Store::make(a, {1}, 0, 1),
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {0}, Add::make(Load::make(a, {0}, 1), x), 1),
               Store::make(a, {1}, Sub::make(Load::make(a, {1}, 1), x), 1)})),
  });

  // Registerize manually to make sure we only replace a single target.
  {
    RegisterizerAnalysis analysis;
    stmt->accept(&analysis);
    auto candidates = analysis.getCandidates();
    ASSERT_EQ(candidates.size(), 2);

    RegisterizerReplacer replacer(candidates.front());
    stmt = stmt->accept_mutator(&replacer);
  }

  // Re-analyze and replace the second target.
  {
    RegisterizerAnalysis analysis;
    stmt->accept(&analysis);
    auto candidates = analysis.getCandidates();
    ASSERT_EQ(candidates.size(), 1);

    RegisterizerReplacer replacer(candidates.front());
    stmt = stmt->accept_mutator(&replacer);
  }

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: int A__1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK:   A__1 =
# CHECK: A[1] = A__1
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize rthe load of A.
void testRegisterizerNoLoads() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0, 1),
       For::make(
           x, 0, 10, Block::make({Store::make(a, {0}, Add::make(x, 1), 1)}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + 1;
   * }
   */

  registerize(stmt);

  /*
   * int A_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_ = x + 1;
   * }
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize the load of A but not the store of B.
void testRegisterizerNoRepeatedStores() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  Buffer b(BufHandle("B", {10}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt =
      Block::make({Store::make(a, {0}, 0, 1),
                   For::make(
                       x,
                       0,
                       10,
                       Block::make({Store::make(
                           b, {x}, Add::make(Load::make(a, {0}, 1), x), 1)}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = (A[0]) + x;
   * }
   */

  registerize(stmt);

  // TODO: its unnecessary to reorder the initializer of A[0], but it's not
  // actually worse so lets not worry for now.

  /*
   * int A_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x + A_;
   * }
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
# CHECK:   B[x] =
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't registerize if there are multiple accesses which may overlap.
void testRegisterizerMultiVarOverlap() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {2}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({
      Store::make(a, {0}, 0, 1),
      Store::make(a, {1}, 0, 1),
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {x}, Add::make(Load::make(a, {0}, 1), x), 1),
               Store::make(
                   a, {x + 1}, Sub::make(Load::make(a, {1}, 1), x), 1)})),
  });

  std::ostringstream before;
  before << *stmt;

  // No change.
  registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

void testRegisterizerAllocs() {
  KernelScope kernel_scope;

  Buffer a(BufHandle("A", {2}, kInt));
  Buffer b(BufHandle("B", {1}, kInt));
  Buffer c(BufHandle("C", {1}, kInt));
  VarHandle x("x", kInt);

  VarHandle b_(b.data()->base_handle());

  Stmt* stmt = Block::make(
      {Allocate::make(b_, kInt, {Load::make(c, {0}, 1)}),
       Store::make(a, {0}, Load::make(c, {0}, 1), 1),
       Store::make(b, {0}, 0, 1),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(b, {0}, Add::make(Load::make(b, {0}, 1), x), 1),
                Store::make(a, {0}, Load::make(c, {0}, 1), 1)})),
       Free::make(b_)});

  /*
   * Allocate(B, int, {C[0]});
   * A[0] = C[0];
   * B[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[0] = (B[0]) + x;
   *   A[0] = C[0];
   * }
   * Free(B);
   */

  registerize(stmt);

  /*
   * int C_ = C[0];
   * Allocate(B, int, {C_});
   * int A_ = C_;
   * int B_ = 0;
   * for (int x = 0; x < 10; x++) {
   *   B_ = B_ + x;
   *   A_ = C_;
   * }
   * B[0] = B_;
   * A[0] = A_;
   * Free(B);
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int C_ = C[0];
# CHECK: Allocate(B
# CHECK: int A_ = C_;
# CHECK: int B_ = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   B_ =
# CHECK:   A_ = C_
# CHECK: B[0] = B_;
# CHECK: A[0] = A_;
# CHECK: Free(B)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

void testRegisterizerNoInitializer() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          {Store::make(a, {0}, Add::make(Load::make(a, {0}, 1), x), 1)}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  registerize(stmt);

  /*
   * int A_ = A[0];
   * for (int x = 0; x < 10; x++) {
   *   A_ = x + A_;
   * }
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = A[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_ =
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

void testRegisterizerLoadThenStore() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  Buffer b(BufHandle("B", {1}, kInt));
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({Store::make(b, {0}, Add::make(Load::make(a, {0}, 1), x), 1),
                   Store::make(a, {0}, Load::make(b, {0}, 1), 1)}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   B[0] = (A[0]) + x;
   *   A[0] = B[0];
   * }
   */

  registerize(stmt);

  /*
   * int A_ = A[0];
   * int B_ = B[0];
   * for (int x = 0; x < 10; x++) {
   *   B_ = x + A_;
   *   A_ = B_;
   * }
   * B[0] = B_;
   * A[0] = A_;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_ = A[0];
# CHECK: int B_ = B[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: B[
# CHECK:   B_ =
# CHECK-NOT: A[
# CHECK:   A_ = B_
# CHECK: B[0] = B_
# CHECK: A[0] = A_;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

void testRegisterizerParallelized() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {1}, kInt));
  VarHandle x("x", kInt);
  LoopOptions loopOpts;
  loopOpts.set_gpu_block_index(0);
  Stmt* stmt =
      Block::make({Store::make(a, {0}, 0, 1),
                   For::make(
                       x,
                       0,
                       10,
                       Block::make({Store::make(
                           a, {0}, Add::make(Load::make(a, {0}, 1), x), 1)}),
                       loopOpts)});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  ASSERT_THROWS_WITH(
      registerize(stmt),
      "Registerization must occur after parallelism flattening");
}

} // namespace jit
} // namespace torch
