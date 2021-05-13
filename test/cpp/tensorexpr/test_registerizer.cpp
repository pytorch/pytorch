#include <gtest/gtest.h>
#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/registerizer.h"

#include <iostream>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

// Can replace a simple scalar access with a local variable.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerSimple) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't do replacement of a loop access.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoop) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {x}, Add::make(Load::make(a, {x}), x))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  // No change.
  stmt = registerize(stmt);

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
# CHECK-NOT: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't replace even if the load is a fixed scalar, since the store could
// invalidate it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopFixedLoad) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {x}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[0]) + x;
   * }
   */

  // No change.
  stmt = registerize(stmt);

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[0]) + x;
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
# CHECK-NOT: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// We can registerize accesses that occur entirely within inner scopes, even if
// they depend on the loop var.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopInternal) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          {Store::make(a, {x}, Add::make(Load::make(a, {x}), x)),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), x))}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   *   A[x] = (A[x]) + x;
   * }
   */

  stmt = registerize(stmt);

  // TODO: the order of terms in addition changes and in general depends on
  // some hash value. This results in unpredictable swaps of the operands from
  // random changes, which is not great. Ideally, we should ensure some
  // specific order (ideally, the original one).
  /*
   * for (int x = 0; x < 10; x++) {
   *   int A_1 = A[x];
   *   A_1 = x + A_1;
   *   A_1 = x + A_1;
   *   A[x] = A_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x = 0; x < 10; x++)
# CHECK: int A_1 = A[x];
# CHECK:   A_1 = x + A_1;
# CHECK:   A_1 = x + A_1;
# CHECK:   A[x] = A_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// An access can be overlapped by another read in the same Expr. In this case
// B[z] and B[y] overlap and prevent registerization of both accesses.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopInternalLoadOverlap) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Store::make(a, {x}, Add::make(Load::make(b, {y}), Load::make(b, {z}))))});
  stmt = IRSimplifier::simplify(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (B[y]) + (B[z]);
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopInternalRepeated) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {1}), x)),
                Store::make(a, {0}, Add::make(Load::make(a, {1}), x))})),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {1}), x)),
                Store::make(a, {0}, Add::make(Load::make(a, {1}), x))}))

      });

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + (A[1]);
   *   A[0] = x + (A[1]);
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + (A[1]);
   *   A[0] = x + (A[1]);
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[1];
   * int A_2 = A[0];
   * for (int x = 0; x < 10; x++) {
   *   A_2 = x + A_1;
   *   A_2 = x + A_1;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A_2 = x + A_1;
   *   A_2 = x + A_1;
   * }
   * A[0] = A_2;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[1];
# CHECK: int A_2 = A[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   A_2 = x + A_1;
# CHECK:   A_2 = x + A_1;
# CHECK: }
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   A_2 = x + A_1;
# CHECK:   A_2 = x + A_1;
# CHECK: }
# CHECK-NOT: A[1]
# CHECK: A[0] = A_2;
# CHECK-NOT: A[1]
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopInternalRepeatedOverlapLoopVar) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {x}), x)),
                Store::make(a, {0}, Add::make(Load::make(a, {x}), x))})),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {x}), x)),
                Store::make(a, {0}, Add::make(Load::make(a, {x}), x))}))

      });
  stmt = IRSimplifier::simplify(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopInternalRepeatedOverlapOther) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make(
      {For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(x, Load::make(a, {y}))),
                Store::make(a, {0}, Add::make(x, Load::make(a, {y})))})),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(x, Load::make(a, {y}))),
                Store::make(a, {0}, Add::make(x, Load::make(a, {y})))}))

      });

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[x]) + x;
   *   A[0] = (A[x]) + x;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Will registerize multiple accesses of different items of the same buffer.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiVar) {
  KernelScope kernel_scope;
  BufHandle a("A", {2}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({
      Store::make(a, {0}, 0),
      Store::make(a, {1}, 0),
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {0}, Add::make(Load::make(a, {0}), x)),
               Store::make(a, {1}, Sub::make(Load::make(a, {1}), x))})),
  });

  /*
   * A[0] = 0;
   * A[1] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   *   A[1] = (A[1]) - x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * int A_2 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_2 = x + A_2;
   *   A_1 = A_1 - x;
   * }
   * A[1] = A_2;
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: int A_2 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK:   A_2 =
# CHECK: A[1] = A_2
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Will registerize the valid accesses while skipping invalid replacements.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerVariableLoad) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle x2("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(x, 0, 10, Store::make(b, {x}, x)),
       For::make(
           x2,
           0,
           10,
           Block::make({Store::make(
               a, {0}, Add::make(Load::make(a, {0}), Load::make(b, {x2})))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x;
   * }
   * for (int x_1 = 0; x_1 < 10; x_1++) {
   *   A[0] = (A[0]) + (B[x_1]);
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x;
   * }
   * for (int x_1 = 0; x_1 < 10; x_1++) {
   *   A_1 = A_1 + (B[x_1]);
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   B[x] = x
# CHECK: for (int x_1 = 0; x_1 < 10; x_1++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize variable accesses so long as the variable does not change.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerSymbolicIndices) {
  KernelScope kernel_scope;
  VarHandle i("i", kInt);
  VarHandle N("N", kInt);
  BufHandle a("A", {N}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {i}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {i}, Add::make(Load::make(a, {i}), x))}))});

  /*
   * A[i] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[i] = (A[i]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[i] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[i] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize accesses dependent on multiple loop vars.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiLoop) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
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
                   Mul::make(Add::make(Load::make(a, {0}), x), y))})))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     A[0] = x * y + (A[0]) * y;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     A_1 = x * y + y * A_1;
   *   }
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   for (int y = 0; y < 10; y++)
# CHECK-NOT: A[
# CHECK:     A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize correctly if scalars already exist in the program.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerRepeated) {
  KernelScope kernel_scope;
  BufHandle a("A", {2}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({
      Store::make(a, {0}, 0),
      Store::make(a, {1}, 0),
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {0}, Add::make(Load::make(a, {0}), x)),
               Store::make(a, {1}, Sub::make(Load::make(a, {1}), x))})),
  });

  // Registerize manually to make sure we only replace a single target.
  {
    registerizer::RegisterizerAnalysis analysis;
    stmt->accept(&analysis);
    auto candidates = analysis.getCandidates();
    ASSERT_EQ(candidates.size(), 2);

    candidates.pop_back();
    registerizer::RegisterizerReplacer replacer(candidates);
    stmt = stmt->accept_mutator(&replacer);
  }

  // Re-analyze and replace the second target.
  {
    registerizer::RegisterizerAnalysis analysis;
    stmt->accept(&analysis);
    auto candidates = analysis.getCandidates();
    ASSERT_EQ(candidates.size(), 1);

    registerizer::RegisterizerReplacer replacer(candidates);
    stmt = stmt->accept_mutator(&replacer);
  }

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: int A_1_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK:   A_1_1 =
# CHECK: A[1] = A_1_1;
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize the load of A.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNoLoads) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x, 0, 10, Block::make({Store::make(a, {0}, Add::make(x, 1))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = x + 1;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + 1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize the load of A but not the store of B.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNoRepeatedStores) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(b, {x}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);

  // TODO: its unnecessary to reorder the initializer of A[0], but it's not
  // actually worse so lets not worry for now.

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   B[x] = x + A_1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A_
# CHECK:   B[x] =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Won't registerize if there are multiple accesses which may overlap.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiVarOverlap) {
  KernelScope kernel_scope;
  BufHandle a("A", {2}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({
      Store::make(a, {0}, 0),
      Store::make(a, {1}, 0),
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {x}, Add::make(Load::make(a, {0}), x)),
               Store::make(a, {x + 1}, Sub::make(Load::make(a, {1}), x))})),
  });
  stmt = IRSimplifier::simplify(stmt);

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerAllocs) {
  KernelScope kernel_scope;

  BufHandle a("A", {2}, kInt);
  BufHandle c("C", {1}, kInt);
  VarHandle x("x", kInt);

  BufHandle b("B", {Load::make(c, {0})}, kInt);

  Stmt* stmt = Block::make(
      {Allocate::make(b),
       Store::make(a, {0}, Load::make(c, {0})),
       Store::make(b, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(b, {0}, Add::make(Load::make(b, {0}), x)),
                Store::make(a, {0}, Load::make(c, {0}))})),
       Free::make(b)});

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

  stmt = registerize(stmt);

  /*
   * int C_1 = C[0];
   * Allocate(B, int, {C_});
   * int A_1 = C_1;
   * int B_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   B_1 = B_1 + x;
   *   A_1 = C_1;
   * }
   * B[0] = B_1;
   * A[0] = A_1;
   * Free(B);
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int C_1 = C[0];
# CHECK: Allocate(B
# CHECK: int A_1 = C_1;
# CHECK: int B_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK:   B_1 =
# CHECK:   A_1 = C_
# CHECK: B[0] = B_1;
# CHECK: A[0] = A_1;
# CHECK: Free(B)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNoInitializer) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[0];
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNoInitializerLoopVar) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({Store::make(a, {x}, Add::make(Load::make(a, {x}), x))}))});
  stmt = IRSimplifier::simplify(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   A[x] = (A[x]) + x;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoadThenStore) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          {Store::make(b, {0}, Add::make(Load::make(a, {0}), x)),
           Store::make(a, {0}, Load::make(b, {0}))}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   B[0] = (A[0]) + x;
   *   A[0] = B[0];
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[0];
   * int B_1 = B[0];
   * for (int x = 0; x < 10; x++) {
   *   B_1 = x + A_1;
   *   A_1 = B_1;
   * }
   * B[0] = B_1;
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: int B_1 = B[0];
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: B[
# CHECK:   B_1 =
# CHECK-NOT: A[
# CHECK:   A_1 = B_
# CHECK: B[0] = B_
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerParallelized) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  LoopOptions loopOpts;
  loopOpts.set_gpu_block_index(0);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}),
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

// Should be able to registerize this since the scalar would exist before the
// branch.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionAfter) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Store::make(a, {x}, Load::make(b, {x})),
       Store::make(c, {x}, Load::make(a, {x})),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr)});

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];
   * C[x] = A_1;
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: C[x] = A_1;
# CHECK: if (
# CHECK:   A_1 = A_1 + 1;
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Should be able to registerize this since the scalar exists in the same form
// after the branch and there is no overlap.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionBefore) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr),
       Store::make(a, {x}, Load::make(b, {x})),
       Store::make(c, {x}, Load::make(a, {x}))});

  /*
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * A[x] = B[x];
   * C[x] = A[x];
   */

  stmt = registerize(stmt);

  /*
   * int A_ 1 = A[x];
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A_1 = B[x];
   * C[x] = A_1;
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if (
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: A_1 = B[x];
# CHECK: C[x] = A_1;
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Should be able to registerize this as the combination of the two above rules.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionInside) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Store::make(a, {x}, Load::make(b, {x})),
       Store::make(c, {x}, Load::make(a, {x})),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr),
       Store::make(b, {x}, Load::make(a, {x})),
       Store::make(a, {x}, Load::make(c, {x}))});

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * B[x] = A[x];
   * A[x] = C[x];
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];
   * C[x] = A_1;
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * B[x] = A_1;
   * A_1 = C[x];
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: C[x] = A_1;
# CHECK: if (
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: B[x] = A_1;
# CHECK: A_1 = C[x];
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// An example where an access is cut by an overlapping access inside a
// condition, and both sides are large enough to be registerized but cannot be
// because there is no safe place to put the initializer or finalizer.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionInsideOverlap1) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  Stmt* stmt = Block::make(
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      {Store::make(a, {x}, Load::make(b, {x})),
       Store::make(c, {x}, Load::make(a, {x})),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
               Store::make(a, {0}, 3),
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           }),
           nullptr),
       Store::make(b, {x}, Load::make(a, {x})),
       Store::make(a, {x}, Load::make(c, {x}))});

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   *   A[0] = 3;
   *   A[x] = (A[x]) + 1;
   * }
   * B[x] = A[x];
   * A[x] = C[x];
   */

  // The A[0] store overlaps, A[x] cutting the region that can be registerized
  // into two groups.
  // Each group has 2 loads and 2 stores however, so we could registerize it,
  // but the first group would need to be finalized inside the condition block,
  // the second would need to be initialized inside the condition block. There's
  // no safe place to put these that's visible to the other uses in the group
  // and so neither registerization is possible.

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Same as the above, but the access group before the condition (and after the
// condition) are large enough to be registerized without needing the access
// from the loop. Registerization occurs but does not include any accesses in
// the condition, and the first group must be finalized before the Cond, the
// second initialized after it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionInsideOverlap2) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  Stmt* stmt = Block::make(
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      {Store::make(a, {x}, Load::make(b, {x})),
       Store::make(a, {x}, Load::make(b, {x + 1})),
       Store::make(c, {x}, Load::make(a, {x})),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
               Store::make(a, {0}, 3),
               Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           }),
           nullptr),
       Store::make(b, {x}, Load::make(a, {x})),
       Store::make(b, {x + 1}, Load::make(a, {x})),
       Store::make(a, {x}, Load::make(c, {x}))});

  /*
   * A[x] = B[x];
   * A[x] = B[x + 1];
   * C[x] = A[x];
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   *   A[0] = 3;
   *   A[x] = (A[x]) + 1;
   * }
   * B[x] = A[x];
   * B[x + 1] = A[x];
   * A[x] = C[x];
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];              // A_1 initializer
   * A_1 = B[x + 1];              //
   * C[x] = A_1;                  //
   * A[x] = A_1;                  // A_1 finalizer
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   *   A[0] = 3;
   *   A[x] = (A[x]) + 1;
   * }
   * int A_2 = A[x];              // A_2 initialier
   * B[x] = A_2;                  //
   * B[x + 1] = A_2;              //
   * A_2 = C[x];                  //
   * A[x] = A_2;                  // A_2 finalizer
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: A_1 = B[x + 1];
# CHECK: C[x] = A_1;
# CHECK: A[x] = A_1;
# CHECK: if (
# CHECK-NOT:   A_1 = A_1 + 1;
# CHECK:   A[x] = (A[x]
# CHECK:   A[0] =
# CHECK:   A[x] = (A[x]
# CHECK: }
# CHECK: int A_2 = A[x];
# CHECK: B[x] = A_2;
# CHECK: B[x + 1] = A_2;
# CHECK: A_2 = C[x];
# CHECK: A[x] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// When accesses are within conditional blocks they are not visible to the wider
// program, because we don't know if the branch would be taken and if it isn't
// the accesses in it don't need to be valid (think size checks on the index).
// In this case the accesses cannot be registerized.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionHidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kGT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * if (x>5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// But... if the same access is found in a non conditional scope, that means
// that that access is valid in the higher scope (or at least if its not it's
// the user's fault). It "unhides" the conditional accesses, allowing
// registerization to occur.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionUnhidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr),
       Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kGT),
           Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   * A[x] = (A[x]) + 1;            <-- this is doing the unhiding.
   * if (x>5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * if (x<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A_1 = A_1 + 1;
   * if (x>5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * }
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if (x<5
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: A_1 = A_1 + 1;
# CHECK: if (x>5
# CHECK:   A_1 = A_1 + 1;
# CHECK: }
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize a load that occurs in the condition of a Cond.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerCondCondition) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Store::make(a, {x}, Load::make(b, {x})),
       Store::make(c, {x}, Load::make(a, {x})),
       Cond::make(
           CompareSelect::make(
               Load::make(a, {x}), 5, CompareSelectOperation::kLT),
           Store::make(c, {x}, Add::make(Load::make(c, {x}), 1)),
           nullptr)});

  /*
   * A[x] = B[x];
   * C[x] = A[x];
   * if ((A[x])<5 ? 1 : 0) {
   *   C[x] = (C[x]) + 1;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = B[x];
   * int C_1 = A_1;
   * if (A_1<5 ? 1 : 0) {
   *   C_1 = C_1 + 1;
   * }
   * C[x] = C_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = B[x];
# CHECK: int C_1 = A_1;
# CHECK: if (A_1<5
# CHECK:   C_1 = C_1 + 1;
# CHECK: C[x] = C_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Appearing in the condition of a Cond makes it visible to the enclosing scope,
// and so we can registerize internal usages.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerCondConditionUnhidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make({Cond::make(
      CompareSelect::make(Load::make(a, {x}), 5, CompareSelectOperation::kLT),
      Store::make(a, {x}, Add::make(Load::make(a, {x}), 1)),
      Store::make(a, {x}, Add::make(Load::make(a, {x}), 10)))});

  /*
   * if ((A[x])<5 ? 1 : 0) {
   *   A[x] = (A[x]) + 1;
   * } else {
   *   A[x] = (A[x]) + 10;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * if (A_1<5 ? 1 : 0) {
   *   A_1 = A_1 + 1;
   * } else {
   *   A_1 = A_1 + 10;
   * }
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if (A_1<5
# CHECK:   A_1 = A_1 + 1;
# CHECK: } else {
# CHECK:   A_1 = A_1 + 10;
# CHECK: }
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Conditional hiding also works for IfThenElse exprs.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseHidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  Stmt* stmt = Block::make(
      {Store::make(
           b,
           {y},
           IfThenElse::make(
               CompareSelect::make(x, 5, CompareSelectOperation::kLT),
               Add::make(Load::make(a, {x}), 1),
               Add::make(Load::make(a, {x + 1}), 2))),
       Store::make(
           b,
           {y + 1},
           IfThenElse::make(
               CompareSelect::make(x, 5, CompareSelectOperation::kLT),
               Add::make(Load::make(a, {x}), 1),
               Add::make(Load::make(a, {x + 1}), 2)))});

  /*
   * B[y] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   * B[y + 1] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Conditional unhiding also works for IfThenElse exprs.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseUnhidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  Stmt* stmt = Block::make({
      Store::make(a, {x}, 0),
      Store::make(
          b,
          {y},
          IfThenElse::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              Add::make(Load::make(a, {x}), 1),
              Add::make(Load::make(a, {x + 1}), 2))),
      Store::make(
          b,
          {y + 1},
          IfThenElse::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              Add::make(Load::make(a, {x}), 1),
              Add::make(Load::make(a, {x + 1}), 2))),
  });

  /*
   * A[x] = 0;
   * B[y] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   * B[y + 1] = IfThenElse(x<5 ? 1 : 0, (A[x]) + 1, (A[x + 1]) + 2);
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * B[y] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
   * B[y + 1] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: B[y] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
# CHECK: B[y + 1] = IfThenElse(x<5 ? 1 : 0, A_1 + 1, (A[x + 1]) + 2);
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Nested IfThenElse exprs can't promote to higher level scopes.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseNested) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  BufHandle d("D", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make({Store::make(
      a,
      {x},
      IfThenElse::make(
          CompareSelect::make(x, 3, CompareSelectOperation::kLT),
          IfThenElse::make(
              CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
              Load::make(d, {x}),
              Load::make(b, {x})),
          IfThenElse::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kEQ),
              Load::make(c, {x}),
              Load::make(d, {x}))))});

  /*
   * A[x] = IfThenElse(x<3 ? 1 : 0,
   *          IfThenElse(x==2 ? 1 : 0, D[x], B[x]),
   *            IfThenElse(x==5 ? 1 : 0, C[x], D[x]));
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Cannot registerize an access completely contained within an IfThenElse
// branch, since it is not a Stmt and cannot hold variable definitions. We need
// to check that we don't promote the initializer/finalizer to the enclosing
// Block.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseInternal) {
  KernelScope kernel_scope;
  // Making these floats so they don't get simplified to a single access.
  BufHandle a("A", {5}, kFloat);
  BufHandle b("B", {5}, kFloat);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make({Store::make(
      a,
      {x},
      IfThenElse::make(
          CompareSelect::make(x, 3, CompareSelectOperation::kLT),
          Add::make(Load::make(b, {x}), Load::make(b, {x})),
          Load::make(b, {x})))});

  /*
   * A[x] = IfThenElse(x<3 ? 1 : 0, (B[x]) + (B[x]), B[x]);
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());

  // If this was a Cond instead of an IfThenElse then we could registerize the
  // two accesses to B[x] in the True branch.

  // Actually lets verify that.

  stmt = Block::make({Cond::make(
      CompareSelect::make(x, 3, CompareSelectOperation::kLT),
      Store::make(a, {x}, Add::make(Load::make(b, {x}), Load::make(b, {x}))),
      Store::make(a, {x}, Load::make(b, {x})))});

  /*
   * if (x<3 ? 1 : 0) {
   *   A[x] = (B[x]) + (B[x]);
   * } else {
   *   A[x] = B[x];
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x<3 ? 1 : 0) {
   *   float B_1 = B[x];
   *   A[x] = B_1 + B_1;
   * } else {
   *   A[x] = B[x];
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK-NOT: float
# CHECK: if (x<3
# CHECK:   float B_1 =
# CHECK:   A[x] = B_1 + B_1
# CHECK: } else {
# CHECK:   A[x] = B[x]
# CHECK: }
# CHECK-NOT: A[x]
# CHECK-NOT: B[x])IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize a load that occurs in the condition of an IfThenElse;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseCondition) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make(
      {Store::make(a, {x}, Load::make(a, {x})),
       Store::make(
           a,
           {x},
           IfThenElse::make(
               CompareSelect::make(
                   Load::make(a, {x}), 5, CompareSelectOperation::kLT),
               Load::make(b, {0}),
               Load::make(c, {0})))});

  /*
   * A[x] = A[x];       <---- just here so there are enough accesses to combine.
   * A[x] = IfThenElse((A[x])<5 ? 1 : 0, B[0], C[0]);
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * A_1 = A_1;
   * A_1 = IfThenElse(A_1<5 ? 1 : 0, B[0], C[0]);
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: A_1 = IfThenElse(A_1<5 ? 1 : 0, B[0], C[0]);
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Appearing in the condition of a Cond makes it visible to the enclosing scope,
// and so we can registerize internal usages.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseConditionUnhidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make({Store::make(
      b,
      {x},
      IfThenElse::make(
          CompareSelect::make(
              Load::make(a, {x}), 5, CompareSelectOperation::kLT),
          Add::make(Load::make(a, {x}), 1),
          Add::make(Load::make(a, {x}), 10)))});

  /*
   * B[x] = IfThenElse((A[x])<5 ? 1 : 0, (A[x]) + 1, (A[x]) + 10);
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * B[x] = IfThenElse(A_1<5 ? 1 : 0, A_1 + 1, A_1 + 10);
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: B[x] = IfThenElse(A_1<5 ? 1 : 0, A_1 + 1, A_1 + 10);)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Cannot promote accesses internal to IfThenElse branches even if the enclosing
// scope if conditional.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerConditionBranchOnly) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make({
          Cond::make(
              CompareSelect::make(x, 5, CompareSelectOperation::kLT),
              Store::make(
                  a,
                  {x},
                  IfThenElse::make(
                      CompareSelect::make(x, 5, CompareSelectOperation::kLT),
                      Add::make(Load::make(a, {x}), x),
                      Add::make(Load::make(a, {x - 5}), x))),
              Store::make(
                  a,
                  {x - 5},
                  IfThenElse::make(
                      CompareSelect::make(x, 5, CompareSelectOperation::kLT),
                      Add::make(Load::make(a, {x}), x),
                      Add::make(Load::make(a, {x - 5}), x)))),
      }))});
  stmt = IRSimplifier::simplify(stmt);

  std::ostringstream before;
  before << *stmt;

  /* for (int x = 0; x < 10; x++) {
   *   if (x<5 ? 1 : 0) {
   *     A[x] = IfThenElse(x<5 ? 1 : 0, (A[x]) + x, (A[x - 5]) + x);
   *   } else {
   *     A[x - 5] = IfThenElse(x<5 ? 1 : 0, (A[x]) + x, (A[x - 5]) + x);
   *   }
   * }
   */

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// We can registerize an IfThenElse that appears in the condition branch of a
// Cond. This is a weird but valid thing to do.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerCondIfThenElse) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);

  Stmt* stmt = Block::make({Cond::make(
      CompareSelect::make(
          IfThenElse::make(
              CompareSelect::make(
                  Load::make(a, {x}), 5, CompareSelectOperation::kLT),
              Load::make(a, {x}),
              Load::make(b, {x})),
          x,
          CompareSelectOperation::kEQ),
      Store::make(c, {x}, Add::make(Load::make(c, {x}), 1)),
      nullptr)});

  /*
   * if ((IfThenElse((A[x])<5 ? 1 : 0, A[x], B[x]))==x ? 1 : 0) {
   *   C[x] = (C[x]) + 1;
   * }
   */

  stmt = registerize(stmt);

  // access to A can be registerized, but not B or C

  /*
   * int A_1 = A[x];
   * if ((IfThenElse(A_1<5 ? 1 : 0, A_1, B[x]))==x ? 1 : 0) {
   *   C[x] = (C[x]) + 1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: if ((IfThenElse(A_1<5 ? 1 : 0, A_1, B[x]
# CHECK:   C[x] = (C[x]) + 1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can registerize a conditional access in the RHS of a store unhidden by it's
// LHS, and hoist it out of a loop.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseLoop) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  Stmt* stmt = For::make(
      y,
      0,
      10,
      Store::make(
          a,
          {x},
          IfThenElse::make(
              CompareSelect::make(x, 3, CompareSelectOperation::kLT),
              Load::make(a, {x}),
              Load::make(b, {y}))));

  /*
   * for (int y = 0; y < 10; y++) {
   *   A[x] = IfThenElse(x<3 ? 1 : 0, A[x], B[y]);
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[x];
   * for (int y = 0; y < 10; y++) {
   *   A_1 = IfThenElse(x<3 ? 1 : 0, A_1, B[y]);
   * }
   * A[x] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[x];
# CHECK: for (
# CHECK:   A_1 = IfThenElse(x<3 ? 1 : 0, A_1, B[y]);
# CHECK: }
# CHECK: A[x] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Cannot registerize if the RHS overlaps the access creating visibility.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerIfThenElseLoopCut) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  Stmt* stmt = Block::make({For::make(
      y,
      0,
      10,
      Store::make(
          a,
          {x},
          IfThenElse::make(
              CompareSelect::make(x, 3, CompareSelectOperation::kLT),
              Load::make(a, {x}),
              Load::make(a, {y}))))});

  /*
   * for (int y = 0; y < 10; y++) {
   *   A[x] = IfThenElse(x<3 ? 1 : 0, A[x], A[y]);
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Simple case where an access is cut by an overlapping access later in the
// program, we can registerize up until the overlap.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialAfter) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}), x))})),
       For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1})))});

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (
# CHECK:   A_1 = x + A_1;
# CHECK: }
# CHECK: A[0] = A_1;
# CHECK: for (
# CHECK:   A[x] = A[x - 1];
# CHECK: }
# CHECK-NOT: A)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// We can registerize an access which overlaps a previous access, the
// initializer must be inserted after the previous access.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialBefore) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}))),
       Store::make(a, {0}, 0),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {0}, Add::make(Load::make(a, {0}), x))}))});

  /*
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   * A[0] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK-NOT: int
# CHECK: for (
# CHECK:   A[x] = A[x - 1];
# CHECK: }
# CHECK: int A_1 = 0;
# CHECK: for (
# CHECK:   A_1 = x + A_1;
# CHECK: }
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// The combination of the previous two tests, an access is cut by an overlapping
// access in both directions.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialInside) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x1("x1", kInt);
  VarHandle x2("x2", kInt);
  VarHandle x3("x3", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 2),
       For::make(
           x1, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x1))),
       For::make(x2, 1, 10, Store::make(a, {x2}, Load::make(a, {x2 - 1}))),
       For::make(
           x3, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x3)))});

  /*
   * A[0] = 2;
   * for (int x1 = 0; x1 < 10; x1++) {
   *   A[0] = (A[0]) + x1;
   * }
   * for (int x2 = 1; x2 < 10; x2++) {
   *   A[x2] = A[x2 - 1];
   * }
   * for (int x3 = 0; x3 < 10; x3++) {
   *   A[0] = (A[0]) + x3;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 2;
   * for (int x1 = 0; x1 < 10; x1++) {
   *   A_1 = x1 + A_1;
   * }
   * A[0] = A_1;
   * for (int x2 = 1; x2 < 10; x2++) {
   *   A[x2] = A[x2 - 1];
   * }
   * int A_2 = A[0];
   * for (int x3 = 0; x3 < 10; x3++) {
   *   A_2 = x3 + A_2;
   * }
   * A[0] = A_2;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 2;
# CHECK: for (
# CHECK:   A_1 = x1 + A_1;
# CHECK: }
# CHECK: A[0] = A_1;
# CHECK: for (
# CHECK:   A[x2] =
# CHECK: }
# CHECK: int A_2 = A[0];
# CHECK: for (
# CHECK:   A_2 = x3 + A_2;
# CHECK: }
# CHECK: A[0] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// An element could be registerized program wide but is cut by a conditional
// access, we should break this into two scalars and write back to the buffer
// before the condition.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialCondition) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 2),
       For::make(
           x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x))),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Store::make(a, {x}, Load::make(a, {x - 1})),
           nullptr),
       For::make(
           x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), x)))});

  /*
   * A[0] = 2;
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   * if (x<5 ? 1 : 0) {
   *   A[x] = A[x - 1];
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[0] = (A[0]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 2;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0] = A_1;
   * if (x<5 ? 1 : 0) {
   *   A[x] = A[x - 1];
   * }
   * int A_2 = A[0];
   * for (int x = 0; x < 10; x++) {
   *   A_2 = x + A_2;
   * }
   * A[0] = A_2;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 2;
# CHECK: for (
# CHECK:   A_1 = x + A_1;
# CHECK: }
# CHECK: A[0] = A_1;
# CHECK: if (
# CHECK:   A[x] =
# CHECK: }
# CHECK: int A_2 = A[0];
# CHECK: for (
# CHECK:   A_2 = x + A_2;
# CHECK: }
# CHECK: A[0] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Tests case where an access is cut by an internal conditional access which
// itself is registerized.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialConditionInternalCut) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 1),
       Store::make(a, {0}, 3),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({Store::make(a, {x}, 1), Store::make(a, {x}, 3)}),
           nullptr),
       Store::make(a, {0}, 4),
       Store::make(a, {0}, 6)});

  /*
   * A[0] = 1;
   * A[0] = 3;
   * if (x<5 ? 1 : 0) {
   *   A[x] = 1;
   *   A[x] = 3;
   * }
   * A[0] = 4;
   * A[0] = 6;
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 1;
   * A_1 = 3;
   * A[0] = A_1;
   * if (x<5 ? 1 : 0) {
   *   int A_2 = 1;
   *   A_2 = 3;
   *   A[x] = A_2;
   * }
   * int A_3 = 4;
   * A_3 = 6;
   * A[0] = A_3;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 1;
# CHECK: A_1 = 3
# CHECK: A[0] = A_1;
# CHECK: if (
# CHECK:   int A_2 = 1;
# CHECK:   A_2 = 3;
# CHECK:   A[x] = A_2;
# CHECK: }
# CHECK: int A_3 = 4;
# CHECK: A_3 = 6;
# CHECK: A[0] = A_3;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// First statment in condition closes outer access, but can be registerized with
// later statements.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialConditionInternalStart) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, 1),
       Store::make(a, {0}, 3),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({Store::make(a, {x}, 1), Store::make(a, {x}, 3)}),
           nullptr),
       Store::make(a, {x}, 4),
       Store::make(a, {x}, 6)});

  /*
   * A[0] = 1;
   * A[0] = 3;
   * if (x<5 ? 1 : 0) {
   *   A[x] = 1;
   *   A[x] = 3;
   * }
   * A[x] = 4;
   * A[x] = 6;
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 1;
   * A_1 = 3;
   * A[0] = A_1;
   * int A_2 = A[x];    <--- must read from the input here.
   * if (x<5 ? 1 : 0) {
   *   A_2 = 1;
   *   A_2 = 3;
   * }
   * A_2 = 4;
   * A_2 = 6;
   * A[x] = A_2;
   */

  // TODO: I suppose we could refactor with a conditional initializier?

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 1;
# CHECK: A_1 = 3
# CHECK: A[0] = A_1;
# CHECK: int A_2 = A[x];
# CHECK: if (
# CHECK:   A_2 = 1;
# CHECK:   A_2 = 3;
# CHECK: }
# CHECK: A_2 = 4;
# CHECK: A_2 = 6;
# CHECK: A[x] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// An access cuts two open overlaps and creates four scalar variables.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerPartialOverlapsTwo) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {1}, Load::make(a, {0})),
       Store::make(a, {0}, Load::make(a, {1})),
       Store::make(a, {0}, Load::make(a, {1})),
       For::make(x, 1, 10, Store::make(a, {x}, x)),
       Store::make(a, {1}, Load::make(a, {0})),
       Store::make(a, {0}, Load::make(a, {1})),
       Store::make(a, {0}, Load::make(a, {1}))});

  /*
   * A[1] = A[0];
   * A[0] = A[1];
   * A[0] = A[1];
   * for (int x = 1; x < 10; x++) {
   *   A[x] = x;
   * }
   * A[1] = A[0];
   * A[0] = A[1];
   * A[0] = A[1];
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[0];
   * int A_2 = A_1;
   * A_1 = A_2;
   * A_1 = A_2;
   * A[1] = A_2;
   * A[0] = A_1;
   * for (int x = 1; x < 10; x++) {
   *   A[x] = x;
   * }
   * int A_3 = A[0];
   * int A_4 = A_3;
   * A_3 = A_4;
   * A_3 = A_4;
   * A[1] = A_4;
   * A[0] = A_3;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: int A_2 = A_1;
# CHECK: A_1 = A_2;
# CHECK: A_1 = A_2;
# CHECK: A[1] = A_2;
# CHECK: A[0] = A_1;
# CHECK: for (
# CHECK:   A[x] = x;
# CHECK: }
# CHECK: int A_3 = A[0];
# CHECK: int A_4 = A_3;
# CHECK: A_3 = A_4;
# CHECK: A_3 = A_4;
# CHECK: A[1] = A_4;
# CHECK: A[0] = A_3;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Nested blocks will automatically be flattened and do not provent
// registerization of enclosed accesses.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedBlocks) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      {Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
       Block::make({Store::make(a, {0}, Add::make(Load::make(a, {0}), 2))}),
       Block::make(
           {Store::make(a, {0}, Add::make(Load::make(a, {0}), 3)),
            Block::make(
                {Store::make(a, {0}, Add::make(Load::make(a, {0}), 4))})})});

  /*
   * A[0] = (A[0]) + 1;
   * {
   *   A[0] = (A[0]) + 2;
   * }
   * {
   *   A[0] = (A[0]) + 3;
   *   {
   *     A[0] = (A[0]) + 4;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[0];
   * A_1 = A_1 + 1;
   * A_1 = A_1 + 2;
   * A_1 = A_1 + 3;
   * A_1 = A_1 + 4;
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: A_1 = A_1 + 1;
# CHECK: A_1 = A_1 + 2;
# CHECK: A_1 = A_1 + 3;
# CHECK: A_1 = A_1 + 4;
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// The access can be registerized internally to a condition, but must ensure
// that both initializer and finalizer are within the same condition.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditions) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make({Cond::make(
      CompareSelect::make(x, 5, CompareSelectOperation::kLT),
      Block::make(
          {Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           Cond::make(
               CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
               Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
               nullptr)}),
      nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   *   if (x==2 ? 1 : 0) {
   *
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x<5 ? 1 : 0) {
   *   int A_1 = A[0];
   *   A_1 = A_1 + 1;
   *   if (x==2 ? 1 : 0) {
   *     A_1 = A_1 + 1;
   *   }
   * A[0] = A_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x<5
# CHECK:   int A_1 = A[0];
# CHECK:   A_1 = A_1 + 1;
# CHECK:   if (x==2
# CHECK:     A_1 = A_1 + 1;
# CHECK:   }
# CHECK: A[0] = A_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// If an access exists outside the scope of the condition then we can lift
// nested conditional usages into the same scalar.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditionsUnhidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make(
               {Store::make(a, {1}, 1),
                Cond::make(
                    CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
                    Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
                    nullptr)}),
           nullptr)});

  /*
   * A[0] = (A[0]) + 1;
   * if (x<5 ? 1 : 0) {
   *   A[1] = 1;
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = A[0];
   * A_1 = A_1 + 1;
   * if (x<5 ? 1 : 0) {
   *   A[1] = 1;
   *   if (x==2 ? 1 : 0) {
   *     A_1 = A_1 + 1;
   *   }
   * }
   * A[0] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = A[0];
# CHECK: A_1 = A_1 + 1;
# CHECK: if (x<5
# CHECK:   A[1] = 1;
# CHECK:   if (x==2
# CHECK:     A_1 = A_1 + 1;
# CHECK: A[0] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditionsHiddenFirst) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
           Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           nullptr),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({Cond::make(
               CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
               Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
               nullptr)}),
           nullptr)});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   * }
   * if (x<5 ? 1 : 0) {
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  stmt = registerize(stmt);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditionsHiddenSecond) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make({Cond::make(
               CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
               Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
               nullptr)}),
           nullptr),
       Cond::make(
           CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
           Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   * if (x==2 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  stmt = registerize(stmt);
}

// If an access is cut by another access internal to a condition block, it still
// cuts the access.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditionsCut) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           Block::make(
               {Store::make(a, {x}, 1),
                Cond::make(
                    CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
                    Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
                    nullptr)}),
           nullptr)});

  /*
   * A[0] = (A[0]) + 1;
   * if (x<5 ? 1 : 0) {
   *   A[x] = 1;
   *   if (x==2 ? 1 : 0) {
   *
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditionLoopHidden) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
           Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
           nullptr),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(b, {x}, 0),
                Cond::make(
                    CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
                    Store::make(a, {0}, Add::make(Load::make(a, {0}), 1)),
                    nullptr)}))});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = (A[0]) + 1;
   * }
   * for (int x = 0; x < 10; x++) {
   *   B[x] = 0;     <-- this is only here to prevent Loop/Cond reordering.
   *   if (x==2 ? 1 : 0) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// Three loops and four element regions, three of which should be registerized
// at different levels of the IR.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedConditionThreeDeep) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {4}, 0),
       Cond::make(
           CompareSelect::make(x, 2, CompareSelectOperation::kGT),
           Cond::make(
               CompareSelect::make(x, 3, CompareSelectOperation::kGT),
               Block::make({
                   Cond::make(
                       CompareSelect::make(x, 4, CompareSelectOperation::kGT),
                       Block::make({
                           Store::make(
                               a, {1}, Add::make(Load::make(a, {1}), 1)),
                           Store::make(
                               a, {2}, Add::make(Load::make(a, {2}), 1)),
                           Store::make(
                               a, {3}, Add::make(Load::make(a, {3}), 1)),
                           Store::make(
                               a, {4}, Add::make(Load::make(a, {4}), 1)),
                           Store::make(
                               a, {1}, Add::make(Load::make(a, {1}), 1)),
                       }),
                       nullptr),
                   Store::make(a, {2}, Add::make(Load::make(a, {2}), 1)),
               }),
               nullptr),
           nullptr)});

  /*
   * A[4] = 0;
   * if (x>2 ? 1 : 0) {
   *   if (x>3 ? 1 : 0) {
   *     if (x>4 ? 1 : 0) {
   *       A[1] = (A[1]) + 1;
   *       A[2] = (A[2]) + 1;
   *       A[3] = (A[3]) + 1;
   *       A[4] = (A[4]) + 1;
   *       A[1] = (A[1]) + 1;
   *     }
   *     A[2] = (A[2]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * if (x>2 ? 1 : 0) {
   *   if (x>3 ? 1 : 0) {
   *     int A_3 = A[2];
   *     if (x>4 ? 1 : 0) {
   *       int A_2 = A[1];
   *       A_2 = A_2 + 1;
   *       A_3 = A_3 + 1;
   *       A[3] = (A[3]) + 1;
   *       A_1 = A_1 + 1;
   *       A_2 = A_2 + 1;
   *       A[1] = A_2;
   *     }
   *     A_3 = A_3 + 1;
   *     A[2] = A_3;
   *   }
   * }
   * A[4] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: if (x>2 ? 1 : 0) {
# CHECK:   if (x>3 ? 1 : 0) {
# CHECK:     int A_3 = A[2];
# CHECK:     if (x>4 ? 1 : 0) {
# CHECK:       int A_2 = A[1];
# CHECK:       A_2 = A_2 + 1;
# CHECK:       A_3 = A_3 + 1;
# CHECK:       A[3] = (A[3]) + 1;
# CHECK:       A_1 = A_1 + 1;
# CHECK:       A_2 = A_2 + 1;
# CHECK:       A[1] = A_2;
# CHECK:     }
# CHECK:     A_3 = A_3 + 1;
# CHECK:     A[2] = A_3;
# CHECK:   }
# CHECK: }
# CHECK: A[4] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Can replace a simple scalar access with a local variable even when that
// variable is an outer loop var.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerNestedLoopSimple) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make({For::make(
      y,
      0,
      10,
      For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(a, {y}, Add::make(Load::make(a, {y}), x))})))});

  /*
   * for (int y = 0; y < 10; y++) {
   *   for (int x = 0; x < 10; x++) {
   *     A[y] = (A[y]) + x;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * for (int y = 0; y < 10; y++) {
   *   int A_1 = A[y];
   *   for (int x = 0; x < 10; x++) {
   *     A_1 = x + A_1;
   *   }
   * A[y] = A_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int y
# CHECK:   int A_1 = A[y];
# CHECK:   for (int x
# CHECK:     A_1 = x + A_1;
# CHECK:   }
# CHECK:   A[y] = A_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Test the positive case of the hiddenAccess split, where an internal
// conditional access can be hoisted up through a loop to match an existing
// access in a higher scope and the two can be registerized.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerHiddenAccessYes) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make({Cond::make(
      CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
      Block::make(
          {Store::make(a, {0}, 0),
           For::make(
               x,
               0,
               10,
               Block::make(
                   {Store::make(b, {x}, 0),
                    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
                    Cond::make(
                        CompareSelect::make(x, 3, CompareSelectOperation::kEQ),
                        For::make(
                            y,
                            0,
                            10,
                            Store::make(
                                a, {0}, Add::make(Load::make(a, {0}), 1))),
                        nullptr)}))}),
      nullptr)});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = 0;
   *   for (int x = 0; x < 10; x++) {
   *     B[x] = 0;
   *     if (x==3 ? 1 : 0) {
   *       for (int y = 0; y < 10; y++) {
   *         A[0] = (A[0]) + 1;
   *       }
   *     }
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x==2 ? 1 : 0) {
   *   int A_1 = 0;
   *   for (int x = 0; x < 10; x++) {
   *     B[x] = 0;
   *     if (x==3 ? 1 : 0) {
   *       for (int y = 0; y < 10; y++) {
   *         A_1 = A_1 + 1;
   *       }
   *     }
   *   }
   *   A[0] = A_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x==2
# CHECK:   int A_1 = 0;
# CHECK:   for (int x
# CHECK:     B[x] = 0;
# CHECK:     if (x==3
# CHECK:       for (int y
# CHECK:         A_1 = A_1 + 1;
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK:  A[0] = A_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Test the negative case of the hiddenAccess split, where the hoisted access is
// never unhidden at a higher scope and registerization occurs at the lower
// scope.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerHiddenAccessNo) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make({Cond::make(
      CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
      Block::make({For::make(
          x,
          0,
          10,
          Block::make(
              {Store::make(b, {x}, 0),
               // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
               Cond::make(
                   CompareSelect::make(x, 3, CompareSelectOperation::kEQ),
                   For::make(
                       y,
                       0,
                       10,
                       Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
                   nullptr)}))}),
      nullptr)});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = 0;
   *   for (int x = 0; x < 10; x++) {
   *     B[x] = 0;
   *     if (x==3 ? 1 : 0) {
   *       for (int y = 0; y < 10; y++) {
   *         A[0] = (A[0]) + 1;
   *       }
   *     }
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x==2 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     B[x] = 0;
   *     if (x==3 ? 1 : 0) {
   *       int A_1 = A[0];
   *       for (int y = 0; y < 10; y++) {
   *         A_1 = A_1 + 1;
   *       }
   *       A[0] = A_1;
   *     }
   *   }
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x==2
# CHECK:   for (int x
# CHECK:     B[x] = 0;
# CHECK:     if (x==3
# CHECK:       int A_1 = A[0];
# CHECK:       for (int y
# CHECK:         A_1 = A_1 + 1;
# CHECK:       }
# CHECK:       A[0] = A_1;
# CHECK:     }
# CHECK:   }
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// In this case the conditional access must be hoisted by two loops, there are
// two accesses here one is unhidden and the other isnt. A[0] can be
// registerized but B[0] cannot.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerHiddenAccessMultiLoop) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make({Cond::make(
      CompareSelect::make(x, 2, CompareSelectOperation::kEQ),
      Block::make(
          {Store::make(a, {0}, 0),
           For::make(
               x,
               0,
               10,
               For::make(
                   y,
                   0,
                   10,
                   Block::make({Cond::make(
                       CompareSelect::make(y, 3, CompareSelectOperation::kEQ),
                       Block::make(
                           {Store::make(
                                a, {0}, Add::make(Load::make(a, {0}), 1)),
                            Store::make(
                                b, {0}, Add::make(Load::make(b, {0}), 1))}),
                       nullptr)})))}),
      nullptr)});

  /*
   * if (x==2 ? 1 : 0) {
   *   A[0] = 0;
   *   for (int x = 0; x < 10; x++) {
   *     for (int y = 0; y < 10; y++) {
   *       if (y==3 ? 1 : 0) {
   *         A[0] = (A[0]) + 1;
   *         B[0] = (B[0]) + 1;
   *       }
   *     }
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x==2 ? 1 : 0) {
   *   int A_1 = 0;
   *   for (int x = 0; x < 10; x++) {
   *     for (int y = 0; y < 10; y++) {
   *       if (y==3 ? 1 : 0) {
   *         A_1 = A_1 + 1;
   *         B[0] = (B[0]) + 1;
   *       }
   *     }
   *   }
   *   A[0] = A_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x==2
# CHECK:   int A_1 = 0;
# CHECK:   for (int x
# CHECK:     for (int y
# CHECK:       if (y==3
# CHECK:         A_1 = A_1 + 1;
# CHECK:         B[0] = (B[0]) + 1;
# CHECK:       }
# CHECK:     }
# CHECK:   }
# CHECK:  A[0] = A_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Accesses are registerized inside two conditions, but the immeidate parent is
// not a condition.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerTwoConditionalLoops) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
           nullptr),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kGT),
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   * if (x>5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x<5 ? 1 : 0) {
   *   int A_1 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_1 = A_1 + 1;
   *   }
   *   A[0] = A_1;
   * }
   * if (x>5 ? 1 : 0) {
   *   int A_2 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_2 = A_2 + 1;
   *   }
   *   A[0] = A_2;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x<5
# CHECK:   int A_1 = A[0];
# CHECK:   for (int x
# CHECK:     A_1 = A_1 + 1;
# CHECK:   }
# CHECK:   A[0] = A_1;
# CHECK: }
# CHECK: if (x>5
# CHECK:   int A_2 = A[0];
# CHECK:   for (int x
# CHECK:     A_2 = A_2 + 1;
# CHECK:   }
# CHECK:   A[0] = A_2;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Accesses are registerized inside two conditions, cut in the middle.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerTwoConditionalLoopsCut) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kLT),
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
           nullptr),
       For::make(x, 0, 10, Store::make(a, {x}, 1)),
       Cond::make(
           CompareSelect::make(x, 5, CompareSelectOperation::kGT),
           For::make(
               x, 0, 10, Store::make(a, {0}, Add::make(Load::make(a, {0}), 1))),
           nullptr)});

  /*
   * if (x<5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[x] = 1;
   * }
   * if (x>5 ? 1 : 0) {
   *   for (int x = 0; x < 10; x++) {
   *     A[0] = (A[0]) + 1;
   *   }
   * }
   */

  stmt = registerize(stmt);

  /*
   * if (x<5 ? 1 : 0) {
   *   int A_1 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_1 = A_1 + 1;
   *   }
   *   A[0] = A_1;
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[x] = 1;
   * }
   * if (x>5 ? 1 : 0) {
   *   int A_2 = A[0];
   *   for (int x = 0; x < 10; x++) {
   *     A_2 = A_2 + 1;
   *   }
   *   A[0] = A_2;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: if (x<5
# CHECK:   int A_1 = A[0];
# CHECK:   for (int x
# CHECK:     A_1 = A_1 + 1;
# CHECK:   }
# CHECK:   A[0] = A_1;
# CHECK: }
# CHECK: for (int x
# CHECK:  A[x] = 1;
# CHECK: if (x>5
# CHECK:   int A_2 = A[0];
# CHECK:   for (int x
# CHECK:     A_2 = A_2 + 1;
# CHECK:   }
# CHECK:   A[0] = A_2;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// references a Let var in a local scope which cannot be hoisted out of the
// loop.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopLetVar) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make({For::make(
      x,
      0,
      10,
      Block::make(
          {Let::make(y, 30),
           Store::make(a, {y}, Add::make(x, Load::make(a, {y})))}))});

  /*
   * for (int x = 0; x < 10; x++) {
   *   int y = 30;
   *   A[y] = x + (A[y]);
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// references a Let var in an outer scope that does not prevent hoisting the
// initializer.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerLoopLetVarOuter) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make(
      {Let::make(y, 30),
       For::make(
           x,
           0,
           10,
           Block::make(
               {Store::make(a, {y}, Add::make(x, Load::make(a, {y})))}))});

  /*
   * int y = 30;
   * for (int x = 0; x < 10; x++) {
   *   A[y] = x + (A[y]);
   * }
   */

  stmt = registerize(stmt);

  /*
   * int y = 30;
   * int A_1 = A[y];
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[y] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int y = 30;
# CHECK: int A_1 = A[y];
# CHECK: for (int x
# CHECK:   A_1 = x + A_1;
# CHECK: A[y] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Okay so the registerizer generally goes after index flattening, but just in
// case. Test multi index registerization.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiDim) {
  KernelScope kernel_scope;
  BufHandle a("A", {3, 4, 5}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0, 1, 2}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(
               a, {0, 1, 2}, Add::make(Load::make(a, {0, 1, 2}), x))}))});

  /*
   * A[0, 1, 2] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0, 1, 2] = (A[0, 1, 2]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * int A_1 = 0;
   * for (int x = 0; x < 10; x++) {
   *   A_1 = x + A_1;
   * }
   * A[0, 1, 2] = A_1;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: int A_1 = 0;
# CHECK: for (int x = 0; x < 10; x++)
# CHECK-NOT: A[
# CHECK:   A_1 =
# CHECK: A[0, 1, 2] = A_1;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// Wont registerize if only some dims match, but will still registerize distinct
// elements.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiDimPartial) {
  KernelScope kernel_scope;
  BufHandle a("A", {3, 4, 5}, kInt);
  VarHandle x("x", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0, 1, 2}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(
               a, {0, 2, 2}, Add::make(Load::make(a, {0, 1, 4}), x))}))});

  /*
   * A[0, 1, 2] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0, 2, 2] = (A[0, 1, 4]) + x;
   * }
   */

  stmt = registerize(stmt);

  /*
   * A[0, 1, 2] = 0;
   * int A_1 = A[0, 1, 4];
   * int A_2 = A[0, 2, 2];
   * for (int x = 0; x < 10; x++) {
   *   A_2 = x + A_1;
   * }
   * A[0, 2, 2] = A_2;
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0, 1, 2] = 0;
# CHECK: int A_1 = A[0, 1, 4];
# CHECK: int A_2 = A[0, 2, 2];
# CHECK: for (
# CHECK:   A_2 = x + A_1;
# CHECK: A[0, 2, 2] = A_2;)IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// If they could overlap across all dimensions we cannot registerize.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiDimOverlap) {
  KernelScope kernel_scope;
  BufHandle a("A", {3, 4, 5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0, 1, 2}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(
               a, {0, x, 2}, Add::make(Load::make(a, {y, 2, 2}), x))}))});
  stmt = IRSimplifier::simplify(stmt);

  /*
   * A[0, 1, 2] = 0;
   * for (int x = 0; x < 10; x++) {
   *   A[0, x, 2] = (A[y, 2, 2]) + x;
   * }
   */

  std::ostringstream before;
  before << *stmt;

  // No change.
  stmt = registerize(stmt);

  std::ostringstream after;
  after << *stmt;

  ASSERT_EQ(before.str(), after.str());
}

// But, if one dimension is known to be distinct they do not overlap.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiDimPartialOverlap) {
  KernelScope kernel_scope;
  BufHandle a("A", {3, 4, 5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  Stmt* stmt = Block::make(
      {Store::make(a, {0, 1, 2}, 0),
       For::make(
           x,
           0,
           10,
           Block::make({Store::make(
               a, {0, x, 2}, Add::make(Load::make(a, {y, 2, 4}), x))}))});

  /*
   * A[0, 1, 2] = 0;                          <---- 2nd dim overlaps with store.
   * for (int x = 0; x < 10; x++) {
   *   A[0, x, 2] = (A[y, 2, 4]) + x;           <---- 3rd dim has constant diff.
   * }
   */

  stmt = registerize(stmt);

  /*
   * A[0, 1, 2] = 0;
   * int A_1 = A[y, 2, 4];
   * for (int x = 0; x < 10; x++) {
   *   A[0, x, 2] = x + A_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: A[0, 1, 2] = 0;
# CHECK: int A_1 = A[y, 2, 4];
# CHECK: for (
# CHECK:   A[0, x, 2] = x + A_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// A 3D reduction with different input dimensionality.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiDim3DReduction1) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10, 10}, kInt);
  BufHandle c("C", {10, 10, 10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  Stmt* stmt = For::make(
      x,
      0,
      10,
      For::make(
          y,
          0,
          10,
          For::make(
              z,
              0,
              10,
              Store::make(
                  c,
                  {x, y, z},
                  Add::make(
                      Load::make(c, {x, y, z}),
                      Mul::make(Load::make(b, {x, y}), Load::make(a, {x})))))));

  /*
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     for (int z = 0; z < 10; z++) {
   *       C[x, y, z] = (C[x, y, z]) + (B[x, y]) * (A[x]);
   *     }
   *   }
   * }
   */

  // We can registerize the A and B access since they can be hoisted before
  // hitting a dependent loop var.

  stmt = registerize(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   int A_1 = A[x];
   *   for (int y = 0; y < 10; y++) {
   *     int B_1 = B[x, y];
   *     for (int z = 0; z < 10; z++) {
   *       C[x, y, z] = A_1 * B_1 + (C[x, y, z]);
   *     }
   *   }
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x
# CHECK:   int A_1 = A[x];
# CHECK:   for (int y
# CHECK:     int B_1 = B[x, y];
# CHECK:       for (int z
# CHECK:         C[x, y, z] = A_1 * B_1 + (C[x, y, z]);
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

// A 3D reduction with the same smaller dimensionality using different loop
// vars.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Registerizer, RegisterizerMultiDim3DReduction2) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  BufHandle c("C", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  Stmt* stmt = For::make(
      x,
      0,
      10,
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      For::make(
          y,
          0,
          10,
          For::make(
              z,
              0,
              10,
              Store::make(
                  c,
                  {x},
                  Add::make(
                      Load::make(c, {x}),
                      Mul::make(Load::make(b, {y}), Load::make(a, {x})))))));

  /*
   * for (int x = 0; x < 10; x++) {
   *   for (int y = 0; y < 10; y++) {
   *     for (int z = 0; z < 10; z++) {
   *       C[x] = (C[x]) + (B[y]) * (A[x]);
   *     }
   *   }
   * }
   */

  // We can registerize all accesses, the A and C access can be hoisted to the
  // outer loop since they depend only on it's loop var while the B can only be
  // raised to the loop of y.

  stmt = registerize(stmt);

  /*
   * for (int x = 0; x < 10; x++) {
   *   int C_1 = C[x];
   *   int A_1 = A[x];
   *   for (int y = 0; y < 10; y++) {
   *     int B_1 = B[y];
   *     for (int z = 0; z < 10; z++) {
   *       C_1 = C_1 + A_1 * B_1;
   *     }
   *   }
   *   C[x] = C_1;
   * }
   */

  std::ostringstream oss;
  oss << *stmt;

  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int x
# CHECK:   int C_1 = C[x];
# CHECK:   int A_1 = A[x];
# CHECK:   for (int y
# CHECK:     int B_1 = B[y];
# CHECK:       for (int z
# CHECK:         C_1 = C_1 + A_1 * B_1;
# CHECK:       }
# CHECK:     }
# CHECK:   C[x] = C_1;
# CHECK: })IR";

  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

} // namespace jit
} // namespace torch
