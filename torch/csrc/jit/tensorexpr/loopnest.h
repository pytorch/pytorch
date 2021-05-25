#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class Var;
class Buf;
class Tensor;
class Function;
class Stmt;
class For;
class Block;
class Store;
class Dtype;

class TORCH_API LoopNest {
 public:
  // A constructor for building a LoopNest from a list of Tensors
  LoopNest(
      const std::vector<Tensor*>& output_tensors,
      const std::vector<Tensor*>& tensors_to_compute);

  // A convenience constructor for the case when all tensors are output tensors
  LoopNest(const std::vector<Tensor*>& output_tensors);

  // A constructor for building a LoopNest from an Stmt and a list of output
  // buffers.
  LoopNest(Stmt* stmt, std::unordered_set<const Buf*> output_bufs);

  // A constructor for building a LoopNest from another loopnest. It clones the
  // other loopnest's stmt.
  LoopNest(const LoopNest& other);

  Stmt* root_stmt() const {
    return root_stmt_;
  }

  std::vector<For*> getLoopStmtsFor(Tensor*) const;
  std::vector<For*> getLoopStmtsFor(const Buf*) const;
  std::vector<For*> getLoopStmtsFor(Stmt*) const;
  Stmt* getLoopBodyFor(Tensor*) const;
  Stmt* getLoopBodyFor(const Buf*) const;

  // Returns the For stmt that is immediately enclosing the given stmt.
  static For* getParentLoop(const Stmt* st);

  // Returns the list of For stmts corresponding to the loopnest that is
  // enclosing the given stmt.
  static std::vector<For*> getEnclosingLoopNest(const Stmt* st);

  // Returns a list of all Stmts that write to the given buf.
  std::vector<const Stmt*> getAllWritesToBuf(const Buf*) const;

  // The following methods return the For loops that contain writes to
  // the given buf.
  //
  // For example, consider the following code:
  //   for i1
  //     for j1
  //       a[i1,j1] =
  //   for i2
  //     for j2
  //       for k2
  //         a[i2,j2] =
  //     for j3
  //       a[i2,j3] =

  // Returns a list of For loops which directly contain a Stmt that writes
  // to buf.
  // For the above example:
  //   getAllInnermostLoopsWritingToBuf(a) => {j1, k2, j3}
  std::vector<For*> getAllInnermostLoopsWritingToBuf(const Buf*) const;

  // Returns a list of For loopnests which contain a Stmt that writes to
  // the given buf. Each loopnest here is a vector For loops.
  // For the above example:
  //   getAllLoopNestsWritingToBuf(a) => {{i1,j1}, {i2,j2,k2}, {i2,j3}}
  std::vector<std::vector<For*>> getAllLoopNestsWritingToBuf(const Buf*) const;

  static void vectorize(For*);
  Stmt* simplify();

  bool computeInline(Stmt* s);
  bool computeInline(const Buf* b);
  void inlineIntermediateBufs(bool allow_duplicated_work);

  // Optimizes conditionals.
  //
  // Currently, only the following pattern of conditionals is optimized.
  // This corresponds to the conditional format that is generated to handle
  // `aten::cat` op.
  //
  //   for (int i = 0; i < 20; i++) {
  //     A[i] = IfThenElse(i<5 ? 1 : 0, B[i], C[i-5])
  //   }
  //
  // Constraints that must be satisfied for this optimization:
  //   * All conditions should be of the form "var < expr".
  //   * All conditions should have the same variable, say v.
  //   * The condition variable found should be the same as the inner-most
  //     loop variable. TODO: Remove this constraint.
  //   * If there are multiple stores that contain conditionals using the same
  //     loop variable, only the first conditional will be optimized.
  //     TODO: Remove this constraint.
  bool optimizeConditionals();

  static void splitWithTail(For* f, int factor);
  static void splitWithTail(
      For* f,
      int factor,
      For** inner,
      For** tail);

  static void splitWithMask(For* f, int factor);
  static void splitWithMask(For* f, int factor, For** outer, For** inner);

  // The following methods support loop distribution.
  // For example, consider the following code. This will be used to
  // demonstrate the methods below.
  //
  // S1:  for i
  // S2:    A[i] = 0
  // S3:    for j
  // S4:      A[i] = A[i] +
  // S5:    B[i] = A[i]
  // S6:    for k
  // S7:      B[i] = B[i] +

  // This method distributes the given loop over its body by splitting
  // after every given pivot stmt.
  //
  // NOTE: Pivot stmts that are not in the given loop's body will be ignored.
  //
  // For the above example:
  //   distributeLoop(S1, {S3, S5})
  // will result in:
  // S1:  for i
  // S2:    A[i] = 0
  // S3:    for j
  // S4:      A[i] = A[i] +
  //   :  for i
  // S5:    B[i] = A[i]
  //   :  for i
  // S6:    for k
  // S7:      B[i] = B[i] +
  static std::vector<For*> distributeLoop(
      For* loop,
      const std::unordered_set<Stmt*>& pivots);

  // This method distributes the given loop over every stmt in its body.
  //
  // For the above example:
  //   distributeLoop(S1)
  // will result in:
  // S1:  for i
  // S2:    A[i] = 0
  //   :  for i
  // S3:    for j
  // S4:      A[i] = A[i] +
  //   :  for i
  // S5:    B[i] = A[i]
  //   :  for i
  // S6:    for k
  // S7:      B[i] = B[i] +
  static std::vector<For*> distributeLoop(For* loop);

  // This method distributes the given loop over its body by splitting
  // after every For stmt in its body.
  //
  // For the above example:
  //   distributeLoopOverInnerLoops(S1)
  // will result in:
  // S1:  for i
  // S2:    A[i] = 0
  // S3:    for j
  // S4:      A[i] = A[i] +
  //   :  for i
  // S5:    B[i] = A[i]
  // S6:    for k
  // S7:      B[i] = B[i] +
  static std::vector<For*> distributeLoopOverInnerLoops(For* loop);

  // This method performs loop fusion.
  // For example, consider the following code.
  //
  // S1:  for m
  // S2:    A[m] = 0
  // S3:    for j
  // S4:      A[m] = A[m] +
  // S5:  for n
  // S5:    B[n] = A[n]
  // S6:    for k
  // S7:      B[n] = B[n] +
  //
  // fuseLoops({S1, S5}), will return the following loop:
  // S1:  for m
  // S2:    A[m] = 0
  // S3:    for j
  // S4:      A[m] = A[m] +
  // S5:    B[m] = A[m]
  // S6:    for k
  // S7:      B[m] = B[m] +
  //
  // Loop fusion is done only when all the conditions below are satisfied.
  //  * All the loops have the same parent.
  //  * There are no statements between these loops in their parent body.
  //  * The start bounds are the same for all loops.
  //  * The stop bounds are the same for all loops.
  //  * Fusing the loops does not violate or add any dependencies.
  static bool fuseLoops(const std::vector<For*>& loops, For** fused);

  void reorderAxis(For* a, For* b);

  // Reorder the given list of loops according to the permutation specified.
  // Here permutation[i] represents the location of the loop i in the result.
  //
  // For example, consider the following code:
  //   for p
  //     for q
  //       for r
  //         for s
  //           A[p,q,r,s] =
  //
  // reorder({p, q, r, s}, {2, 3, 0, 1}) will return the list of loops in the
  // following form:
  //    for r
  //      for s
  //        for p
  //          for q
  //            A[p,q,r,s] =
  static std::vector<For*> reorder(
      const std::vector<For*>& loops,
      const std::vector<size_t>& permutation);

  // Returns true if the given loops are perfectly nested, i.e., every loop
  // (except the innermost) should have exactly one statement in its body
  // and that statement must be the next inner loop.
  static bool areLoopsPerfectlyNested(const std::vector<For*>& loops);

  // Returns true if the given loop has a loop-carried dependence.
  static bool hasLoopCarriedDependence(For* loop);

  static void unroll(For* f, Stmt** unrolled);
  static void unroll(For* f);

  static bool normalize(For* f);
  static bool isNormalized(For* f);

  static bool flatten(const std::vector<For*>& f, For** flattened);
  static bool flatten(const std::vector<For*>& f);

  // Compresses the given buffer based on its use in the given Stmts.
  // For example, given the input:
  //
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[i,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[i,j] + A[i, j+1]
  //   }
  // }
  //
  // compressBuffer(A, ...) will compress buffer A from
  // [100, 200] to [1, 200] and modify the code as follows:
  //
  // for (int i = 0; i < 100; ++i) {
  //   for (int j = 0; j < 200; ++j) {
  //     A[0,j] = sin(i*j)
  //   }
  //   for (int j = 0; j < 199; ++j) {
  //     B[i,j] = A[0,j] + A[0, j+1]
  //   }
  // }
  static void compressBuffer(Buf* buf, Stmt* stmt);

  // Get 'num' loops from the loopnest starting at 'f'.
  static std::vector<For*> getLoopStmtsInLoopNest(For* f, size_t num);

  // LoopOptions are propagated to tail.
  void sliceHead(For* f, int factor, For** head, For** tail);
  void sliceHead(For* f, int factor);
  // LoopOptions are propagated to head.
  void sliceTail(For* f, int factor, For** head, For** tail);
  void sliceTail(For* f, int factor);

  void setGPUBlockIndex(For* f, int idx);
  void setGPUThreadIndex(For* f, int idx);

  using AccessResult = std::pair<const Buf*, Stmt*>;
  // Insert a cache for the consumer's usages of the buffer produced in
  // consumer, and redirect reads and writes in the consumer to that cache.
  // Returns a pair of the new cache buffer, and the new rewritten consumer.
  AccessResult cacheAccesses(
      const Buf* producer,
      const std::string& name,
      Stmt* consumer);

  // Insert a temporary computation of statement S in the scope of loop AT.
  // S is assumed to be a Store or a Block containing a Store. Along with the
  // computation itself, this transformation inserts Alloc/Free statements for
  // the temporary buffer used in the computation.
  void computeAt(Stmt* s, For* at);

  // Rfactor a reduction axis into a normal axis.
  //
  // Requirements:
  //  * S is the reduction store
  //  * S is the only statement in the innermost loop
  //  * There is at least two reduction arguments in S
  //  * OUTER_REDUCTION_FOR loop corresponds to the outermost reduction variable
  //  used in the store and all other reduction variables are index variables of
  //  children loops of OUTER_REDUCTION_FOR
  //  * OUTER_REDUCTION_FOR is a perfect loop nest, i.e. it has only loops
  //  corresponding to the other reduction variables and the store, nested into
  //  each other
  //
  // What it does:
  //   * Introduce a new buffer with an extra dimension of a size equal to the
  //   span of the loop OUTER_REDUCTION_FOR (the new buffer is returned via
  //   RFAC_BUF_PTR)
  //   * Insert an initialization store for the new buffer in
  //   OUTER_REDUCTION_FOR before its nested loop
  //   * Replace the reduction store to the original buffer with the reduction
  //   store to the temp buffer, removing the index var of OUTER_REDUCTION_FOR
  //   from reduction arguments
  //   * Insert a final reduction store over the extra dimension of the new
  //   buffer to the original buffer
  //   * Returns TRUE if the transformation succeeded and FALSE otherwise
  //
  // Example:
  // Original IR:
  // S1: for i      # normal axis
  // S2:   X[i] = 0
  // S3:   for j    # reduction axis
  // S4:     for k  # reduction axis
  // S5:       X[i] = ReduceOp(X[i] + Y[i,j,k], reduce_axis={j,k})
  //
  // After RFACTOR(S5, S3)
  // S1: for i               # normal axis
  // S2:   X[i] = 0
  // S3:   for j             # reduction axis for X, normal axis for X_rfac
  //         X_rfac[i,j] = 0
  // S4:     for k           # reduction axis
  //           X_rfac[i,j] = ReduceOp(X_rfac[i,j] + Y[i,j,k], reduce_axis={k})
  //         X[i] = ReduceOp(X[i] + X_rfac[i,j], reduce_axis={j})
  bool rfactor(Stmt* s, For* outer_reduction_for);
  bool rfactor(Stmt* s, For* outer_reduction_for, Buf** rfac_buf_ptr);

  void setBufferMap(
      For* f,
      const std::unordered_map<std::string, const Buf*>& map);

  void eliminateDeadStores();
  void prepareForCodegen();

  // Find the inner-most loops and vectorize them. Currently, this only works
  // for the LLVM backend, when no reductions are involved.
  void vectorizeInnerLoops();

  const std::unordered_set<const Buf*> getInputBufs() const;
  const std::unordered_set<const Buf*> getOutputBufs() const {
    return output_bufs_;
  }

 private:
  void initialize(
      const std::vector<Tensor*>& output_tensors,
      const std::vector<Tensor*>& tensors_to_compute);
  Stmt* insertAllocFree(Stmt* stmt);
  const std::unordered_set<const Buf*> getIntermediateBufs() const;

  Stmt* root_stmt_;

  std::unordered_set<const Buf*> output_bufs_;
};

TORCH_API Stmt* FlattenIndexes(Stmt* s);

// TODO: Revisit this once we decide on how dependencies analysis should look
// like. Maybe we would choose to use a different API and BufUse would be
// removed, or if we decide to keep it we need to properly document its API.
struct BufLoadOrStoreUse {
  Stmt* s;
  bool isStore;
};

/*
 * Returns a map ( Buf -> uses of this Buf), uses are represented as vectors of
 * BufUse elements, which are Stmt* and a bool isStore flag. The order of uses
 * in the vectors reflects the order in which the uses appear in the given
 * statement.
 */
std::unordered_map<const Buf*, std::vector<BufLoadOrStoreUse>>
findLoadOrStoreUses(Stmt* s);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
