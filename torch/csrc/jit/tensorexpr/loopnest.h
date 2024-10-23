#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

namespace torch::jit::tensorexpr {

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
      const std::vector<Tensor>& output_tensors,
      const std::vector<Tensor>& tensors_to_compute);

  // A convenience constructor for the case when all tensors are output tensors
  LoopNest(const std::vector<Tensor>& output_tensors);

  // A constructor for building a LoopNest from an Stmt and a list of output
  // buffers.
  LoopNest(StmtPtr stmt, std::unordered_set<BufPtr> output_bufs);

  // A constructor for building a LoopNest from another loopnest. It clones the
  // other loopnest's stmt.
  LoopNest(const LoopNest& other);

  StmtPtr root_stmt() const {
    return root_stmt_;
  }

  std::vector<ForPtr> getLoopStmtsFor(const Tensor&) const;
  std::vector<ForPtr> getLoopStmtsFor(const BufPtr&) const;
  std::vector<ForPtr> getLoopStmtsFor(StmtPtr) const;
  StmtPtr getLoopBodyFor(const Tensor&) const;
  StmtPtr getLoopBodyFor(BufPtr) const;

  // Returns the For stmt indexed by 'indices' in the 'root' For stmt.
  //'indices' indicates the path to the returned loop from 'root' in AST, e.g.,
  //
  // root: for(int i...){
  // j_loop: for (int j...){
  // k1_loop:  for (int k1...){
  //            A[i, j, k1] = ....
  //          }
  //          B[i, j] = ...
  // k2_loop:  for (int k2...){
  //            A[i, j, k2] = ...
  //          }
  //        }
  //      }
  //
  // the path from 'root' to 'j_loop' is [0]
  // the path from 'root' to 'k1_loop' is [0, 0]
  // the path from 'root' to 'k2_loop' is [0, 2]
  ForPtr getLoopAt(ForPtr root, const std::vector<int>& indices) const;

  // Returns the For stmt that is immediately enclosing the given stmt.
  static ForPtr getParentLoop(const StmtPtr& st);

  // Returns the list of For stmts corresponding to the loopnest that is
  // enclosing the given stmt.
  static std::vector<ForPtr> getEnclosingLoopNest(const StmtPtr& st);

  // Returns a list of all Stmts that write to the given buf.
  std::vector<StmtPtr> getAllWritesToBuf(BufPtr) const;

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
  std::vector<ForPtr> getAllInnermostLoopsWritingToBuf(BufPtr) const;

  // Returns a list of For loopnests which contain a Stmt that writes to
  // the given buf. Each loopnest here is a vector For loops.
  // For the above example:
  //   getAllLoopNestsWritingToBuf(a) => {{i1,j1}, {i2,j2,k2}, {i2,j3}}
  std::vector<std::vector<ForPtr>> getAllLoopNestsWritingToBuf(BufPtr) const;

  StmtPtr simplify();

  // Sanitize variables and buffer names.
  // The pass assigns predefined names for loop index variables
  // (i,j,k,l,m,n,o,p,i1,j1,k1,...) and ensures these names are not conflicting
  // anywhere. It also removes duplicates from other Buf nad Var names as well
  // as replaces illegal characters in them with underscores.
  //
  // Note: since it's currently technically possible to use the same variable
  // as index in two different loops, this transformation finds such cases and
  // introduces new variables to avoid duplication.
  static StmtPtr sanitizeNames(StmtPtr s);

  bool computeInline(const StmtPtr& s);
  bool computeInline(const BufPtr& b);
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

  // Splits the given loop into 2 nested loops with the given factor as the
  // inner loop bound. If the factor does not evenly divide the loop bound,
  // then the remaining iterations are extracted into a tail loop that is
  // added after the given loop.
  //
  // For example, consider the following code:
  //   for (int i = 0; i < 100; ++i) {
  //     A[i] =
  //   }
  //
  // splitWithTail(i, 8, ...) will result in:
  //   for (int i_outer = 0; i_outer < 12; ++i_outer) {
  //     for (int i_inner = 0; i_inner < 8; ++i_inner) {
  //       A[i_outer * 8 + i_inner] =
  //     }
  //   }
  //   for (int i_tail = 0; i_tail < 4; ++i_tail) {
  //     A[i_tail + 96] =
  //   }
  //
  // The given loop will be transformed to the outer loop after splitting.
  // So, the pointer to the input loop should be valid after splitting and
  // will point to the outer loop. The `inner` and `tail` parameters will be
  // set to point to the inner and tail loops that are generated.
  static void splitWithTail(
      const ForPtr& f,
      int factor,
      ForPtr* inner,
      ForPtr* tail);
  // A convenience wrapper when the caller does not need to access the
  // split loops.
  static void splitWithTail(const ForPtr& f, int factor);

  // Splits the given loop into 2 nested loops with the given factor as the
  // inner loop bound. If the factor does not evenly divide the loop bound,
  // then a conditional is inserted into the body to handle the remaining
  // iterations appropriately.
  //
  // For example, consider the following code:
  //   for (int i = 0; i < 100; ++i) {
  //     A[i] =
  //   }
  //
  // splitWithMask(i, 8, ...) will result in:
  //   for (int i_outer = 0; i_outer < 13; ++i_outer) {
  //     for (int i_inner = 0; i_inner < 8; ++i_inner) {
  //       if (i_outer * 8 + i_inner < 100) {
  //         A[i_outer * 8 + i_inner] =
  //       }
  //     }
  //   }
  //
  // The given loop will be transformed to the outer loop after splitting.
  // So, the pointer to the input loop should be valid after splitting and
  // will point to the outer loop. The `inner` parameter will be set to point
  // to the inner loop that is generated.
  static void splitWithMask(const ForPtr& f, int factor, ForPtr* inner);
  // A convenience wrapper when the caller does not need to access the
  // split loops.
  static void splitWithMask(const ForPtr& f, int factor);

  // The following methods support loop distribution.
  // For example, consider the following code. This will be used to
  // demonstrate the methods below.
  //
  // S0:  for m
  // S1:    for i
  // S2:      A[i] = 0
  // S3:      for j
  // S4:        A[i] = A[i] +
  // S5:      B[i] = A[i]
  // S6:      for k
  // S7:        B[i] = B[i] +

  // This method distributes the given loop over its body by splitting
  // after every given pivot stmt.
  //
  // NOTE: Pivot stmts that are not in the given loop's body will be ignored.
  //
  // For the above example:
  //   distributeLoop(S1, {S3, S5})
  // will result in:
  // S0:  for m
  // S1:    for i
  // S2:      A[i] = 0
  // S3:      for j
  // S4:        A[i] = A[i] +
  //   :    for i
  // S5:      B[i] = A[i]
  //   :    for i
  // S6:      for k
  // S7:        B[i] = B[i] +
  static std::vector<ForPtr> distributeLoop(
      const ForPtr& loop,
      const std::unordered_set<StmtPtr>& pivots);

  // This method distributes the given loop over every stmt in its body.
  //
  // For the above example:
  //   distributeLoop(S1)
  // will result in:
  // S0:  for m
  // S1:    for i
  // S2:      A[i] = 0
  //   :    for i
  // S3:      for j
  // S4:        A[i] = A[i] +
  //   :    for i
  // S5:      B[i] = A[i]
  //   :    for i
  // S6:      for k
  // S7:        B[i] = B[i] +
  static std::vector<ForPtr> distributeLoop(const ForPtr& loop);
  // Same as above, but also distribute parent loops.
  // Returns the result of distributing the outermost loop.
  //
  // For the above example:
  //   distributeLoopAndParents(S1) will result in:
  // S0:  for m
  // S1:    for i
  // S2:      A[i] = 0
  //   :  for m
  //   :    for i
  // S3:      for j
  // S4:        A[i] = A[i] +
  //   :  for m
  //   :    for i
  // S5:      B[i] = A[i]
  //   :  for m
  //   :    for i
  // S6:      for k
  // S7:        B[i] = B[i] +
  static std::vector<ForPtr> distributeLoopAndParents(const ForPtr& loop);

  // This method distributes the given loop over its body by splitting
  // after every For stmt in its body.
  //
  // For the above example:
  //   distributeLoopOverInnerLoops(S1)
  // will result in:
  // S0:  for m
  // S1:    for i
  // S2:      A[i] = 0
  // S3:      for j
  // S4:        A[i] = A[i] +
  //   :    for i
  // S5:      B[i] = A[i]
  // S6:      for k
  // S7:        B[i] = B[i] +
  static std::vector<ForPtr> distributeLoopOverInnerLoops(const ForPtr& loop);
  // Same as above, but also distribute parent loops.
  // Returns the result of distributing the outermost loop.
  //
  // For the above example:
  //   distributeLoopAndParentsOverInnerLoops(S1)
  // will result in:
  // S0:  for m
  // S1:    for i
  // S2:      A[i] = 0
  // S3:      for j
  // S4:        A[i] = A[i] +
  //   :  for m
  //   :    for i
  // S5:      B[i] = A[i]
  // S6:      for k
  // S7:        B[i] = B[i] +
  static std::vector<ForPtr> distributeLoopAndParentsOverInnerLoops(
      const ForPtr& loop);

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
  // This transformation is unsafe as it simply add all loops into the body of
  // the first loop for fusion without correctness checks.
  //
  // Below are the two requirements to apply unsafeFuseLoops:
  //  * All the loops have the same parent.
  //  * There are no statements between these loops in their parent body.
  static bool unsafeFuseLoops(const std::vector<ForPtr>& loops, ForPtr* fused);

  // Loop fusion is done only when all the conditions below are satisfied.
  //  * All the loops have the same parent.
  //  * There are no statements between these loops in their parent body.
  //  * The start bounds are the same for all loops.
  //  * The stop bounds are the same for all loops.
  //  * Fusing the loops does not violate or add any dependencies.
  static bool fuseLoops(const std::vector<ForPtr>& loops, ForPtr* fused);

  static void reorderAxis(const ForPtr& a, const ForPtr& b);

  // Reorder the given list of loops according to the permutation specified.
  // Here `permutation[i]` represents the position of the loop in the input
  // which will end up at position `i` after the reorder.
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
  static std::vector<ForPtr> reorder(
      const std::vector<ForPtr>& loops,
      const std::vector<size_t>& permutation);

  // Tile takes a 2d domain (x, y) and splits it into small rectangular blocks
  // each with shape (x_factor, y_factor). The traversal over the domain turns
  // into an outer iteration over the blocks and an inner traversal over all
  // points in the block.
  // Note that if x dim % x_factor or y dim % y_factor does not equal to 0, the
  // loop body will generate corresponding tailing loops.
  // The transformation is in-place and returns 'xtail'.
  //
  // For example, consider the following code:
  //   for i: [0, 64)
  //     for j: [0, 64)
  //       for k: [0, 32)
  //         A[i, j] = B[i, k] + C[j, k]
  //
  // tile(i, j, 4, 8) will transform "i" for-stmt into the following nested
  // loop:
  //   for i_outer: [0, 16)
  //     for j_outer: [0, 8)
  //       for i_inner: [0, 4)
  //         for j_inner: [0, 8)
  //           for k: [0, 32)
  //             A[i_outer * 4 + i_inner, j_outer * 8 + j_inner] =
  //             B[i_outer * 4 + i_inner, k] + C[j_outer * 8 + j_inner, k]
  //
  // tile(i, j, 4, 9) will transform "i" for-stmt into the following nested
  // loop:
  //   for i_outer: [0, 16)
  //     for j_outer: [0, 7)
  //       for i_inner: [0, 4)
  //         for j_inner: [0, 9)
  //           for k: (0, 32)
  //             A[i_outer * 4 + i_inner, j_outer * 9 + j_inner] =
  //             B[i_outer * 4 + i_inner, k] + C[j_outer * 9 + j_inner, k]
  //     for j_tail: [0, 1)
  //       for i_inner: [0, 4)
  //         for k: (0, 32)
  //           A[i_outer * 4 + i_inner, 7 * 9 + j_tail] =
  //           B[i_outer * 4 + i_inner, k] + C[7 * 9 + j_tail, k]
  ForPtr tile(const ForPtr& x, const ForPtr& y, int x_factor, int y_factor);

  // Returns true if the given loops are perfectly nested, i.e., every loop
  // (except the innermost) should have exactly one statement in its body
  // and that statement must be the next inner loop.
  static bool areLoopsPerfectlyNested(const std::vector<ForPtr>& loops);

  // Returns true if the given loop has a loop-carried dependence.
  static bool hasLoopCarriedDependence(const ForPtr& loop);

  // Unrolls all the iterations of the given loop.
  // Requires that the loop bounds are constant.
  static void fullUnroll(const ForPtr& f, StmtPtr* unrolled);
  static void fullUnroll(const ForPtr& f);

  // Unrolls the given loop for the specified factor.
  // This does not require constant bounds for the loop being unrolled.
  static void unroll(const ForPtr& f, int factor, ForPtr* tail);
  static void unroll(const ForPtr& f, int factor);

  static bool normalize(const ForPtr& f);
  static bool isNormalized(const ForPtr& f);

  static bool flatten(const std::vector<ForPtr>& f, ForPtr* flattened);
  static bool flatten(const std::vector<ForPtr>& f);

  // Compresses the given buffer based on its use in the given Stmts.
  //
  // NOTE: This API assumes that there are no accesses to the given buffer
  // outside the given statement. So, this should be called with the entire
  // kernel statement to avoid incorrect buffer compressions.
  //
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
  static void compressBuffer(const BufPtr& buf, const StmtPtr& stmt);

  // Compresses all buffers in the given statement.
  //
  // NOTE: This API assumes that there are no accesses to buffers outside
  // the given statement. So, this should be called with the entire
  // kernel statement to avoid incorrect buffer compressions.
  //
  // TODO: Add an IR verifier check to detect invalidly compressed buffers.
  static void compressAllBuffers(const StmtPtr& stmt);

  // Get 'num' loops from the loopnest starting at 'f'.
  static std::vector<ForPtr> getLoopStmtsInLoopNest(
      const ForPtr& f,
      size_t num);

  // LoopOptions are propagated to tail.
  static void sliceHead(
      const ForPtr& f,
      int factor,
      ForPtr* head,
      ForPtr* tail);
  static void sliceHead(const ForPtr& f, int factor);
  // LoopOptions are propagated to head.
  static void sliceTail(
      const ForPtr& f,
      int factor,
      ForPtr* head,
      ForPtr* tail);
  static void sliceTail(const ForPtr& f, int factor);

  using AccessResult = std::pair<BufPtr, StmtPtr>;
  // Insert a cache for the consumer's usages of the buffer produced in
  // consumer, and redirect reads and writes in the consumer to that cache.
  // Returns a pair of the new cache buffer, and the new rewritten consumer.
  static AccessResult cacheAccesses(
      const BufPtr& producer,
      const std::string& name,
      const StmtPtr& consumer);

  // Insert a temporary computation of statement S in the scope of loop AT.
  // S is assumed to be a Store or a Block containing a Store. Along with the
  // computation itself, this transformation inserts Alloc/Free statements for
  // the temporary buffer used in the computation.
  static void computeAt(const StmtPtr& s, const ForPtr& at);

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
  static bool rfactor(const StmtPtr& s, const ForPtr& outer_reduction_for);
  static bool rfactor(
      const StmtPtr& s,
      const ForPtr& outer_reduction_for,
      BufPtr* rfac_buf_ptr);

  // Vectorize the given loop. This method requires that the given loop
  // does not perform a reduction.
  // It returns true if vectorization is successful and false otherwise.
  static bool vectorize(const ForPtr&);

  // Find the inner-most loops and vectorize them. Currently, this only works
  // for the LLVM backend, when no reductions are involved.
  void vectorizeInnerLoops();

  void eliminateDeadStores();

  void prepareForCodegen();

  const std::unordered_set<BufPtr> getInputBufs() const;
  const std::unordered_set<BufPtr> getOutputBufs() const {
    return output_bufs_;
  }
  std::vector<BufPtr> getIntermediateBufs() const;

  // Finds which is the outer For between a and b for loops. If neither of the 2
  // Fors is an ancestor of the other, it returns nullptr.
  static ForPtr findOuterFor(ForPtr a, ForPtr b);

 private:
  void initialize(
      const std::vector<Tensor>& output_tensors,
      const std::vector<Tensor>& tensors_to_compute);

  StmtPtr root_stmt_;

  std::unordered_set<BufPtr> output_bufs_;
};

TORCH_API StmtPtr FlattenIndexes(const StmtPtr& s);

// TODO: Revisit this once we decide on how dependencies analysis should look
// like. Maybe we would choose to use a different API and BufUse would be
// removed, or if we decide to keep it we need to properly document its API.
struct BufLoadOrStoreUse {
  StmtPtr s;
  bool isStore;
};

/*
 * Returns a map ( Buf -> uses of this Buf), uses are represented as vectors of
 * BufUse elements, which are StmtPtr and a bool isStore flag. The order of uses
 * in the vectors reflects the order in which the uses appear in the given
 * statement.
 */
std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findLoadOrStoreUses(
    const StmtPtr& s);

// replaces all invalid characters with underscore
TORCH_API std::string sanitizeName(const std::string& input_name);

} // namespace torch::jit::tensorexpr
