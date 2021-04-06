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
  LoopNest(const std::vector<Tensor*>& output_tensors);
  LoopNest(
      const std::vector<Tensor*>& output_tensors,
      const std::vector<Tensor*>& tensors_to_compute);

  // A constructor for building a LoopNest from an Stmt and a list of output
  // buffers.
  LoopNest(Stmt* stmt, const std::unordered_set<const Buf*>& output_bufs);

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
  bool hasLoopBodyFor(Tensor*) const;

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

  static void splitWithTail(For* f, int factor);
  static void splitWithTail(
      For* f,
      int factor,
      For** outer,
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

  void reorderAxis(For* a, For* b);

  static void unroll(For* f, Stmt** unrolled);
  static void normalize(For* f, For** normalized);
  static bool flatten(const std::vector<For*>& f, For** flattened);

  // Get 'num' loops from the loopnest starting at 'f'.
  static std::vector<For*> getLoopStmtsInLoopNest(For* f, size_t num);

  // LoopOptions are propagated to tail.
  void sliceHead(For* f, int factor, For** head, For** tail);
  // LoopOptions are propagated to head.
  void sliceTail(For* f, int factor, For** head, For** tail);

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

  void rfactor(
      const Expr* f,
      const Var* reduction_var,
      Block* insertion_point = nullptr /* optional */);

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
