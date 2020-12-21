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

  // A constructor for building a LoopNest from a pre-baked Stmt and meta-info
  // TODO: Nuke intermediate_bufs_ and possibly buf_initializers from here if
  // they can be deduced.
  LoopNest(
      Stmt* stmt,
      const std::unordered_set<const Buf*>& output_bufs,
      const std::unordered_set<const Buf*>& intermediate_bufs,
      const std::unordered_map<const Buf*, const Expr*>& buf_initializers)
      : root_stmt_(stmt),
        output_bufs_(output_bufs),
        intermediate_bufs_(intermediate_bufs),
        buf_initializers_(buf_initializers) {}

  Stmt* root_stmt() const {
    return root_stmt_;
  }

  std::vector<For*> getLoopStmtsFor(Tensor*) const;
  std::vector<For*> getLoopStmtsFor(Stmt*) const;
  Stmt* getLoopBodyFor(Tensor*) const;
  bool hasLoopBodyFor(Tensor*) const;

  static void vectorize(For*);

  bool computeInline(Stmt* s);
  bool computeInline(const Buf* b);
  void inlineIntermediateBufs(bool inline_output_buffers);

  static void splitWithTail(For* f, int factor);
  static void splitWithTail(
      For* f,
      int factor,
      For** outer,
      For** inner,
      For** tail);

  static void splitWithMask(For* f, int factor);
  static void splitWithMask(For* f, int factor, For** outer, For** inner);

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

  const std::unordered_set<const Buf*> getInputBufs() {
    return input_bufs_;
  }
  const std::unordered_set<const Buf*> getOutputBufs() {
    return output_bufs_;
  }

 private:
  std::vector<Tensor*> findAllNeededTensors(
      const std::vector<Tensor*>& tensors);
  Stmt* lowerToStmt(Tensor* t);
  Stmt* insertAllocFree(Stmt* stmt);

  Stmt* root_stmt_;

  std::unordered_set<const Buf*> input_bufs_;
  std::unordered_set<const Buf*> output_bufs_;
  std::unordered_set<const Buf*> intermediate_bufs_;
  // Holds the initializer Expr of buffers that have been initialized.
  std::unordered_map<const Buf*, const Expr*> buf_initializers_;
};

TORCH_API Stmt* FlattenIndexes(Stmt* s);

// TODO: Revisit this once we decide on how dependencies analysis should look
// like. Maybe we would choose to use a different API and BufUse would be
// removed, or if we decide to keep it we need to properly document its API.
struct BufUse {
  Stmt* s;
  bool isStore;
};

/*
 * Returns a map ( Buf -> uses of this Buf), uses are represented as vectors of
 * BufUse elements, which are Stmt* and a bool isStore flag. The order of uses
 * in the vectors reflects the order in which the uses appear in the given
 * statement.
 */
std::unordered_map<const Buf*, std::vector<BufUse>> findUses(Stmt* s);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
