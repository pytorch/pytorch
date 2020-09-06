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
  LoopNest(const std::vector<Tensor*>& output_tensors);
  Stmt* root_stmt() const {
    return root_stmt_;
  }

  std::vector<For*> getLoopStmtsFor(Tensor*) const;
  Stmt* getLoopBodyFor(Tensor*) const;
  bool hasLoopBodyFor(Tensor*) const;

  void vectorize(Stmt*);
  void computeInline(Stmt* s);
  void computeInlineWithRandom(Stmt* s);
  void prepareForCodegen();
  void splitWithTail(For* f, int factor, For** outer, For** inner, For** tail);
  void splitWithMask(For* f, int factor, For** outer, For** inner);
  void reorderAxis(For* a, For* b);
  static void unroll(For* f, Stmt** unrolled);
  static void normalize(For* f, For** normalized);

  void setGPUBlockIndex(For* f, int idx);
  void setGPUThreadIndex(For* f, int idx);
  void setBufferMap(
      For* f,
      const std::unordered_map<std::string, const Buf*>& map);

  // Insert a temporary computation of statement S in the scope of loop AT.
  // S is assumed to be a Store or a Block containing a Store. Along with the
  // computation itself, this transformation inserts Alloc/Free statements for
  // the temporary buffer used in the computation.
  void computeAt(Stmt* s, For* at);
  void rfactor(
      const Expr* f,
      const Var* reduction_var,
      Block* insertion_point = nullptr /* optional */);

 private:
  std::vector<Tensor*> findAllNeededTensors(
      const std::vector<Tensor*>& tensors);
  Stmt* lowerToStmt(Tensor* t);
  Stmt* insertAllocFree(Stmt* stmt);

  std::unordered_set<Function*> inlined_functions_;
  std::unordered_set<Function*> inlined_random_functions_;
  std::unordered_map<Tensor*, Stmt*> tensor_to_stmt_;
  std::unordered_map<Stmt*, Tensor*> stmt_to_tensor_;
  Stmt* root_stmt_;

  std::unordered_set<Tensor*> output_tensors_;
  std::unordered_set<Tensor*> intermediate_tensors_;
  std::vector<const Buf*> temp_bufs_;
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
