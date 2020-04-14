#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class Var;
class Tensor;
class Function;
class Stmt;
class For;
class Block;
class Store;
class Range;

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

  void setGPUBlockIndex(For* f, int idx);
  void setGPUThreadIndex(For* f, int idx);

  // Insert a temporary computation of statement S in the scope of loop AT.
  // S is assumed to be a Store or a Block containing a Store. Along with the
  // computation itself, this transformation inserts Alloc/Free statements for
  // the temporary buffer used in the computation.
  void computeAt(Stmt* s, For* at);

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
};

TORCH_API Stmt* FlattenIndexes(Stmt* s);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
