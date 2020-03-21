#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>

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

  void Vectorize(Stmt*);
  void ComputeInline(Stmt* s);
  void ComputeInlineWithRandom(Stmt* s);
  void ApplyInlines();
  void SplitWithTail(For* f, int factor, For** outer, For** inner, For** tail);
  void SplitWithMask(For* f, int factor, For** outer, For** inner);

  void SetGPUBlockIndex(For* f, int idx);
  void SetGPUThreadIndex(For* f, int idx);

 private:
  std::vector<Tensor*> FindAllNeededTensors(
      const std::vector<Tensor*>& tensors);
  Stmt* LowerToStmt(Tensor* t);

  std::unordered_set<Function*> inlined_functions_;
  std::unordered_set<Function*> inlined_random_functions_;
  std::unordered_map<Tensor*, Stmt*> tensor_to_stmt_;
  std::unordered_map<Stmt*, Tensor*> stmt_to_tensor_;
  Stmt* root_stmt_;

  std::unordered_set<Tensor*> output_tensors_;
  std::unordered_set<Tensor*> intermediate_tensors_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
