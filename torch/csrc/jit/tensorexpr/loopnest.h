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

  void vectorize(Stmt*);
  void computeInline(Stmt* s);
  void computeInlineWithRandom(Stmt* s);
  void prepareForCodegen();
  void splitWithTail(For* f, int factor, For** outer, For** inner, For** tail);
  void splitWithMask(For* f, int factor, For** outer, For** inner);

  void setGPUBlockIndex(For* f, int idx);
  void setGPUThreadIndex(For* f, int idx);

 private:
  std::vector<Tensor*> findAllNeededTensors(
      const std::vector<Tensor*>& tensors);
  Stmt* lowerToStmt(Tensor* t);

  std::unordered_set<Function*> inlined_functions_;
  std::unordered_set<Function*> inlined_random_functions_;
  std::unordered_map<Tensor*, Stmt*> tensor_to_stmt_;
  std::unordered_map<Stmt*, Tensor*> stmt_to_tensor_;
  Stmt* root_stmt_;

  std::unordered_set<Tensor*> output_tensors_;
  std::unordered_set<Tensor*> intermediate_tensors_;
};

// represent a range [start, stop)
class Range {
 public:
  Range() {}
  Range(const Expr* start, const Expr* stop) : start_(start), stop_(stop) {}
  const Expr* start() const {
    return start_;
  }
  const Expr* stop() const {
    return stop_;
  }

 private:
  const Expr* start_;
  const Expr* stop_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
