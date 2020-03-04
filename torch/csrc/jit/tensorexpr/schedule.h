#pragma once

#include <memory>
#include <unordered_map>

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {
namespace schedule {

TORCH_API Stmt* Vectorize(const Stmt*);

class TORCH_API LoopNest {
 public:
  LoopNest(const std::vector<Tensor*>& output_tensors);
  Stmt* root_stmt() const {
    return root_stmt_;
  }

  std::vector<Stmt*> getLoopStmtsFor(Tensor*) const;
  Stmt* getLoopBodyFor(Tensor*) const;
  std::unordered_map<Tensor*, Stmt*> tensor_to_stmt_;

  void ComputeInline(Stmt* s);
  void ApplyInlines();
  void SplitWithTail(
      Stmt* s,
      int factor,
      Stmt** outer,
      Stmt** inner,
      Stmt** tail);
  void SplitWithMask(Stmt* s, int factor, Stmt** outer, Stmt** inner);

  void SetGPUBlockIndex(Stmt* s, int idx);
  void SetGPUThreadIndex(Stmt* s, int idx);

 private:
  std::vector<Tensor*> FindAllNeededTensors(
      const std::vector<Tensor*>& tensors);
  Stmt* LowerToStmt(Tensor* t);

  std::unordered_set<Function*> inlined_functions_;
  std::unordered_map<Stmt*, Tensor*> stmt_to_tensor_;
  Stmt* root_stmt_;

  std::unordered_set<Tensor*> output_tensors_;
  std::unordered_set<Tensor*> intermediate_tensors_;
};
} // namespace schedule
} // namespace tensorexpr
} // namespace jit
} // namespace torch
