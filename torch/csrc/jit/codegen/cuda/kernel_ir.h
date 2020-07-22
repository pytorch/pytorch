
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

// TODO: remove these
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>

#include <c10/util/Optional.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace kir {

// TODO: Fill out TensorIndex, which is a list of Ints used to directly index a
// TensorView. It is not the flattened index, which needs to be computed using
// stride information.
class TORCH_CUDA_API TensorIndex : public Val {
 public:
  TensorIndex(const TensorView* view, std::vector<Val*> indices);

  std::vector<Val*>::size_type nDims() const {
    return indices_.size();
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  Val* index(int i) const;

  const std::vector<Val*>& indices() const {
    return indices_;
  }

  const TensorView* view() const {
    return view_;
  }

 private:
  const TensorView* view_ = nullptr;
  std::vector<Val*> indices_;
};

class TORCH_CUDA_API BroadcastOp : public Expr {
 public:
  BroadcastOp(Val* _out, Val* _in);

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

// Allocate is a lower level Node that describes a buffer of memory that
// is required as an intermediate within a kernel.  The extent is the expression
// of the size of the buffer that is generated from the TensorView that
// describes the output of an operation.
//
// TODO: The components of Allocate like Type and Name could be separated from
// the the assocated TensorView.  Perhaps that is more appropriate?
class TORCH_CUDA_API Allocate : public Expr {
 public:
  explicit Allocate(
      Val* _buffer,
      MemoryType _memory_type = MemoryType::Local,
      Val* _size = nullptr);

  Val* buffer() const {
    return buffer_;
  }

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  Val* size() const {
    return size_;
  }

  DataType buffer_type() const {
    return buffer_->getDataType().value();
  }

 private:
  Val* buffer_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;
  Val* size_ = nullptr;
};

class TORCH_CUDA_API Scope {
 public:
  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  void push_back(Expr* e) {
    exprs_.push_back(e);
  }

  void insert(std::vector<Expr*>::iterator it, Expr* expr) {
    exprs_.insert(it, expr);
  }

  void erase(std::vector<Expr*>::iterator it) {
    exprs_.erase(it);
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

  auto& operator[](size_t i) {
    return exprs_[i];
  }

  auto& operator[](size_t i) const {
    return exprs_[i];
  }

  // Insert expr before ref
  void insert_before(Expr* ref, Expr* expr);

  // Insert expr after ref
  void insert_after(Expr* ref, Expr* expr);

  bool contains(Expr* expr) const;

  void erase(Expr* ref);

  void clear();

 private:
  std::vector<Expr*> exprs_;
};

// ForLoop provides scoping around an int iterator from 0 to range. Exprs placed
// in its body are considered inside the scope of the for loop. In the future
// the implementation should look quite different so that we can do proper
// dependency annalysis like in Fusion.
class TORCH_CUDA_API ForLoop : public Expr {
 public:
  ForLoop(
      Val* _index,
      IterDomain* _iter_domain,
      const std::vector<Expr*>& _body = {},
      Expr* parent_scope = nullptr);

  Val* index() const {
    return index_;
  }

  IterDomain* iter_domain() const {
    return iter_domain_;
  }

  Scope& body() {
    return body_;
  }

  const Scope& constBody() const {
    return body_;
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

 private:
  Val* const index_ = nullptr;
  IterDomain* const iter_domain_;
  Scope body_;
  Expr* parent_scope_ = nullptr;
};

// IfThenElse provides scoping for an boolean operator. Exprs placed in its body
// are considered inside the scope of the if statement. In the future the
// implementation should look quite different so that we can do proper
// dependency annalysis like in Fusion.
class TORCH_CUDA_API IfThenElse : public Expr {
 public:
  IfThenElse(
      Bool* _cond,
      const std::vector<Expr*>& _if_body = {},
      const std::vector<Expr*>& _else_body = {},
      Expr* _parent_scope = nullptr);

  Bool* cond() const {
    return cond_;
  }

  const Scope& constBody() const {
    return body_;
  }

  const Scope& constElseBody() const {
    return else_body_;
  }

  Scope& body() {
    return body_;
  }

  Scope& elseBody() {
    return else_body_;
  }

  bool hasElse() const {
    return !else_body_.empty();
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

 private:
  Bool* const cond_ = nullptr;
  Scope body_;
  Scope else_body_;
  Expr* parent_scope_ = nullptr;
};

// Grid reduction operation, this node is used only after lowering a fusion to
// explicitly mark a grid reduction and the buffer allocation needed to do it.
// This node provides FusionExecutor the information it needs to allocate the
// reduction and sync buffers.
class TORCH_CUDA_API GridReduction : public Expr {
 public:
  explicit GridReduction(ReductionOp* reduction_op);
  GridReduction(
      ReductionOp* reduction_op,
      Allocate* reduction_buffer,
      Allocate* sync_buffer);

  ReductionOp* reduction_op() const {
    return reduction_op_;
  }

  Allocate* reduction_buffer() const {
    return reduction_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

 private:
  ReductionOp* reduction_op_ = nullptr;
  Allocate* reduction_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
};

} // namespace kir
} // namespace fuser
} // namespace jit
} // namespace torch
