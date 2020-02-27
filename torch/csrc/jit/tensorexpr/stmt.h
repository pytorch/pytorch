#pragma once

#include <string>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
namespace torch {
namespace jit {
namespace tensorexpr {

class Buffer;

// The common base between all statement node.
class Stmt : public KernelScopedObject {
 public:
  Stmt() {}
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  virtual Stmt* accept_mutator(IRMutator* mutator) = 0;
};

template <class Op>
class StmtNode : public Stmt {
 public:
  using StmtNodeBase = StmtNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Stmt* accept_mutator(IRMutator* mutator) override;
  StmtNode() {}
};

template <class Op>
Stmt* StmtNode<Op>::accept_mutator(IRMutator* mutator) {
  StmtNode* this_mutable = const_cast<StmtNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

// Concrete Stmt classes
class LetStmt : public StmtNode<LetStmt> {
 public:
  const Var* var() const {
    return var_;
  }

  const Expr* value() const {
    return value_;
  }

  Stmt* body() const {
    return body_;
  }

  static Stmt* make(const VarHandle& var, const ExprHandle& value, Stmt* body) {
    return new LetStmt(var.node(), value.node(), body);
  }

  LetStmt(const Var* var, const Expr* value, Stmt* body)
      : var_(var), value_(value), body_(body) {}

 private:
  const Var* var_;
  const Expr* value_;
  Stmt* body_;
};

class Block : public StmtNode<Block> {
 public:
  static Stmt* make(const std::vector<Stmt*>& stmts) {
    std::vector<Stmt*> valid_stmts;
    for (size_t i = 0; i < stmts.size(); i++) {
      if (!stmts[i]) {
        continue;
      }
      valid_stmts.push_back(stmts[i]);
    }
    if (valid_stmts.empty()) {
      return nullptr;
    }
    return new Block(valid_stmts);
  }
  int nstmts() const {
    return stmts_.size();
  }
  Stmt* stmt(int index) const {
    return stmts_[index];
  }

 private:
  explicit Block(const std::vector<Stmt*>& stmts) : stmts_(stmts) {}
  std::vector<Stmt*> stmts_;
};

class TORCH_API Store : public StmtNode<Store> {
 public:
  const Var* base_handle() const {
    return base_handle_;
  }
  const Expr* index() const {
    return index_;
  }
  const Expr* value() const {
    return value_;
  }
  const Expr* mask() const {
    return mask_;
  }

  static Stmt* make(
      const Buffer& buffer,
      const ExprHandle& index,
      const ExprHandle& value,
      const ExprHandle& mask) {
    return new Store(buffer, index.node(), value.node(), mask.node());
  }

  static Stmt* make(
      const VarHandle& base_handle,
      const ExprHandle& index,
      const ExprHandle& value,
      const ExprHandle& mask) {
    return new Store(base_handle.node(), index.node(), value.node(), mask.node());
  }

  static Stmt* make(
      const VarHandle& base_handle,
      const ExprHandle& index,
      const ExprHandle& value) {
    return new Store(base_handle.node(), index.node(), value.node(), ExprHandle(1).node());
  }

  // TODO: merge this with Load.
  Store(
      const Buffer& buffer,
      const Expr* index,
      const Expr* value,
      const Expr* mask);

  Store(
      const Var* base_handle,
      const Expr* index,
      const Expr* value,
      const Expr* mask)
      : base_handle_(base_handle), index_(index), value_(value), mask_(mask) {
    CHECK_EQ(base_handle_->dtype(), kHandle);
    CHECK_EQ(index->dtype().lanes(), mask->dtype().lanes());
    CHECK_EQ(index->dtype().lanes(), value->dtype().lanes());
    CHECK_EQ(index->dtype().scalar_type(), ScalarType::Int);
  }
 private:

  const Var* base_handle_;
  const Expr* index_;
  const Expr* value_;
  const Expr* mask_;
};

// Allocate a buffer of given shapes and dtypes and bind it with the given
// buffer var. The life span is at most through the current program, until it is
// explicitly freed. An unfreed memory is likely considered an error.
class Allocate : public StmtNode<Allocate> {
 public:
  static Stmt* make(
      const VarHandle& buffer_var,
      Dtype dtype,
      const std::vector<ExprHandle>& dims) {
    std::vector<const Expr*> dims_nodes(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      dims_nodes[i] = dims[i].node();
    }
    return new Allocate(buffer_var.node(), dtype, dims_nodes);
  }

  const Var* buffer_var() const {
    return buffer_var_;
  }

  Dtype dtype() const {
    return dtype_;
  }

  const std::vector<const Expr*>& dims() const {
    return dims_;
  }

  Allocate(const Var* buffer_var, Dtype dtype, const std::vector<const Expr*>& dims)
      : buffer_var_(buffer_var), dtype_(dtype), dims_(dims) {}

 private:
  const Var* buffer_var_;
  Dtype dtype_;
  std::vector<const Expr*> dims_;
  // TODO: add memory types.
};

// Free the specific buffer. It is an error.
class Free : public StmtNode<Free> {
 public:
  static Stmt* make(const VarHandle& buffer_var) {
    return new Free(buffer_var.node());
  }

  const Var* buffer_var() const {
    return buffer_var_;
  }

  Free(const Var* buffer_var) : buffer_var_(buffer_var) {}

 private:
  const Var* buffer_var_;
};

class Cond : public StmtNode<Cond> {
 public:
  static Stmt* make(
      const ExprHandle& condition,
      Stmt* true_stmt,
      Stmt* false_stmt) {
    return new Cond(condition.node(), true_stmt, false_stmt);
  }

  const Expr* condition() const {
    return condition_;
  }

  Stmt* true_stmt() const {
    return true_stmt_;
  }

  Stmt* false_stmt() const {
    return false_stmt_;
  }

  Cond(const Expr* condition, Stmt* true_stmt, Stmt* false_stmt)
      : condition_(condition), true_stmt_(true_stmt), false_stmt_(false_stmt) {}

 private:
  const Expr* condition_;
  Stmt* true_stmt_;
  Stmt* false_stmt_;
};

class LoopOptions {
 public:
  // GPU Block Index
  bool is_gpu_block_index() const {
    return gpu_block_index_ != -1;
  }

  bool gpu_block_index() const {
    return gpu_block_index_;
  }

  std::string gpu_block_index_str() const {
    DCHECK(is_gpu_block_index());
    static const char* kBlockIndexNames[] = {
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "blockIdx.w",
    };
    DCHECK(gpu_block_index_ >= 0 && gpu_block_index_ < 4);
    return kBlockIndexNames[gpu_block_index_];
  }

  void set_gpu_block_index(int index) {
    if (is_gpu_thread_index()) {
      throw std::runtime_error("Cannot set both gpu block and thread index");
    }
    if (is_gpu_block_index() && gpu_block_index() != index) {
      throw std::runtime_error(
          "Cannot set a previously set block index: " +
          std::to_string(gpu_block_index()) + " vs " + std::to_string(index));
    }
    gpu_block_index_ = index;
  }

  // GPU Thread Index
  bool is_gpu_thread_index() const {
    return gpu_thread_index() != -1;
  }

  int gpu_thread_index() const {
    return gpu_thread_index_;
  }

  std::string gpu_thread_index_str() const {
    DCHECK(is_gpu_thread_index());
    static const char* kThreadIndexNames[] = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z", "threadIdx.w"};
    DCHECK(gpu_thread_index_ >= 0 && gpu_thread_index_ < 4);
    return kThreadIndexNames[gpu_thread_index_];
  }

  void set_gpu_thread_index(int index) {
    if (is_gpu_block_index()) {
      throw std::runtime_error("Cannot set both gpu thread and block index");
    }
    if (is_gpu_thread_index() && gpu_thread_index() != index) {
      throw std::runtime_error(
          "Cannot set a previously set thread index: " +
          std::to_string(gpu_thread_index()) + " vs " + std::to_string(index));
    }
    gpu_thread_index_ = index;
  }

  std::string ToString() const {
    std::ostringstream oss;
    if (is_gpu_block_index()) {
      oss << gpu_block_index_str();
    } else if (is_gpu_thread_index()) {
      oss << gpu_thread_index_str();
    }
    return oss.str();
  }

 private:
  int gpu_block_index_ = -1;
  int gpu_thread_index_ = -1;
};

class For : public StmtNode<For> {
 public:
  const Var* var() const {
    return var_;
  }
  const Expr* start() const {
    return start_;
  }
  const Expr* stop() const {
    return stop_;
  }
  Stmt* body() const {
    return body_;
  }
  static Stmt* make(
      const VarHandle& var,
      const ExprHandle& start,
      const ExprHandle& stop,
      Stmt* body) {
    if (!body) {
      return nullptr;
    }
    return new For(var.node(), start.node(), stop.node(), body);
  }
  static Stmt* make(
      const VarHandle& var,
      const ExprHandle& start,
      const ExprHandle& stop,
      Stmt* body,
      const LoopOptions& loop_options) {
    if (!body) {
      return nullptr;
    }
    return new For(var.node(), start.node(), stop.node(), body, loop_options);
  }
  const LoopOptions loop_options() const {
    return loop_options_;
  }

  For(const Var* var, const Expr* start, const Expr* stop, Stmt* body)
      : var_(var), start_(start), stop_(stop), body_(body) {
          CHECK(var && start && stop && body);
      }

  For(const Var* var,
      const Expr* start,
      const Expr* stop,
      Stmt* body,
      const LoopOptions& loop_options)
      : var_(var),
        start_(start),
        stop_(stop),
        body_(body),
        loop_options_(loop_options) {
          CHECK(var && start && stop && body);
        }

 private:
  const Var* var_;
  const Expr* start_;
  const Expr* stop_;
  Stmt* body_;
  LoopOptions loop_options_;
};
} // namespace tensorexpr
} // namespace jit
} // namespace torch
