#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {
class HasRand : public IRVisitor {
 public:
  HasRand(Stmt* stmt) : stmt_(stmt) {
    stmt_->accept(this);
  }

  bool has_rand() const {
    return has_rand_;
  }

 private:
  void visit(const Intrinsics* v) override {
    if (v->op_type() == IntrinsicsOp::kRand) {
      has_rand_ = true;
    } else {
      IRVisitor::visit(v);
    }
  }
  Stmt* stmt_;
  bool has_rand_ = false;
};

template <typename Node>
class NodeFinder : public IRVisitor {
 public:
  virtual void visit(const Node* v) override {
    nodes.push_back((Node*)v);
    IRVisitor::visit(v);
  }

  static std::vector<Node*> find(const Stmt* s) {
    NodeFinder<Node> nf;
    s->accept(&nf);
    return nf.nodes;
  }

  static std::vector<Node*> find(const Expr* e) {
    NodeFinder<Node> nf;
    e->accept(&nf);
    return nf.nodes;
  }

  std::vector<Node*> nodes;
};

class VarFinder : public IRVisitor {
 public:
  virtual void visit(const Var* v) override {
    vars_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<const Var*> find(Stmt* s) {
    VarFinder nf;
    s->accept(&nf);
    return nf.vars();
  }

  static std::unordered_set<const Var*> find(const Expr* e) {
    VarFinder nf;
    e->accept(&nf);
    return nf.vars();
  }

  const std::unordered_set<const Var*>& vars() {
    return vars_;
  }

 private:
  std::unordered_set<const Var*> vars_;
};

// Finds all kinds of write operations to the provided Buf.
class WritesToBuf : public IRVisitor {
 public:
  WritesToBuf(const Buf* target) : target_(target) {}

  std::vector<const Stmt*> writes() {
    return writes_;
  }

  static std::vector<const Stmt*> find(Stmt* s, const Buf* b) {
    WritesToBuf finder(b);
    s->accept(&finder);
    return finder.writes();
  }

 private:
  void visit(const Store* v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  void visit(const AtomicAdd* v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  const Buf* target_;
  std::vector<const Stmt*> writes_;
};

// Traverses the IR to determine if a particular Var is modified within it.
class ModifiesVarChecker : public IRVisitor {
 public:
  ModifiesVarChecker(const Var* v) : var_(v) {}

  static bool check(const Stmt* s, const Var* v) {
    ModifiesVarChecker checker(v);
    s->accept(&checker);
    return checker.found();
  }

  bool found() {
    return found_;
  }

 private:
  void visit(const Store* v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(const AtomicAdd* v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(const Let* v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(const For* v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  const Var* var_;
  bool found_{false};
};

// A class that analyzes the given program relevant for Block backend
// It creates a map of multi dim buffers and their flat verions
class CreateBufferMap : public IRVisitor {
 public:
  const std::unordered_map<std::string, const Buf*>& getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(const Store* v) override {
    auto load_node = dynamic_cast<const Load*>(v->value());
    if (load_node) {
      auto t_buf = load_node->buf();
      map_input_to_tensor_bufs_.emplace(t_buf->name_hint(), v->buf());
    } else {
      auto add_node = dynamic_cast<const Add*>(v->value());
      auto mul_node = dynamic_cast<const Mul*>(v->value());
      // This means for now, v->value() can be Add or Mul
      TORCH_INTERNAL_ASSERT((add_node || mul_node));
      map_input_to_tensor_bufs_.emplace(v->buf()->name_hint(), v->buf());
    }
    v->value()->accept(this);
  }
  std::unordered_map<std::string, const Buf*> map_input_to_tensor_bufs_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
