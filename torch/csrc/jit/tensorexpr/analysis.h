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
  void visit(Intrinsics* v) override {
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
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class NodeFinder : public IRVisitor {
 public:
  void visit(Node* v) override {
    nodes.push_back((Node*)v);
    IRVisitor::visit(v);
  }

  static std::vector<Node*> find(Stmt* s) {
    NodeFinder<Node> nf;
    s->accept(&nf);
    return nf.nodes;
  }

  static std::vector<Node*> find(Expr* e) {
    NodeFinder<Node> nf;
    e->accept(&nf);
    return nf.nodes;
  }

  std::vector<Node*> nodes;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class VarFinder : public IRVisitor {
 public:
  void visit(Var* v) override {
    vars_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<Var*> find(Stmt* s) {
    VarFinder nf;
    s->accept(&nf);
    return nf.vars();
  }

  static std::unordered_set<Var*> find(Expr* e) {
    VarFinder nf;
    e->accept(&nf);
    return nf.vars();
  }

  const std::unordered_set<Var*>& vars() {
    return vars_;
  }

 private:
  std::unordered_set<Var*> vars_;
};

class BufFinder : public IRVisitor {
 public:
  void visit(Buf* v) override {
    bufs_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<Buf*> find(Stmt* s) {
    BufFinder nf;
    s->accept(&nf);
    return nf.bufs();
  }

  static std::unordered_set<Buf*> find(Expr* e) {
    BufFinder nf;
    e->accept(&nf);
    return nf.bufs();
  }

  const std::unordered_set<Buf*>& bufs() {
    return bufs_;
  }

 private:
  std::unordered_set<Buf*> bufs_;
};

// Finds all kinds of write operations to the provided Buf.
class WritesToBuf : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  WritesToBuf(Buf* target) : target_(target) {}

  std::vector<Stmt*> writes() {
    return writes_;
  }

  static std::vector<Stmt*> find(Stmt* s, Buf* b) {
    WritesToBuf finder(b);
    s->accept(&finder);
    return finder.writes();
  }

 private:
  void visit(Store* v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  void visit(AtomicAdd* v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  Buf* target_;
  std::vector<Stmt*> writes_;
};

class StmtsReadingBuf : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  StmtsReadingBuf(Buf* target) : target_(target) {}

  std::vector<Stmt*> reads() {
    return reads_;
  }

  static std::vector<Stmt*> find(Stmt* s, Buf* b) {
    StmtsReadingBuf finder(b);
    s->accept(&finder);
    return finder.reads();
  }

 private:
  bool readsBuffer(Stmt* s) {
    auto loads = NodeFinder<Load>::find(s);
    for (auto l : loads) {
      if (l->buf() == target_) {
        return true;
      }
    }
    return false;
  }

  void visit(Store* v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(Let* v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(Cond* v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(AtomicAdd* v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  Buf* target_;
  std::vector<Stmt*> reads_;
};

// Traverses the IR to determine if a particular Var is modified within it.
class ModifiesVarChecker : public IRVisitor {
 public:
  ModifiesVarChecker(Var* v) : var_(v) {}

  static bool check(Stmt* s, Var* v) {
    ModifiesVarChecker checker(v);
    s->accept(&checker);
    return checker.found();
  }

  bool found() {
    return found_;
  }

 private:
  void visit(Store* v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(AtomicAdd* v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(Let* v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(For* v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  Var* var_;
  bool found_{false};
};

// A class that analyzes the given program relevant for Block backend
// It creates a map of multi dim buffers and their flat verions
class CreateBufferMap : public IRVisitor {
 public:
  const std::unordered_map<std::string, Buf*>& getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(Store* v) override {
    auto load_node = dynamic_cast<Load*>(v->value());
    if (load_node) {
      auto t_buf = load_node->buf();
      map_input_to_tensor_bufs_.emplace(t_buf->name_hint(), v->buf());
    } else {
      auto add_node = dynamic_cast<Add*>(v->value());
      auto mul_node = dynamic_cast<Mul*>(v->value());
      // This means for now, v->value() can be Add or Mul
      TORCH_INTERNAL_ASSERT((add_node || mul_node));
      map_input_to_tensor_bufs_.emplace(v->buf()->name_hint(), v->buf());
    }
    v->value()->accept(this);
  }
  std::unordered_map<std::string, Buf*> map_input_to_tensor_bufs_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
