#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

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

  static std::vector<Node*> find(Stmt* s) {
    NodeFinder<Node> nf;
    s->accept(&nf);
    return nf.nodes;
  }

  std::vector<Node*> nodes;
};
} // namespace tensorexpr
} // namespace jit
} // namespace torch
