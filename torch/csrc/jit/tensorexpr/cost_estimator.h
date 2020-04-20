#pragma once

#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {

struct SubExprInfo {
  SubExprInfo() {
    throw std::logic_error("blah");
  }
  SubExprInfo(size_t c, const Expr* e, const Stmt* p)
      : count(c), cost(e), parent(p) {}
  SubExprInfo(size_t c, size_t o, const Stmt* p)
      : count(c), cost(getImmediateByType(kInt, o)), parent(p) {}
  size_t count{0};
  const Expr* cost{0};
  const Stmt* parent{nullptr};
};

// Fixed costs per Op type for estimating the runtime cost of a subexpr.
struct OpCostDictionary {
  int IMMEDIATE_COST{0};
  int BINARY_OP_COST{1};
  int COMPARE_OP_COST{1};
  int CAST_OP_COST{1};
  int VAR_REF_COST{0};
  int LOAD_OP_COST{10};
  int STORE_OP_COST{15};
  int CALL_OP_COST{5}; // TODO: different funcs obv have different costs.
  float ALLOC_COST{0.01};
  int FREE_OP_COST{2};
};

class TORCH_API CostEstimator : public IRVisitor {
  OpCostDictionary dict_;
  HashProvider hasher_;
  std::unordered_map<SimplifierHashType, SubExprInfo> exprInfo_;
  const Stmt* lastBlock_{nullptr};

  template <typename Op>
  bool canUpdateExisting(const Op* e, SimplifierHashType hash) {
    auto existing = exprInfo_.find(hash);
    if (existing == exprInfo_.end()) {
      return false;
    }

    existing->second.count++;
    // assume cost is the same.

    // Set the shared parent.
    if (lastBlock_ != existing->second.parent) {
      existing->second.parent =
          getSharedParent(lastBlock_, existing->second.parent);
    }

    return true;
  }

  const Stmt* getSharedParent(const Stmt* a, const Stmt* b);

  template <typename Op>
  void scanBinaryOp(const BinaryOpNode<Op>* v) {
    v->lhs()->accept(this);
    v->rhs()->accept(this);

    auto hash = hasher_.hash(v);
    if (canUpdateExisting(v, hash)) {
      return;
    }

    auto lhsInfo = exprInfo_[hasher_.hash(v->lhs())];
    auto rhsInfo = exprInfo_[hasher_.hash(v->rhs())];
    Expr* cost = new Add(
        getImmediateByType(kInt, dict_.BINARY_OP_COST),
        new Add(lhsInfo.cost, rhsInfo.cost));
    exprInfo_.emplace(
        hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
  }

 public:
  CostEstimator() = default;
  CostEstimator(OpCostDictionary& dict) : dict_(dict) {}

  const Expr* estimateCost(const Expr* e) {
    auto it = exprInfo_.find(hasher_.hash(e));
    if (it == exprInfo_.end()) {
      e->accept(this);
      it = exprInfo_.find(hasher_.hash(e));
    }
    return it->second.cost;
  }

  const Expr* estimateCost(Stmt* s) {
    auto it = exprInfo_.find(hasher_.hash(s));
    if (it == exprInfo_.end()) {
      s->accept(this);
      it = exprInfo_.find(hasher_.hash(s));
    }
    return it->second.cost;
  }

  SubExprInfo getInfo(const Expr* e) {
    auto it = exprInfo_.find(hasher_.hash(e));
    if (it == exprInfo_.end()) {
      e->accept(this);
      it = exprInfo_.find(hasher_.hash(e));
    }
    return it->second;
  }

  SubExprInfo getInfo(Stmt* s) {
    auto it = exprInfo_.find(hasher_.hash(s));
    if (it == exprInfo_.end()) {
      s->accept(this);
      it = exprInfo_.find(hasher_.hash(s));
    }
    return it->second;
  }

  void clear() {
    exprInfo_.clear();
    lastBlock_ = nullptr;
    // don't clear hasher state.
  }

  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const Mod* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const And* v) override;
  void visit(const Or* v) override;
  void visit(const Xor* v) override;
  void visit(const Lshift* v) override;
  void visit(const Rshift* v) override;
  void visit(const CompareSelect* v) override;

// NOLINTNEXTLINE
#define IMM_VISIT(Type, Name)                                               \
  void visit(const Name##Imm* v) override {                                 \
    exprInfo_.emplace(                                                      \
        hasher_.hash(v), SubExprInfo(1, dict_.IMMEDIATE_COST, lastBlock_)); \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

  void visit(const Cast* v) override;
  void visit(const Var* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const Block* v) override;
  void visit(const For* v) override;
  void visit(const Broadcast* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const BaseCallNode* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Cond* v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
