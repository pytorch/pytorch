#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

/* The Registerizer performs scalar replacement by looking for common Stores and
Loads to a single item in a buffer and replacing them with a local temporary
scalar which is cheaper to write.

For example it can replace:

{
  A[0] = 0;
  for (int x = 0; x < 10; x++) {
    A[0] = (A[0]) + x;
  }
}

with:

{
  int A_ = 0;
  for (int x = 0; x < 10; x++) {
    A_ = x + A_;
  }
  A[0] = A_;
}

This is particularly useful on GPUs when parallelizing, since after replacing
loops with metavars we have a lot of accesses like this. */

// Holds analysis information about accesses to a specific range of a
// buffer, including the number of loads and stores and the lowest common
// parent Block.
struct AccessInfo {
  AccessInfo() = default;
  AccessInfo(const Buf* b, const std::vector<const Expr*>& i)
      : buf(b),
        indices(i),
        store_cost(new IntImm(0)),
        load_cost(new IntImm(0)) {}

  void addStore(const Store* s, const Block* p, const Expr* cost) {
    store_cost = IRSimplifier::simplify(new Add(store_cost, cost));
    stores.push_back(s);
    parent = parent ? Block::getSharedParent(parent, p) : p;
    first_usage = first_usage ? first_usage : s;
  }

  void addLoad(
      const Load* l,
      const Block* p,
      const Expr* cost,
      const Stmt* usage) {
    load_cost = IRSimplifier::simplify(new Add(load_cost, cost));
    loads.push_back(l);
    parent = parent ? Block::getSharedParent(parent, p) : p;
    first_usage = first_usage ? first_usage : usage;
  }

  const Buf* buf;
  std::vector<const Expr*> indices;
  const Block* parent{nullptr};

  const Stmt* first_usage{nullptr};

  const Expr* store_cost;
  const Expr* load_cost;

  std::vector<const Store*> stores;
  std::vector<const Load*> loads;

  bool invalid{false};
};

// Walks the IR generating AccessInfo for each access.
class TORCH_API RegisterizerAnalysis : public IRVisitor {
 public:
  RegisterizerAnalysis() : loopCost_(new IntImm(1)) {}
  virtual ~RegisterizerAnalysis() {}

  void visit(const For* v) override;

  void visit(const Block* v) override;

  void visit(const Store* v) override;

  void visit(const Load* v) override;

#define STMT_ON_STACK(Op)                    \
  virtual void visit(const Op* v) override { \
    stmtStack_.push_front(v);                \
    IRVisitor::visit(v);                     \
    stmtStack_.pop_front();                  \
  }

  STMT_ON_STACK(AtomicAdd);
  STMT_ON_STACK(Allocate);
  STMT_ON_STACK(Free);
  STMT_ON_STACK(Let);
  STMT_ON_STACK(Cond);

#undef STMT_ON_STACK

  std::vector<std::shared_ptr<AccessInfo>> getCandidates();

 private:
  std::unordered_map<SimplifierHashType, std::shared_ptr<AccessInfo>>
      candidates_;
  std::unordered_map<const Block*, const Expr*> costByBlock_;
  std::vector<std::shared_ptr<AccessInfo>> encounterOrder_;

  const Expr* loopCost_;

  std::deque<const Stmt*> stmtStack_;
  const Block* enclosingBlock_;
  HashProvider hasher_;
};

// Walks the IR an replaces a single Acccess with a local scalar Var.
class TORCH_API RegisterizerReplacer : public IRMutator {
 public:
  RegisterizerReplacer(std::shared_ptr<AccessInfo> i) : info_(i) {
    var_ = new Var(info_->buf->name_hint() + "_", info_->buf->dtype());
    var_wrapper_ = new Buf(var_, {}, info_->buf->dtype());

    initializer_ = nullptr;
  }

  const Expr* mutate(const Load* v) override;

  Stmt* mutate(const Store* v) override;

  // Finds the Stmt in parent which contains stmt.
  const Stmt* findInsertionPoint(const Stmt* stmt, const Block* parent);

  Stmt* mutate(const Block* v) override;

 private:
  std::shared_ptr<AccessInfo> info_;
  Var* var_;
  Buf* var_wrapper_;
  const Store* initializer_;
  bool dirty_{false};
  bool initializerReady_{true};
};

// Apply scalar replacement to all accesses in s.
// To produce safe code, this must occur after handling parallelized axes and
// atomics.
TORCH_API Stmt* registerize(Stmt* s);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
