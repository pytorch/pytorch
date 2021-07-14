#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace registerizer {

/* The Registerizer performs scalar replacement by looking for common Stores and
Loads to a single item in a buffer and replacing them with a local temporary
scalar which is cheaper to write.

For example it can replace:

{
  A[0] = 0;
  for(const auto x : c10::irange(10)) {
    A[0] = (A[0]) + x;
  }
}

with:

{
  int A_ = 0;
  for(const auto x : c10::irange(10)) {
    A_ = x + A_;
  }
  A[0] = A_;
}

This is particularly useful on GPUs when parallelizing, since after replacing
loops with metavars we have a lot of accesses like this. */

class Scope;

/*  Holds analysis information about accesses to a specific range of a
 buffer, including the number of loads and stores and the lowest common parent
 Block.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class AccessInfo {
 public:
  AccessInfo() = default;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AccessInfo(
      SimplifierHashType h,
      const Buf* b,
      std::vector<const Expr*> i,
      size_t accessOrder)
      : hash_(h),
        buf_(b),
        indices_(std::move(i)),
        store_cost_(new IntImm(0)),
        load_cost_(new IntImm(0)),
        accessOrder_(accessOrder) {}

  // Adds a Store to this access, which is in the provided scope.
  void addStore(const Store* store, const std::shared_ptr<Scope>& scope);

  // Adds a Load to this access, which occurs in the usage Stmt in the provided
  // scope.
  void addLoad(
      const Load* load,
      const std::shared_ptr<Scope>& scope,
      const Stmt* usage);

  // Merge another AccessInfo into this one.
  void merge(const std::shared_ptr<AccessInfo>& other);

  // Returns true if the other AccessInfo's bounds may overlap this one.
  bool overlaps(const std::shared_ptr<AccessInfo>& other);

  // Returns true if the indices of this access depend on the provided Var.
  bool dependsOnVar(const Var* v);

  // Clone this AccessInfo, and set this as the new accesses' hiddenAccess.
  static std::shared_ptr<AccessInfo> cloneWithHiddenInfo(
      const std::shared_ptr<AccessInfo>& orig);

  // print for debugging.
  void print() const;

  SimplifierHashType hash() const {
    return hash_;
  }

  const Buf* buf() const {
    return buf_;
  }

  const std::vector<const Expr*>& indices() const {
    return indices_;
  }

  const Block* block() const {
    return block_;
  }

  void setEnclosingBlock(const Block* b) {
    block_ = b;
  }

  const Stmt* first_usage() const {
    return first_usage_;
  }
  const Stmt* last_usage() const {
    return last_usage_;
  }

  void setUsageMarks(const Stmt* first, const Stmt* last) {
    first_usage_ = first;
    last_usage_ = last;
  }

  bool firstUsageOverlapped() const {
    return firstUsageOverlapped_;
  }

  const Expr* store_cost() const {
    return store_cost_;
  }

  const Expr* load_cost() const {
    return load_cost_;
  }

  const std::vector<const Store*>& stores() const {
    return stores_;
  }

  const std::vector<const Load*>& loads() const {
    return loads_;
  }

  void hoistCosts(const Expr* extent) {
    store_cost_ = IRSimplifier::simplify(new Mul(store_cost_, extent));
    load_cost_ = IRSimplifier::simplify(new Mul(load_cost_, extent));
  }

  size_t conditionId() const {
    return conditionId_;
  }

  void setConditionId(size_t c) {
    conditionId_ = c;
  }

  size_t accessOrder() const {
    return accessOrder_;
  }

  std::shared_ptr<AccessInfo> hiddenAccess() const {
    return hiddenAccess_;
  }

  // Holds state relating to the scalar variable we will insert to replace some
  // number of loads and stores.
  struct ScalarReplacement {
    Var* var{nullptr};
    Buf* var_wrapper{nullptr};
    Let* initializer{nullptr};
  };

  ScalarReplacement& replacement() {
    return replacement_;
  }

 private:
  SimplifierHashType hash_;
  const Buf* buf_;
  std::vector<const Expr*> indices_;
  const Block* block_{nullptr};

  const Stmt* first_usage_{nullptr};
  const Stmt* last_usage_{nullptr};

  // Whether or not this access is overlapped in the first Stmt it appears. This
  // means we cannot use it's first Store as the initializer.
  bool firstUsageOverlapped_{false};

  // The cost in real ops that this access represents, to enable
  // filtering accesses that wont save any loads or stores.
  const Expr* store_cost_;
  const Expr* load_cost_;

  // The actual Stores and Loads which represent this access.
  // Be careful with these, any mutator will invalidate these pointers.
  std::vector<const Store*> stores_;
  std::vector<const Load*> loads_;

  // An identifier representing the conditional block, if any, this access
  // depends on.
  size_t conditionId_{0};

  // An identifier representing the order this access was first encountered, for
  // sorting returned results.
  size_t accessOrder_{0};

  // Sometimes when traversing the tree we need to record what would happen if
  // we hoisted an access, but sometimes it doesn't work out. This lets us
  // "undo" some mutation and return to the internal hidden AccessInfo.
  // It will be removed after any further additions to this AccessInfo.
  std::shared_ptr<AccessInfo> hiddenAccess_;

  ScalarReplacement replacement_;
};

using AccessHashMap =
    std::unordered_map<SimplifierHashType, std::shared_ptr<AccessInfo>>;

// Represents a scope block and holds all accesses contained within it.
class Scope {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Scope(const Block* b, std::shared_ptr<Scope> parent, size_t conditionId = 0)
      : block_(b), parent_(std::move(parent)), conditionId_(conditionId) {}

  AccessHashMap& getAccessMapByBuf(const Buf* b);

  std::unordered_map<const Buf*, AccessHashMap>& openAccesses() {
    return openAccesses_;
  }

  std::vector<std::shared_ptr<AccessInfo>>& closedAccesses() {
    return closedAccesses_;
  }

  const Block* block() const {
    return block_;
  }

  std::shared_ptr<Scope> parent() const {
    return parent_;
  }

  size_t conditionId() const {
    return conditionId_;
  }

  const std::unordered_set<const Var*>& localVars() const {
    return localVars_;
  }
  void addLocalVar(const Var* v) {
    localVars_.insert(v);
  }

  void closeAccess(const std::shared_ptr<AccessInfo>& info);

  void filterClosed();

 private:
  // Map of map to access, narrowing by Buf then by hash(Buf+Indices).
  // This allows us to find a candidate access easily, and also check for
  // overlap with other accesses to the same buf. Buf ->
  //    Hash ->
  //        Access
  std::unordered_map<const Buf*, AccessHashMap> openAccesses_;
  std::vector<std::shared_ptr<AccessInfo>> closedAccesses_;

  // The Block object this scope represents.
  const Block* block_;

  // The enclosing scope object.
  std::shared_ptr<Scope> parent_;

  // An identifier representing the condition block this scope depends on.
  size_t conditionId_;

  // A set of variables local to this scope (e.g. loop vars).
  std::unordered_set<const Var*> localVars_;
};

/* Analyzes the graph and collects accesses to the same symbolic tensor element
 * which can be replaced by a single local scalar.
 *
 * This works by recursively walking the tree in postfix order, building sets of
 * accesses to the same symbolic element by scope and then merging lower scopes
 * into their enclosing scope.
 *
 * It is safe to move two accesses of the same Tensor element to a local scalar
 * Var if between all usages of the element there are no other Loads or Stores
 * that may refer to it. In the comments I refer to this as overlapping the
 * access, or "cutting" the existing AccessInfo. In the case where a candidate
 * for registerization is cut, it may be possible to finalize the access early
 * by writing it back to the Tensor and then create a new scalar variable after
 * the overlapping access is complete. We will attempt to do this when it saves
 * memory accesses.
 *
 * There are a few cases that make this more challenging:
 *
 *  - For: Loops change the number of real usages of a buffer by the loop
 * extent, but only if we can pull the definition and finalization of the scalar
 * variable out of the loop block.
 *
 * - Cond: Conditions complicate lifting scalars out of internal scopes.
 * Generally we cannot lift an access outside of a conditional scope unless
 * there is already a reference to that same access at the higher scope, since
 * we don't know if the condition was guarding an array access not safe at the
 * higher scope. In the comments I refer to this as the condition "hiding" the
 * access, and the outer access "unhiding" it.
 *
 * - IfThenElse: Same situation as Cond, except since IfThenElse is an Expr
 * rather than a Stmt we cannot insert the scalar definition or finalizer
 * within the conditional scope. Acccesses inside an IfThenElse can be safely
 * combined with external accesses but cannot exist completely within.
 *
 * - Let: Accesses dependent on local variables via Let Stmts, or loop vars,
 * cannot be raised outside of the scope of the dependent var.
 */
class TORCH_API RegisterizerAnalysis : public IRVisitor {
 public:
  RegisterizerAnalysis()
      : currentScope_(std::make_shared<Scope>(nullptr, nullptr, 0)) {}
  ~RegisterizerAnalysis() override = default;

  void visit(const For* v) override;

  void visit(const Cond* v) override;

  void visit(const Block* v) override;

  void visit(const Store* v) override;

  void visit(const Load* v) override;

  void visit(const IfThenElse* v) override;

  void visit(const Let* v) override;

#define STMT_ON_STACK(Op)                    \
  virtual void visit(const Op* v) override { \
    stmtStack_.push_front(v);                \
    IRVisitor::visit(v);                     \
    stmtStack_.pop_front();                  \
  }

  STMT_ON_STACK(AtomicAdd);
  STMT_ON_STACK(Allocate);
  STMT_ON_STACK(Free);

#undef STMT_ON_STACK

  std::vector<std::shared_ptr<AccessInfo>> getCandidates();

 private:
  void mergeCurrentScopeIntoParent();
  void mergeHiddenScope(bool allowClosed);
  void closeAccessIntoScope(
      const std::shared_ptr<AccessInfo>& info,
      const std::shared_ptr<Scope>& scope);

  std::unordered_set<size_t> exprConditionals_;

  // A stack of enclosing Stmts for tracking the usage Stmt of Loads.
  std::deque<const Stmt*> stmtStack_;

  // The current scope being analyzed.
  std::shared_ptr<Scope> currentScope_;

  HashProvider hasher_;

  size_t conditionId_{0};
  size_t accessOrder_{0};
};

/* Replaces each registerizable access with a Scalar variable, including
 * definition, initializer and finalizer.
 */
class TORCH_API RegisterizerReplacer : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  RegisterizerReplacer(std::vector<std::shared_ptr<AccessInfo>>& vec)
      : infoSet_(vec) {
    buildReplacements();
  }

  const Expr* mutate(const Load* v) override;

  Stmt* mutate(const Store* v) override;

  Stmt* mutate(const Block* v) override;

 private:
  struct ReplacerScope {
    std::unordered_map<const Stmt*, std::deque<std::shared_ptr<AccessInfo>>>
        initializerPoints_;
    std::unordered_map<const Stmt*, std::deque<std::shared_ptr<AccessInfo>>>
        finalizePoints_;
  };

  // Creates the various ReplacerScope objects and builds internal maps.
  void buildReplacements();

  // State relating to the accesses yet to be replaced.
  std::vector<std::shared_ptr<AccessInfo>>& infoSet_;
  std::unordered_map<const Store*, std::shared_ptr<AccessInfo>> storeToAccess_;
  std::unordered_map<const Load*, std::shared_ptr<AccessInfo>> loadToAccess_;
  std::unordered_map<const Block*, ReplacerScope> parentToAccesses_;

  // Holds the set of Stores that should be pulled into an initializer, so they
  // can be eliminated.
  std::set<const Store*> eliminatedIntializers_;

  // Tracks the number of times we've seen each buffer, so we can name the
  // scalar Vars appropriately.
  std::unordered_map<const Buf*, unsigned int> bufferAccessCounts_;
  unsigned int getBufferAccessCount(const Buf* b) {
    return ++bufferAccessCounts_[b];
  }
};
} // namespace registerizer

// Apply scalar replacement to all accesses in s.
// To produce safe code, this must occur after handling parallelized axes and
// atomics.
TORCH_API Stmt* registerize(Stmt* s);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
