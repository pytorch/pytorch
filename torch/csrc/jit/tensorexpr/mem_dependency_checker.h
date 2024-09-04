#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <utility>
#include <vector>

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch::jit::tensorexpr::analysis {

enum class AccessType {
  Input,
  Output,
  Load,
  Store,
  Call,
  AtomicAdd,
  Alloc,
  Free
};
const char* AccessToString(AccessType a);

class AccessInfo;
using DependencySet = std::unordered_set<std::shared_ptr<AccessInfo>>;

/* AccessInfo
 *
 * Represents a single bounded memory access to a buffer, for instance a Load or
 * a Store. Holds information relating to the specific access and links to
 * connected accesses in the dependency graph.
 */
class TORCH_API AccessInfo {
 public:
  AccessInfo(
      size_t id,
      AccessType type,
      StmtPtr stmt,
      VarPtr var,
      IndexBounds bounds)
      : id_(id),
        type_(type),
        stmt_(std::move(stmt)),
        expr_(nullptr),
        var_(std::move(var)),
        bounds_(std::move(bounds)) {}

  AccessInfo(
      size_t id,
      AccessType type,
      ExprPtr expr,
      StmtPtr stmt,
      VarPtr var,
      IndexBounds bounds)
      : id_(id),
        type_(type),
        stmt_(std::move(stmt)),
        expr_(std::move(expr)),
        var_(std::move(var)),
        bounds_(std::move(bounds)) {}

  // Id is a unique int representing the order this access occurred in the
  // graph.
  size_t id() const {
    return id_;
  }

  // The type of the access (Load, Store, etc).
  AccessType type() const {
    return type_;
  }

  // The enclosing Stmt this access represents. E.g. if this is a Store then
  // Stmt is the Store itself, while if the access is caused by an Expr, this is
  // the most immediate parent Stmt.
  StmtPtr stmt() const {
    return stmt_;
  }

  // If the access is represented by an Expr (such as Load or Call) then this is
  // it, otherwise it's nullptr.
  ExprPtr expr() const {
    return expr_;
  }

  // The Var representing the underlying Buffer.
  VarPtr var() const {
    return var_;
  }

  // A vector of Bounds representing the start and end expression for each
  // dimension.
  IndexBounds& bounds() {
    return bounds_;
  }

  // Each access that this depends upon,
  // eg. if this is a Load, then it contains every Store that immediately
  // contributes to a load of the bounds.
  // or: if this is a Store, it contains all reads on the RHS of the Store.
  const std::map<size_t, std::shared_ptr<AccessInfo>>& dependencies() const {
    return dependencies_;
  }

  // Each access that depends on this one.
  // ie. this access is present in the dependencies map of all accesses that are
  // dependent.
  std::map<size_t, std::shared_ptr<AccessInfo>> dependents() const {
    std::map<size_t, std::shared_ptr<AccessInfo>> res;
    for (const auto& kv : dependents_) {
      res.emplace(kv.first, kv.second.lock());
    }
    return res;
  }

  // Returns the symbolic expression of the indices of this access.
  std::vector<ExprPtr> getIndices() const;

  // Establishes a dependency or dependent relationship with another access.
  void addDependency(const std::shared_ptr<AccessInfo>& write);
  void addDependent(const std::shared_ptr<AccessInfo>& read);

  // helper for checking dependencies.
  bool hasDependency(const std::shared_ptr<AccessInfo>& info) const;

  // Returns the set of all nodes that are direct (immediate) dependencies of
  // this access.
  DependencySet getDirectDependencies();
  // likewise, returns all nodes that directly depend on this one.
  DependencySet getDirectDependents();

  // Returns the full list of all nodes in the graph that this access depends
  // on, and all nodes they depend on, and so forth, back to the inputs.
  DependencySet getIndirectDependencies();
  // likewise, returns the full list of all nodes that depend on this node, and
  // all nodes that depend on those nodes and so on down to the outputs.
  DependencySet getIndirectDependents();

  // Does this access represent a read of memory (Load, ReduceOp, Call, etc).
  bool isRead() const;
  // Does this access represent a write of memory (Store, etc).
  bool isWrite() const;

  // Helpers for dumping accesses in various formats.
  void print() const;
  void dumpDOT(std::ostream& os) const;
  const char* AccessTypeColour() const;

 private:
  size_t id_;
  AccessType type_;
  StmtPtr stmt_;
  ExprPtr expr_;
  VarPtr var_;
  IndexBounds bounds_;

  // Yes these should be sorted.
  std::map<size_t, std::shared_ptr<AccessInfo>> dependencies_;
  std::map<size_t, std::weak_ptr<AccessInfo>> dependents_;
};

using VarBoundMap = std::unordered_map<VarPtr, Bound>;

/* MemDependencyChecker analyses a IR fragment and builds a dependency graph of
 * accesses contained within.
 *
 * It's possible to retrieve the entire graph in node-object form, or can be
 * used as an oracle for answering dependency questions. e.g:
 *
 *  analyzer.hasIndirectDependency(BufA, BufB); or,
 *  analyzer.hasDirectDependency(LoadA, StoreB);
 */
class TORCH_API MemDependencyChecker : public IRVisitor {
  struct Scope;

 public:
  MemDependencyChecker();
  MemDependencyChecker(
      const std::unordered_set<BufPtr>& inputs,
      const std::unordered_set<BufPtr>& outputs);
  MemDependencyChecker(
      const std::vector<BufHandle>& inputs,
      const std::vector<BufHandle>& outputs);

  ~MemDependencyChecker() override = default;

  // Whether or not to allow loop execution order to influence dependency
  // calculation. If the loop may later be parallelized you don't want this.
  bool allowLoopExecutionOrderAnalysis(bool allow = true);

  // Dependency Checking API.
  // The goal is to have enough overloads here so you don't really have to think
  // about it.

  // Returns true if any read in A has a direct dependence on a write in B.
  bool dependsDirectly(const StmtPtr& A, const StmtPtr& B);
  bool dependsDirectly(const ExprPtr& A, const StmtPtr& B);

  // Returns true of the output depends directly on a write contained in B.
  bool dependsDirectly(const BufPtr& output, const StmtPtr& B);

  // Returns true if a read in A depends directly on the provided input.
  bool dependsDirectly(const StmtPtr& A, const BufPtr& input);
  bool dependsDirectly(const ExprPtr& A, const BufPtr& input);

  // Outputs/inputs cannot depend directly.

  // Returns true if the access A has B as an immediate dependency.
  bool dependsDirectly(
      const std::shared_ptr<AccessInfo>& A,
      const std::shared_ptr<AccessInfo>& B);

  // Returns true if any read in A has an ancestor write contained in B.
  bool dependsIndirectly(const StmtPtr& A, const StmtPtr& B);
  bool dependsIndirectly(const ExprPtr& A, const StmtPtr& B);

  // Returns true of the output depends indirectly on a write contained in B.
  bool dependsIndirectly(const BufPtr& output, const StmtPtr& B);

  // Returns true if a read in A depends indirectly on the provided input.
  bool dependsIndirectly(const StmtPtr& A, const BufPtr& input);
  bool dependsIndirectly(const ExprPtr& A, const BufPtr& input);

  // returns true if the output uses any load of the input.
  bool dependsIndirectly(const BufPtr& output, const BufPtr& input);

  // Returns true if the access A has a dependency chain to access B.
  bool dependsIndirectly(
      const std::shared_ptr<AccessInfo>& A,
      const std::shared_ptr<AccessInfo>& B);

  // Returns the AccessInfo
  std::shared_ptr<AccessInfo> accessFor(const StmtPtr& A) const;
  std::shared_ptr<AccessInfo> accessFor(const ExprPtr& A) const;

  // Returns all AccessInfos.
  std::unordered_set<std::shared_ptr<AccessInfo>> accessesWithin(
      const StmtPtr& A) const;
  // TODO: this will return only the AccessInfo for A. It's included for
  // completeness but be aware it wont return accesses used in the computation
  // of A.
  std::unordered_set<std::shared_ptr<AccessInfo>> accessesWithin(
      const ExprPtr& A) const;

  // Accesses relating to input and output buffers.
  std::shared_ptr<AccessInfo> input(const BufPtr& B) const;
  std::shared_ptr<AccessInfo> output(const BufPtr& B) const;

  // Returns the full history of reads and writes.
  const std::vector<std::shared_ptr<AccessInfo>>& getHistory() const;

  // Dumps the dependency graph in DOT format.
  void dumpDAG(const std::string& filename) const;

 private:
  // Node visitors.
  void visit(const StorePtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const CondPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const CompareSelectPtr& v) override;
  void visit(const BlockPtr& v) override;
  void visit(const LetPtr& v) override;
  void visit(const AtomicAddPtr& v) override;
  void visit(const AllocatePtr& v) override;
  void visit(const FreePtr& v) override;

  using BoundRelationship = std::pair<IndexBounds, std::shared_ptr<AccessInfo>>;

  // An internal struct holding the accesses found within a scope Block.
  struct Scope {
    Scope(BlockPtr b, std::shared_ptr<Scope> p)
        : block(std::move(b)), parent(std::move(p)) {}

    BlockPtr block;
    std::shared_ptr<Scope> parent;

    std::unordered_map<VarPtr, Bound> shadowedVarBounds;
    std::unordered_set<VarPtr> localVars;

    std::vector<std::shared_ptr<AccessInfo>> accesses_;

    std::unordered_map<VarPtr, std::list<BoundRelationship>> openWrites_;
  };
  std::shared_ptr<Scope> currentScope_;

  bool allowExecutionOrderAnalysis_{false};

  std::unordered_multimap<StmtPtr, std::shared_ptr<AccessInfo>> stmtToAccess_;
  std::unordered_multimap<ExprPtr, std::shared_ptr<AccessInfo>> exprToAccess_;
  std::unordered_map<StmtPtr, std::vector<std::shared_ptr<AccessInfo>>>
      scopeToAccesses_;

  VarBoundMap knownVarBounds_;

  // Finds all accesses that are reads within the scope of v.
  template <typename StmtOrExprPtr>
  DependencySet getAllReadsWithin(const StmtOrExprPtr& v) {
    DependencySet reads;
    auto insertAllReads = [&](const auto& nodes) {
      for (const auto& l : nodes) {
        auto bound = exprToAccess_.equal_range(l);
        for (auto it = bound.first; it != bound.second; ++it) {
          if (it->second->isRead()) {
            reads.insert(it->second);
          }
        }
      }
    };

    // Look for and insert accesses belonging to all nodes that act like
    // reads.
    insertAllReads(NodeFinder<Load>::find(v));
    insertAllReads(NodeFinder<ReduceOp>::find(v));

    return reads;
  }

  // Finds all accesses that are writes within the scope of v.
  // Writes cannot occur in Exprs, so this is a little simpler.
  DependencySet getAllWritesWithin(const StmtPtr& v) {
    DependencySet writes;

    // writes just Store currently.
    auto stores = NodeFinder<Store>::find(v);
    for (const auto& s : stores) {
      auto bound = stmtToAccess_.equal_range(s);
      for (auto it = bound.first; it != bound.second; ++it) {
        if (it->second->isWrite()) {
          writes.insert(it->second);
        }
      }
    }
    return writes;
  }

  // Templated helpers to work on either Exprs or Stmts.
  template <typename StmtOrExprPtr>
  bool dependsDirectlyHelper(const StmtOrExprPtr& A, const StmtPtr& B) {
    auto aReads = getAllReadsWithin(A);
    auto bWrites = getAllWritesWithin(B);

    for (auto& read : aReads) {
      for (auto& depPair : read->dependencies()) {
        if (bWrites.count(depPair.second) != 0) {
          return true;
        }
      }
    }

    return false;
  }

  template <typename StmtOrExprPtr>
  bool dependsIndirectlyHelper(StmtOrExprPtr A, const StmtPtr& B) {
    auto aReads = getAllReadsWithin(A);
    auto bWrites = getAllWritesWithin(B);

    auto aDeps = getAllWriteDependencies(aReads);

    for (auto& dependency : aDeps) {
      if (bWrites.count(dependency) != 0) {
        return true;
      }
    }

    return false;
  }

  DependencySet getAllWriteDependencies(const DependencySet& products);

  // Maps for inputs and outputs, since they aren't present directly in the IR.
  std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>> inputs_;
  std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>> outputs_;
  std::unordered_map<VarPtr, std::shared_ptr<AccessInfo>> intermediates_;

  // Inserts accesses for Buf's: specifically for inputs and outputs.
  void insertBuffers(
      std::unordered_map<BufPtr, std::shared_ptr<AccessInfo>>& bufs,
      AccessType type);

  // Update the write history with a new write, adding dependencies and closing
  // any overlapped writes (if possible).
  void updateWriteHistory(
      std::list<BoundRelationship>& writeHistory,
      const std::shared_ptr<AccessInfo>& info,
      size_t latestAccessToClose,
      bool closeOverlapped = true,
      bool insert = true);

  // Merge a child scope into a parent scope, adding dependencies for open
  // writes in the parent to accesses in the child.
  void mergeScope(
      const std::shared_ptr<Scope>& child,
      const std::shared_ptr<Scope>& parent,
      bool closeOverlapped = true);

  // Binds symbolic vars in indices with the low and high bound for those vars.
  std::vector<Bound> getIndicesBounds(const std::vector<ExprPtr>& indices);

  size_t nextAccess_{0};
  StmtPtr lastStmt_{nullptr};
};

} // namespace torch::jit::tensorexpr::analysis
