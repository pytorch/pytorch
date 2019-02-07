#pragma once

#include <torch/csrc/jit/alias_info.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/utils/alias_tracker.h>

namespace torch {
namespace jit {

/**
 * Alias analysis pass.
 *
 * This pass produces an AliasDb that contains aliasing and mutation
 * information about the graph. Users can use this information to determine
 * whether mutations to the graph are safe, i.e. they don't reorder/change
 * nodes in a way that affects output.
 *
 * Every value with a mutable type (Tensors, Lists, Tuples, etc.) will be
 * associated with one or more "alias sets". If two values share an alias set,
 * that means they may alias, implying that a mutation to one value cannot be
 * reordered past a use of the other. Only reordering two reads of an alias set
 * is considered safe.
 *
 * There is a special alias set called the "wildcard set", which indicates that
 * we're not sure what this value may alias. To be conservative, we consider
 * the wildcard alias set as potentially aliasing any value.
 */

class AliasDb {
 public:
  TORCH_API explicit AliasDb(std::shared_ptr<Graph> graph);
  TORCH_API ~AliasDb();

  // There are limitations to what effects the alias analysis can track. Two
  // kinds of nodes may have untracked effects:
  // 1. Nodes that write to a value that may alias the graph inputs (since
  //    the inputs can be used outside the graph).
  // 2. Nodes that write to something in the wildcard set.
  //
  // These nodes are considered not safe to eliminate or mutate under any
  // circumstances.
  bool hasUntrackedEffects(Node* n) const;

  // Get all the values that `n` writes to.
  // NOTE: this only returns values directly written to, not aliases thereof
  //
  // if `recurseBlocks` is true, gather writes on the nodes in `n`s sub-blocks
  ValueSet getWrites(Node* n, bool recurseBlocks = false) const;

  // Do any values in group `a` potentially share a memory location with any
  // value in group `b`?
  bool mayAlias(const ValueSet& a, const ValueSet& b) const;

  // Do any nodes write to an alias set inputed/outputed by `n`?
  bool hasWriters(const Node* n) const;

  // Move 'n' (already in the graph) after 'movePoint' in the topological order.
  //
  // Tries to preserve value dependencies, so other nodes might be moved. We
  // make two gurantees about the postcondition of the node list:
  //   - `n` is directly after `movePoint`.
  //   - only nodes between `n` and `movePoint` have been moved.
  //
  // Returns `false` if it's impossible to move `n` after `MovePoint` without
  // violating dependencies, otherwise executes the move and returns `true`
  bool moveAfterTopologicallyValid(Node* n, Node* movePoint);
  bool moveBeforeTopologicallyValid(Node* n, Node* movePoint);

  bool couldMoveAfterTopologically(Node* n, Node* movePoint);
  bool couldMoveBeforeTopologically(Node* n, Node* movePoint);

  // For debugging: print alias db state to stdout
  TORCH_API void dump() const;

 private:
  // Helper for topologically-safe node moves.
  class WorkingSet;
  enum class MoveSide { BEFORE, AFTER };
  bool tryMove(Node* toMove, Node* movePoint, MoveSide moveSide, bool dryRun);
  void move(Node* toMove, Node* movePoint, MoveSide moveSide);
  bool isBeforeOrAfter(const Node* n, MoveSide moveSide) const;

  ValueSet getWrites(Block* b) const;
  void getWritesImpl(Block* b, ValueSet& ret, bool recurseBlocks = false) const;
  void getWritesImpl(Node* n, ValueSet& ret, bool recurseBlocks = false) const;

  // Get all the values that `n` reads from.
  // if `recurseBlocks` is true, gather reads on the nodes in `n`s sub-blocks
  ValueSet getReads(Node* n, bool recurseBlocks = false) const;

  void getReadsImpl(Node* n, ValueSet& ret, bool recurseBlocks = false) const;
  // Does `n` write to any alias sets?
  bool hasWrites(Node* n) const;

  // Does `n` use or write to any wildcard aliases?
  bool hasWildcard(const Node* n) const;
  // Returns nullopt if there are no wildcard nodes
  c10::optional<const Node*> getLastWildcard() const;

  // Does `n` write to a value that may alias one of the graph inputs?
  bool writesToInputAlias(Node* n) const;

  void analyze(const std::shared_ptr<Graph>& graph);
  void analyze(Block* block);
  void analyze(Node* node);
  void analyzeImpl(Node* node);

  void analyzeIf(Node* node);
  void analyzeLoop(Node* node);
  void analyzeSubgraph(Node* node);
  void analyzeCreator(Node* node);
  void analyzeExtractor(Node* node);
  void analyzeChunk(Node* node);
  void analyzeBroadcastingChunk(Node* node);
  void analyzeFork(Node* node);
  void analyzeWait(Node* node);

  void makeAliasOf(const Value* value, const Value* to);
  void mapAliases(at::ArrayRef<Value*> to, at::ArrayRef<Value*> from);
  void giveFreshAlias(const Value* value);

  bool hasUsesAfter(Symbol alias, const Node* n) const;
  bool writesTo(Node* n, const Value* v) const;
  bool isBeforeSameGraph(const Node* lhs, const Node* rhs) const;

  std::shared_ptr<Graph> graph_;
  std::unordered_map<const Graph*, const Node*> subgraphToOwner_;
  std::unordered_set<const Node*> wildcardNodes_;
  std::unique_ptr<AliasTracker> aliasTracker_;
};
} // namespace jit
} // namespace torch
